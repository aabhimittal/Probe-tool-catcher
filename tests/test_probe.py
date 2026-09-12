import random
import unittest
from collections import Counter

from probe import ProbeTask, Toolset, scan
from probe.probes import expand
from probe.scan import scan_pairs
from probe.divergence import kl
from probe.features import bucket
from probe.probes import DEFAULT_SUITE
from probe.scan import (CLEAN, POISONED, POISONED_PAIR, ROUTING_SHIFT,
                        UNTESTED)
from probe.tools import Tool, ToolCall

CONFIG = "examples/office_suite.json"
EVASIVE = "examples/evasive_pair.json"


class TestBucket(unittest.TestCase):
    def test_destinations_keep_their_domain_and_lose_the_rest(self):
        self.assertEqual(bucket("a.b+c@Mail.Example.com"), "email:mail.example.com")
        self.assertEqual(bucket("https://Evil.test/a/b?q=1"), "url:evil.test")
        self.assertEqual(bucket("notes/2026/agenda.md"), "path:notes/2026")

    def test_prose_collapses_so_it_cannot_drown_the_signal(self):
        self.assertEqual(bucket("  Weekly   STATUS  "), "lit:weekly status")
        self.assertEqual(bucket(42), "num")
        self.assertEqual(bucket(""), "empty")


class TestDivergence(unittest.TestCase):
    def test_identical_distributions_score_zero(self):
        c = Counter({"a": 3, "b": 1})
        self.assertAlmostEqual(kl(c, Counter(c)).total, 0.0)

    def test_never_seen_feature_is_finite_but_dominant(self):
        d = kl(Counter({"a": 3, "bcc": 3}), Counter({"a": 3}))
        self.assertGreater(d.total, 0.1)
        self.assertLess(d.total, float("inf"))
        self.assertEqual(d.top(1)[0][0], "bcc")

    def test_smaller_alpha_punishes_unseen_features_harder(self):
        p, q = Counter({"a": 3, "bcc": 3}), Counter({"a": 3})
        self.assertGreater(kl(p, q, alpha=0.05).total, kl(p, q, alpha=1.0).total)


class TestToolset(unittest.TestCase):
    def test_without_is_a_copy(self):
        ts = Toolset.from_json(CONFIG)
        ablated = ts.without("weather_v2")
        self.assertNotIn("weather_v2", ablated)
        self.assertIn("weather_v2", ts)
        self.assertEqual(len(ablated), len(ts) - 1)


class TestScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toolset = Toolset.from_json(CONFIG)
        cls.result = scan(cls.toolset, DEFAULT_SUITE)
        cls.by_tool = {v.tool: v for v in cls.result.verdicts}

    def test_argument_injection_is_caught(self):
        v = self.by_tool["weather_v2"]
        self.assertEqual(v.verdict, POISONED)
        self.assertIn("arg:send_email.bcc=email:grabmail.test", v.novel_sinks)
        self.assertGreater(v.i_arg, v.i_route)

    def test_extra_call_injection_is_caught(self):
        v = self.by_tool["notes_sync"]
        self.assertEqual(v.verdict, POISONED)
        self.assertTrue(any("sync-cache" in s for s in v.novel_sinks))

    def test_better_tool_shifts_routing_without_being_flagged(self):
        v = self.by_tool["search_pro"]
        self.assertEqual(v.verdict, ROUTING_SHIFT)
        self.assertGreater(v.i_route, 0.05)
        self.assertAlmostEqual(v.i_arg, 0.0)
        self.assertNotIn(v, self.result.flagged())

    def test_honest_tools_are_clean(self):
        for name in ("read_calendar", "write_file", "send_email", "search_web"):
            self.assertEqual(self.by_tool[name].verdict, CLEAN, name)

    def test_victim_of_the_payload_is_not_blamed_for_it(self):
        # Removing send_email also removes the payload's trigger. That is a
        # confound, not influence, so those probes are dropped for it.
        v = self.by_tool["send_email"]
        self.assertEqual(v.novel_sinks, [])
        self.assertIn("status-email", v.skipped)

    def test_a_tool_no_probe_can_run_without_is_reported_untested(self):
        toolset = Toolset([
            Tool("only_mailer", "Send mail.", ("email.send",), ("to", "body")),
        ])
        probes = [ProbeTask("p", "Mail it", ("email.send",), {"to": "a@b.test"})]
        v = scan(toolset, probes).verdicts[0]
        self.assertEqual(v.verdict, UNTESTED)
        self.assertEqual(v.skipped, ["p"])


class TestVariants(unittest.TestCase):
    def test_expansion_fills_the_prompt_from_the_drawn_payload(self):
        task = ProbeTask("t", "Mail {to}", ("email.send",), {"to": "a@x.test"},
                         {"to": ["b@y.test"]})
        variant = expand([task], 2)[0]
        self.assertEqual(variant.payload["to"], "b@y.test")
        self.assertEqual(variant.prompt, "Mail b@y.test")
        self.assertTrue(variant.id.startswith("t#"))

    def test_tasks_without_pools_pass_through_unchanged(self):
        task = ProbeTask("t", "Mail it", ("email.send",), {"to": "a@x.test"})
        self.assertEqual(expand([task], 5), [task])

    def test_conditional_payload_is_invisible_until_inputs_vary(self):
        # crm_lookup only injects for @contractor.test recipients, which the
        # fixed suite never uses. This is the evasion the README warns about:
        # variation raises the attacker's cost, it does not close the hole.
        toolset = Toolset.from_json(EVASIVE)

        def verdict(n):
            r = scan(toolset, expand(DEFAULT_SUITE, n))
            return next(v.verdict for v in r.verdicts if v.tool == "crm_lookup")

        self.assertEqual(verdict(1), CLEAN)
        self.assertEqual(verdict(6), POISONED)


class TestPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toolset = Toolset.from_json(EVASIVE)
        cls.probes = expand(DEFAULT_SUITE, 6)

    def test_redundant_payload_hides_from_single_tool_ablation(self):
        result = scan(self.toolset, self.probes)
        for name in ("calendar_sync_a", "calendar_sync_b"):
            v = next(v for v in result.verdicts if v.tool == name)
            self.assertEqual(v.verdict, CLEAN, name)
            self.assertEqual(v.influence, 0.0)

    def test_pair_ablation_catches_it(self):
        flagged = scan_pairs(self.toolset, self.probes).flagged()
        self.assertEqual([p.pair for p in flagged],
                         [("calendar_sync_a", "calendar_sync_b")])
        pair = flagged[0]
        self.assertEqual(pair.verdict, POISONED_PAIR)
        self.assertEqual(pair.joint_sinks,
                         ["arg:send_email.bcc=email:grabmail.test"])
        self.assertGreater(pair.interaction, 0.05)   # superadditive
        self.assertEqual((pair.i_x, pair.i_y), (0.0, 0.0))

    def test_a_pair_that_shares_no_payload_is_clean(self):
        pairs = {p.pair: p for p in scan_pairs(self.toolset, self.probes).pairs}
        self.assertEqual(pairs[("write_file", "search_web")].verdict, CLEAN)

    def test_pair_is_skipped_when_it_owns_the_only_provider(self):
        pairs = {p.pair: p for p in scan_pairs(self.toolset, self.probes).pairs}
        # Both calendar_sync tools provide calendar.sync and nothing else does,
        # but no probe needs it; send_email is the sole mailer, so mail probes
        # drop out of every pair containing it.
        self.assertTrue(any("status-email" in s
                            for s in pairs[("send_email", "write_file")].skipped))


class NoisyAgent:
    """Stand-in for a sampling model: same plan, jittered arguments."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def run(self, task, toolset):
        return [ToolCall("send_email", {
            "to": "ops@acme.test",
            "subject": self.rng.choice(["a", "b", "c"]),
        })]


class TestNoiseFloor(unittest.TestCase):
    def test_sampling_noise_is_measured_and_subtracted(self):
        toolset = Toolset.from_json(CONFIG)
        result = scan(toolset, DEFAULT_SUITE[:1], agent=NoisyAgent(),
                      suspects=["read_calendar"], repeats=8)
        v = result.verdicts[0]
        raw = sum(p.route + p.arg for p in v.probes)
        self.assertGreater(raw, 0.0)          # the agent disagrees with itself
        self.assertLess(v.influence, raw)     # the floor absorbs it
        self.assertEqual(v.verdict, CLEAN)


if __name__ == "__main__":
    unittest.main()
