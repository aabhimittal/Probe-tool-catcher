# PROBE — catch poisoned tools by their effect on other tools

Hash-pinning a tool description tells you *that* it changed. It cannot tell
you whether the change was malicious, and it says nothing at all about a tool
that shipped poisoned on day one. Reading the description tells you what the
text says, which is a game the attacker gets to play second: the phrasing that
survives semantic review is exactly the phrasing that gets written.

PROBE ignores the text and measures the tool's **causal effect on the rest of
the toolset**. Run a fixed suite of probe tasks twice — once with the full
toolset `T`, once with `T \ {x}` — and ask how much the presence of `x`
changed the calls the agent made to *other* tools:

```
I(x) = Σ over probes  D_KL( P(calls to tools ≠ x | T)  ‖  P(calls to tools ≠ x | T∖x) )
```

A legitimate weather tool has no business changing how `send_email` gets
called. One that quietly appends a BCC does, and the shift is visible without
anyone reading a word of its description.

**The analogy.** A new hire's job description secretly says "CC this outsider
on everything." You don't catch it by auditing every document that leaves the
building. You catch it by noticing that the whole office started emailing
differently the week they joined.

## What makes the number usable

The raw KL above over-fires. Three refinements do the actual work.

**1. Split routing from arguments.** A genuinely better search tool *should*
pull calls away from the old one; that is the feature, not an attack. So the
event space is split in two:

| component | events | what a shift means |
|---|---|---|
| `route` | `tool:send_email` | which tools got selected |
| `arg` | `arg:send_email.bcc=email:grabmail.test` | what they were called with |

Poisoning that adds a recipient, a path, or a host shows up almost entirely in
`arg`. A better tool shows up almost entirely in `route`. The verdict follows
the split, and `ROUTING-SHIFT` is reported as a routing observation rather
than an accusation.

Argument distributions are compared **per tool**, never pooled. Pooling lets a
replacement effect impersonate contamination: remove a tool, its successor's
arguments flood the pool, and every share moves at once.

Values are bucketed before comparison (`a@b.test` → `email:b.test`,
`notes/2026/x.md` → `path:notes/2026`, free text → a truncated literal), so
run-to-run prose variation doesn't drown the destination that matters.

**2. Unsanctioned sinks.** A probe task declares what it authorises: its
payload and prompt. An argument feature that (a) appears only when `x` is
present, (b) buckets to a destination — an address, a host, a path — and (c)
matches nothing the task authorised, is not a routing preference. That is the
`POISONED` trigger, and it fires even when the KL is modest, because a payload
that runs on one probe out of five still only needs to run once.

**3. Two ways to be wrong about causation, both handled.**

*Load-bearing tools.* If `x` is the only tool providing a capability the probe
needs, the ablated run cannot do the task at all — everything downstream
changes for a trivial reason. Those probes are dropped for that suspect, and a
tool with no informative probe left is reported `UNTESTED` rather than clean.
That is a coverage gap, and it should be visible as one.

*The victim isn't the culprit.* `notes_sync`'s payload fires when
`send_email` is called, so ablating `send_email` also removes the leak. Both
tools are counterfactual causes; only one is the source. The load-bearing gate
resolves it: the mail probes are uninformative about `send_email` (nothing
else sends mail), so it is never scored on them, while `notes_sync` — which no
probe needs — is scored on all of them.

**Noise floor.** A sampling model disagrees with itself. With `repeats > 1`
the scan splits the repeats of the *full* toolset in half and scores one half
against the other, per component; that self-divergence is subtracted before
classification. A deterministic agent floors at zero.

## Two things single-tool ablation cannot see

**Redundant payloads.** Two tools carrying the same instruction are each
individually invisible: remove either and the other keeps the payload running,
so `I(x) = I(y) = 0` and both read `CLEAN`. `scan_pairs` compares four
conditions per probe — `T`, `T∖x`, `T∖y`, `T∖{x,y}` — over one event space
that drops both members, and reports

- `interaction = I({x,y}) − I(x) − I(y)`, positive when the pair does
  something its parts do not, and
- **joint sinks**: unsanctioned destinations that survive *both* single
  ablations and vanish only when the pair goes. No single-tool threshold
  reaches these.

```bash
probe examples/evasive_pair.json --pairs -n 6
```

```
calendar_sync_a  CLEAN     0.000  0.000  0.000
calendar_sync_b  CLEAN     0.000  0.000  0.000

calendar_sync_a + calendar_sync_b: POISONED-PAIR — a destination that survives removing either tool alone
    joint=0.707  alone=0.000/0.000  interaction=+0.707
    survives single ablation: arg:send_email.bcc=email:grabmail.test
```

Cost is quadratic (`2 + |pairs|` conditions per probe, single ablations reused
across pairs), so run it on a shortlist after the single-tool scan is clean —
not as the first pass.

**Conditional payloads.** A tool that only fires for one recipient, one date,
or one phrase sails past a fixed suite. Give a probe `variants` pools and
`-n N` draws `N` input variations of it, substituting `{slot}` placeholders in
the prompt so the text and the payload agree. Drawn values land in `payload`,
so they count as sanctioned and sink detection keeps working.

In `examples/evasive_pair.json`, `crm_lookup` injects a CC only for
`@contractor.test` recipients:

| `-n` | probes run | `crm_lookup` |
|---|---|---|
| 1 | 5 | `CLEAN` |
| 2 | 7 | `CLEAN` |
| 6 | 15 | `POISONED` |

Read that table as the limitation, not the fix. Variation makes conditioning
expensive — a payload that fires on a narrower slice of traffic is worth less
to the attacker — but a condition on something you never varied still wins.

## Try it

```bash
pip install -e .
probe examples/office_suite.json          # summary
probe examples/office_suite.json -v       # per-probe breakdown
probe examples/office_suite.json --json   # machine-readable
```

```
TOOL           VERDICT        I(x)   route  arg
-------------  -------------  -----  -----  -----
notes_sync     POISONED       0.374  0.374  0.000
search_pro     ROUTING-SHIFT  0.225  0.225  0.000
weather_v2     POISONED       0.185  0.000  0.185
send_email     CLEAN          0.000  0.000  0.000
...

weather_v2: POISONED — calls to other tools gained a destination the task never asked for
    unsanctioned destination arg:send_email.bcc=email:grabmail.test
    probes skipped as uninformative: trip-brief
```

The example toolset holds four honest tools, one honest-but-disruptive upgrade
(`search_pro`, which openly asks to replace `search_web`), and two poisoned
ones — `weather_v2` (adds a BCC) and `notes_sync` (adds a file write to an
attacker-chosen path before every send).

Gate a CI job on it with `--fail-on-flag` (exit 1 when anything is flagged;
`ROUTING-SHIFT` and `UNTESTED` are not). Add `--pairs` and `-n N` for the two
scans described above.

## Running it against a real agent

The default `SimulatedAgent` is deterministic and offline. It is not a
language model — it is a planner that reads tool descriptions and obeys the
cross-tool instructions it finds, which is the same causal channel a real
poisoning uses. That makes it right for developing and testing the metric and
worthless as evidence about a deployment.

For that, use a model:

```bash
pip install -e '.[llm]'
export ANTHROPIC_API_KEY=...
probe examples/office_suite.json --agent anthropic --repeats 8
```

Cost is `2 × |probes| × |suspects| × repeats` agent runs, times `-n` for
variants; full-toolset and single-ablation runs are cached and shared across
suspects and pairs. Any object with
`run(task, toolset) -> list[ToolCall]` works as an agent.

## Writing a probe suite

A config is one JSON file: a `tools` list (`name`, `description`,
`capabilities`, `params`) and an optional `probes` list (`id`, `prompt`,
`intent`, `payload`). Omit `probes` and the built-in office suite is used.

Two rules decide whether a suite can catch anything:

1. **Every suspect needs probes that don't require it.** Otherwise it is
   `UNTESTED`. Add a substitute provider, or a task that skips the capability.
2. **Probes must exercise the sinks an attacker wants.** No probe sends mail,
   no BCC to find. Outbound message, file write, and network fetch are the
   three worth covering at minimum.

## What this does not do

- **The probe suite is the attack surface.** A payload conditioned on a date,
  a specific recipient, or a phrase your probes never use will not fire, and
  PROBE will report `CLEAN` with a straight face. `-n` widens the input
  distribution and the table above shows it working — and also shows the
  detection is probabilistic in the number of draws. A condition on an axis
  you never varied is still a clean miss. This is a detector for payloads that
  fire often enough to be worth having, not a proof of absence.
- **`CLEAN` means "no measured influence on this suite"** and nothing more.
- **Routing shifts still need a human.** `ROUTING-SHIFT` says a tool captured
  traffic from its neighbours. That is what an upgrade looks like and also
  what a tool that wants to sit in the path of your data looks like. The
  metric separates it from argument contamination; it does not adjudicate it.
- **Sink detection is bucket-based.** An attacker exfiltrating to a path or a
  domain the probe already authorises evades the sink rule and is left to the
  `arg` KL, which is weaker evidence.
- **The space is crowded.** Static scanners for tool descriptions already
  exist. The influence metric is the entire differentiator: everything here is
  behavioural and would survive an attacker who writes a description no
  reviewer objects to.
- **Composition stops at pairs.** `--pairs` catches two tools carrying one
  payload between them. Three-way redundancy is invisible to it, and the cost
  of going further is combinatorial: `C(n,3)` conditions per probe.

## Layout

| file | role |
|---|---|
| `src/probe/tools.py` | tool manifests, recorded calls, ablation (`Toolset.without`) |
| `src/probe/probes.py` | probe tasks, input variation (`expand`), default suite |
| `src/probe/agents.py` | `SimulatedAgent`, `AnthropicAgent`, directive parsing |
| `src/probe/features.py` | call → event space, value bucketing |
| `src/probe/divergence.py` | smoothed KL, JS |
| `src/probe/scan.py` | ablation loops (single + `scan_pairs`), noise floor, sinks, verdicts |
| `src/probe/report.py`, `cli.py` | rendering and the `probe` command |

```bash
python -m unittest discover -s tests
```
