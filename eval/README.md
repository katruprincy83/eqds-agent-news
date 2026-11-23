# News agent evaluation (MLflow)

Evaluates the **deployed** news agent end to end: each question goes through the same `/invocations` contract the moderator uses,
so tool calls, the bridge MCP, You.com and the answer text are all in scope. Results land in the MLflow experiment
`/Shared/eqds/eval/news-<target>` (default target `demo`), one MLflow run per suite per evaluation, with traces per question.

```
make eval-setup      # once: .venv with mlflow 3.18 + databricks-agents (the default judge needs it)
make eval-dataset    # rebuild datasets/*.json from eqds_demo.prod.news_items (needs warehouse access)
make eval            # build + run both suites          (ARGS="--suite feed|web|all --limit N --gate")
```

Set `DATABRICKS_CONFIG_PROFILE` (e.g. `eqds-demo`) first. `--gate` exits non-zero when any score is below its threshold, for CI.

## Suites

| Suite | Questions | What it covers |
|---|---|---|
| `feed` | 15, ground truth computed from the Delta table | latest headlines per company (9), a named headline, a comparison, a time window, and 3 *gap* questions (unknown company, out-of-scope, company not in feed) where the right answer is to say so |
| `web` | 4 live questions | `web_search` through the bridge MCP to You.com; no fixed ground truth, so it checks form and sourcing, not facts |

Each trace has an `AGENT` span for the call and a `TOOL` child span per tool call (arguments, rows returned, reported latency). The tools run
inside the deployed app, so these spans are **rebuilt from the agent's own `tool_calls` and `sources`**; the agent reports no start
offsets, so every tool span starts at the call start and lasts its reported latency (`reconstructed: true`).

Each record is `{"inputs": {"question"}, "expectations": {"expected_facts", "expected_tools"}, "tags": {...}}`.

## Scorers

| Scorer | Kind | Meaning |
|---|---|---|
| `answered` | code | contract status ok and non-empty answer |
| `correctness` | LLM judge (built-in) | answer contains the expected facts (feed only) |
| `relevance_to_query` | LLM judge (built-in) | answer addresses the question |
| `dated_headlines`, `headlines_only`, `states_gaps`, `dated_sources`, `only_search_results` | LLM judge (Guidelines) | every headline carries a date; no invented content beyond headlines; gaps are stated, not filled; sources dated; only search results used |
| `citation_validity` | code | every citation points at a tool call that actually ran |
| `claim_citation_coverage` | code | share of claims that carry a citation |
| `expected_tools_used` | code | the tools named in `expected_tools` were called |
| `source_system_label` | code | citation `system` matches the tool (`web_search` → `web_search_vendor`, `news_latest` → `azure_sql`) |
| `within_timeout` | code | answered inside 150 s |

Minimum scores are in `thresholds.yaml`. They are a starting point and have not been tuned against history.

## Baseline (2026-11-23, agent 0.8.0) and how to read it

Both suites miss their thresholds. That is the starting point, not a regression:

| Miss | Cause |
|---|---|
| `source_system_label` (feed 0.73, web 0.00) | **real bug**: web-search sources are labelled `azure_sql` (tools.py) |
| `dated_headlines`, `headlines_only` | judges are stricter than the guideline text (they ask for a date per headline and flag sentiment summaries such as "mixed"); partly real, partly judge noise. Re-running moves them by a few points |
| `correctness`, `states_gaps` | partly the dataset: `gap_out_of_scope` asks for a price target, but the agent has web search and cites a real JPMorgan target, so "not available" is no longer the right answer |
| not scored | dates cannot be verified: a feed source carries retrieval time (`as_of`), not publication date |

Other finding: for one question the agent calls `news_latest` twice with identical arguments (about 10 s each).
LLM-judge scores vary between runs; rely on code scorers for gating and read the judge rationales in the UI.

## Reading a run

Each run is tagged with `agent`, `suite`, `target`, `version`s, `git_sha`, `limited` (true for `--limit` smoke runs) and `passed`,
and carries `gate.json`. Compare runs in the experiment's *Evaluation runs* tab. After changing the skill or tools, rerun and compare.
