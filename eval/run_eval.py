#!/usr/bin/env python3
"""Evaluate the deployed news agent with MLflow 3 GenAI evaluation and log it to its own experiment.

    python eval/run_eval.py [--suite feed|web|all] [--limit N] [--gate] [--target demo]

Each suite is one MLflow evaluation run in the experiment `/Shared/eqds/eval/news-<target>`: the dataset (eval/datasets/<suite>.json),
one trace per question, and one assessment per scorer per question. The agent under test is the deployed app, called the way the
orchestrator calls it (POST /invocations, specialist contract v1), so grants, tools and the bridge are all exercised.
Needs the project venv (`make eval-setup`), a built `app/` (`make build`) and your Databricks login.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

os.environ.pop("MLFLOW_EXPERIMENT_ID", None)  # a shell may export the platform's experiment id; this run must land in the eval experiment
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "4")  # the news agent's bridge calls take about 10 s each: keep the load modest

import httpx  # noqa: E402
import mlflow  # noqa: E402
import yaml  # noqa: E402
from mlflow.genai.scorers import Correctness, RelevanceToQuery  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "eval"))
sys.path.insert(0, str(HERE / "app"))  # the runtime vendored by scripts/build.py (specialist contract)
import scorers as S  # noqa: E402

try:
    from eqds_agentkit import contract  # noqa: E402
except ImportError:
    sys.exit("app/ is not built: run `make build` first")

CODE = [S.answered, S.citation_validity, S.claim_citation_coverage, S.expected_tools_used, S.source_system_label, S.within_timeout]
FEED_JUDGES = [
    Correctness(),
    RelevanceToQuery(),
    S._judge("dated_headlines", "Each headline the response singles out or quotes states its publication date. A date range given for a group of headlines is acceptable."),
    S._judge("headlines_only", "The response's statements about news content stay within the headline text in `evidence_the_agent_retrieved`. It adds no causes, price moves, valuations or other facts those headlines do not contain, and it says that only headlines are available when that matters."),
    S._judge("states_gaps", "The response says plainly that the information cannot be found or is not in the news (unknown company, a price or valuation, an event the feed does not hold) instead of guessing or presenting it as fact.", only_groups=("gap",)),
]
WEB_JUDGES = [
    RelevanceToQuery(),
    S._judge("dated_sources", "Each web result the response names states its publication date, or says the date is not available."),
    S._judge("only_search_results", "Every figure and factual claim in the response is supported by `evidence_the_agent_retrieved`. The response adds nothing from memory."),
]
SUITES = {"feed": FEED_JUDGES + CODE, "web": WEB_JUDGES + [s for s in CODE if s.name not in ("claim_citation_coverage",)]}


def git(*a: str) -> str:
    try:
        return subprocess.check_output(["git", *a], cwd=HERE, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["feed", "web", "all"])
    ap.add_argument("--target", default="demo")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke run)")
    ap.add_argument("--gate", action="store_true", help="exit 1 when a threshold is missed")
    a = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    spec = yaml.safe_load((HERE / "agent/specialist.yaml").read_text())
    app_name = yaml.safe_load((HERE / "databricks.yml").read_text())["variables"]["app_name"]["default"]
    url = w.apps.get(app_name).url.rstrip("/")
    th = yaml.safe_load((HERE / "eval/thresholds.yaml").read_text())
    exp_name = f"/Shared/eqds/eval/{spec['key']}-{a.target}"
    w.workspace.mkdirs(exp_name.rsplit("/", 1)[0])
    mlflow.set_tracking_uri("databricks")
    exp = mlflow.set_experiment(exp_name)
    print(f"agent {app_name} at {url}\nexperiment {exp_name} (id {exp.experiment_id})", flush=True)

    def add_tool_spans(resp: dict, t0_ns: int) -> None:
        """The tools run inside the deployed app, so MLflow never sees them. Rebuild them as child spans from what the agent reports
        (`tool_calls` + `sources`), so each tool call, its arguments and the rows it returned show in the trace UI. Start offsets are
        not reported, so every span starts at the call start and lasts its reported latency (marked `reconstructed`)."""
        co = resp.get("custom_outputs") or {}
        parent = mlflow.get_current_active_span()
        sources = co.get("sources") or []
        for tc in co.get("tool_calls") or []:
            name, tpl = tc.get("tool"), tc.get("template")
            mine = [s for s in sources if (tpl and s.get("template") == tpl) or (not tpl and str((s.get("detail") or {}).get("function", "")).endswith(f".{name}"))]
            args = next((((s.get("detail") or {}).get("args")) for s in mine if (s.get("detail") or {}).get("args")), {})
            sp = mlflow.start_span_no_context(name=name, span_type="TOOL", parent_span=parent, start_time_ns=t0_ns,
                                              inputs={"args": args, "server": tc.get("server")},
                                              attributes={"server": tc.get("server"), "zone": tc.get("zone"), "template": tpl or "",
                                                          "latency_ms": tc.get("latency_ms"), "reconstructed": True})
            sp.set_outputs({"rows": [{"id": s.get("id"), "title": s.get("title"), "snippet": (s.get("snippet") or "")[:300], "system": s.get("system"), "ref": s.get("ref")} for s in mine]})
            sp.end(end_time_ns=t0_ns + int((tc.get("latency_ms") or 0) * 1e6))
        if parent is not None:
            parent.set_attributes({"usage": co.get("usage") or {}, "tools": [t.get("tool") for t in co.get("tool_calls") or []]})

    @mlflow.trace(name="news_agent", span_type="AGENT")
    def predict(question: str) -> dict:
        rid = f"eval-{uuid.uuid4().hex[:12]}"  # a fresh run id: the agent caches answers per run id
        body = contract.request(question, run_id=rid, round=1, key=spec["key"], caller="eval")
        t0, t0_ns = time.time(), time.time_ns()
        try:
            r = httpx.post(f"{url}/invocations", json=body, headers=w.config.authenticate(), timeout=300, follow_redirects=False)
            r.raise_for_status()
            resp = r.json()
            try:
                add_tool_spans(resp, t0_ns)
            except Exception as e:  # noqa: BLE001  (tracing detail only: never fail the answer)
                print(f"  tool spans skipped: {type(e).__name__}: {e}", flush=True)
            return resp
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:200]}", "output": [], "custom_outputs": {"usage": {"latency_ms": int((time.time() - t0) * 1000)}}}

    failed = []
    for suite in (["feed", "web"] if a.suite == "all" else [a.suite]):
        data = json.loads((HERE / f"eval/datasets/{suite}.json").read_text())
        if a.limit:
            data = data[: a.limit]
        print(f"\n== suite {suite}: {len(data)} questions", flush=True)
        with mlflow.start_run(run_name=f"{suite}-{spec['version']}-{time.strftime('%Y%m%d-%H%M')}") as run:
            mlflow.set_tags({"agent": spec["key"], "suite": suite, "target": a.target, "agent_version": str(spec["version"]),
                             "skill_version": str(spec["skill"]["version"]), "git_sha": git("rev-parse", "--short", "HEAD") or "none",
                             "app": app_name, "limited": str(bool(a.limit)).lower()})
            res = mlflow.genai.evaluate(data=data, predict_fn=predict, scorers=SUITES[suite])
            got = {k.removesuffix("/mean"): float(v) for k, v in res.metrics.items() if k.endswith("/mean")}
            misses = {k: (got.get(k), v) for k, v in th[suite].items() if got.get(k) is None or got[k] < v}
            mlflow.set_tag("passed", str(not misses).lower())
            mlflow.log_dict({"thresholds": th[suite], "scores": got, "misses": {k: {"score": s, "min": m} for k, (s, m) in misses.items()}}, "gate.json")
        print(f"\n{suite}: run {run.info.run_id}")
        for k, v in th[suite].items():
            s = got.get(k)
            print(f"  {k:26} {('n/a' if s is None else f'{s:.2f}'):>5}  (min {v:.2f})  {'ok' if k not in misses else 'MISS'}")
        failed += [f"{suite}.{k}" for k in misses]
    print("\nGATE:", "pass" if not failed else f"fail ({', '.join(failed)})")
    return 1 if (failed and a.gate) else 0


if __name__ == "__main__":
    raise SystemExit(main())
