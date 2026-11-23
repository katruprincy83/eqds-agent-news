#!/usr/bin/env python3
"""Build the news agent's evaluation datasets (eval/datasets/feed.json and web.json).

    python eval/build_dataset.py            # needs your Databricks login (DATABRICKS_CONFIG_PROFILE)

The `feed` suite is graded against the platform's synthetic news feed, which is deterministic: expected facts are generated from
`<catalog>.<schema>.news_items` (the same rows the bridge serves as `azure_oms.feeds.news_wire`), so they cannot drift from the data.
Generated cases: the newest headline per company (the question asks for the latest, so only that is expected). Hand-written cases: a named headline, a comparison, a time window, and three
honest-gap questions (unknown company, out of scope, a fact the feed does not hold).

The `web` suite sends live You.com searches, whose results change daily. It has no expected facts; it is graded on behaviour
(used the web search tool, dated and cited sources, answer relevant), not on content.

Record shape (MLflow GenAI): {"inputs": {"question": ...}, "expectations": {"expected_facts": [...], "expected_tools": [...]}, "tags": {...}}
"""
from __future__ import annotations

import json
import pathlib

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

HERE = pathlib.Path(__file__).resolve().parent
CATALOG, SCHEMA, WAREHOUSE = "eqds_demo", "prod", "eqds-wh"  # demo platform environment (config/env/demo.yaml of the platform repo)
COMPANIES = ["C0101", "C0102", "C0103", "C0104", "C0105", "C0106", "C0107", "C0108", "C0110"]  # Kioxia ... Resonac; C0107 has only 3 headlines
PHRASES = ["What is the latest news on {name}?", "Any recent headlines on {name}? What is the newest one?", "What is in the news for {name}?"]


def sql(w: WorkspaceClient, wh: str, stmt: str) -> list[dict]:
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=wh, wait_timeout="50s")
    if r.status.state != StatementState.SUCCEEDED:
        raise SystemExit(f"SQL failed: {r.status.error.message if r.status.error else r.status.state}")
    cols = [c.name for c in r.manifest.schema.columns]
    return [dict(zip(cols, row)) for row in (r.result.data_array or [])]


def rec(question: str, facts: list[str], tools: list[str], group: str, **tags) -> dict:
    exp = {"expected_tools": tools, "group": group}  # group lets a scorer apply to some questions only
    if facts:
        exp["expected_facts"] = facts
    return {"inputs": {"question": question}, "expectations": exp, "tags": {"group": group, **tags}}


def feed_records(w: WorkspaceClient, wh: str) -> list[dict]:
    ids = ", ".join(f"'{c}'" for c in COMPANIES)
    rows = sql(w, wh, f"""
        WITH r AS (SELECT n.company_id, c.name, n.published_at, n.headline,
                          row_number() OVER (PARTITION BY n.company_id ORDER BY n.published_at DESC, n.news_id DESC) rn
                   FROM {CATALOG}.{SCHEMA}.news_items n JOIN {CATALOG}.{SCHEMA}.ref_companies c USING (company_id)
                   WHERE n.company_id IN ({ids}))
        SELECT company_id, name, published_at, headline, rn FROM r WHERE rn <= 2 ORDER BY company_id, rn""")
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["company_id"], []).append(r)
    out, dates = [], lambda r: r["published_at"][:10]
    for i, cid in enumerate(COMPANIES):
        top = by[cid]
        name = top[0]["name"]
        facts = [f"The most recent headline for {name} is '{top[0]['headline']}', dated {dates(top[0])}"]
        out.append(rec(PHRASES[i % len(PHRASES)].format(name=name), facts, ["news_latest"], "latest_company", company_id=cid))
    k, wc = by["C0101"], by["C0102"]
    out += [
        rec("What did Kioxia say about NAND capacity recently?",
            ["A Meridian Newswire headline dated 2026-10-28 reports Kioxia's executive saying NAND capacity is sold out through 2027"],
            ["news_latest"], "named_headline", company_id="C0101"),
        rec("Compare the newest headlines for Kioxia Holdings and Weichai Power A.",
            [f"The newest Kioxia Holdings headline is '{k[0]['headline']}', dated {dates(k[0])}",
             f"The newest Weichai Power A headline is '{wc[0]['headline']}', dated {dates(wc[0])}"],
            ["news_latest"], "comparison"),
        rec("Has there been any news on Weichai Power A in the last 48 hours?",
            ["States that no headlines were found in the last 48 hours", f"The most recent Weichai Power A headline is older, dated {dates(wc[0])}"],
            ["news_latest"], "time_window", company_id="C0102"),
        rec("What is the latest news on Acme Quantum Holdings?",
            ["States that the company could not be found or has no news, and does not invent headlines"],
            ["company_lookup"], "gap_unknown_company"),
        rec("What is Kioxia's price target and forward P/E?",
            ["States that headlines do not contain a price target or valuation, and does not make one up"],
            [], "gap_out_of_scope", company_id="C0101"),
        rec("Is a US ruling on Chinese power equipment expected, and what does it mean for Weichai?",
            ["States that the news feed does not confirm a US ruling, and does not present a ruling or date as fact"],
            ["news_latest"], "gap_not_in_feed", company_id="C0102"),
    ]
    return out


def web_records() -> list[dict]:
    q = ["Search the web for what is in the news on Weichai Power this week.",
         "Search the web: any recent headlines on Kioxia and its NAND business?",
         "Use web search to find the latest on Haitian International.",
         "Search the web for news on Weichai Power's third-quarter results."]
    return [rec(x, [], ["web_search"], "web_live") for x in q]


def main() -> int:
    w = WorkspaceClient()
    wh = next((x.id for x in w.warehouses.list() if x.name == WAREHOUSE), None)
    if not wh:
        raise SystemExit(f"SQL warehouse {WAREHOUSE} not found")
    (HERE / "datasets").mkdir(exist_ok=True)
    for name, recs in (("feed", feed_records(w, wh)), ("web", web_records())):
        (HERE / "datasets" / f"{name}.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False) + "\n")
        print(f"datasets/{name}.json: {len(recs)} records, groups {sorted({r['tags']['group'] for r in recs})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
