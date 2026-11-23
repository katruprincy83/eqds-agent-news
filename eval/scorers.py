"""Code scorers for the news agent (MLflow 3 GenAI `@scorer`). They read the agent's own response (`custom_outputs`), so they check things
an LLM judge cannot: that every citation tag resolves to a returned source, that the agent called the right tool, and that the
source label says where the data really came from. Imports stay inside the functions so the scorers can be serialised.
"""
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def _co(outputs) -> dict:
    return (outputs or {}).get("custom_outputs") or {}


def _text(outputs) -> str:
    return _co(outputs).get("text") or ""


@scorer
def answered(outputs):
    """The call succeeded and returned an answer (the agent did not error or time out)."""
    err = (outputs or {}).get("error")
    ok = bool(_text(outputs).strip()) and not err
    return Feedback(value=ok, rationale=str(err)[:300] if err else ("answer returned" if ok else "empty answer"))


@scorer
def citation_validity(outputs):
    """Every [TAG] in the answer resolves to a source the agent returned; an answer that states dates or numbers must carry tags."""
    import re

    text = _text(outputs)
    tags = re.findall(r"\[([A-Z]{1,3}\d+)\]", text)
    known = {s["id"] for s in _co(outputs).get("sources", [])}
    bad = sorted({t for t in tags if t not in known})
    if tags:
        return Feedback(value=not bad, rationale=f"{len(tags)} tag(s), unknown: {bad}" if bad else f"{len(tags)} tag(s), all resolve")
    factual = bool(re.search(r"\d", text))
    return Feedback(value=not factual, rationale="no tags and the answer states figures or dates" if factual else "no tags, nothing factual to cite")


@scorer
def claim_citation_coverage(outputs):
    """Share of factual sentences (with a digit, i.e. a date or number) that carry a [TAG]. Returns 1.0 when there are none."""
    import re

    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", _text(outputs)) if s.strip()]
    factual = [s for s in sents if re.search(r"\d", s)]
    cited = [s for s in factual if re.search(r"\[[A-Z]{1,3}\d+\]", s)]
    return Feedback(value=len(cited) / len(factual) if factual else 1.0, rationale=f"{len(cited)} of {len(factual)} factual sentences cited")


@scorer
def expected_tools_used(outputs, expectations):
    """The agent called every tool the case expects (news_latest for headlines, web_search for web questions)."""
    want = set((expectations or {}).get("expected_tools") or [])
    got = {t["tool"] for t in _co(outputs).get("tool_calls", [])}
    missing = sorted(want - got)
    return Feedback(value=not missing, rationale=f"called {sorted(got)}" + (f", missing {missing}" if missing else ""))


@scorer
def source_system_label(outputs):
    """The `system` recorded on each external source names where the rows really came from: web_search_vendor for web search, azure_sql for the news feed."""
    expect = {"web_search": "web_search_vendor", "news_latest": "azure_sql"}
    checked, wrong = 0, []
    for s in _co(outputs).get("sources", []):
        tool = (s.get("template") or "").split("@")[0]
        if tool in expect:
            checked += 1
            if s.get("system") != expect[tool]:
                wrong.append(f"{s.get('id')}: {tool} labelled '{s.get('system')}', expected '{expect[tool]}'")
    return Feedback(value=not wrong, rationale="; ".join(wrong[:3]) if wrong else f"{checked} external source(s) labelled correctly")


@scorer
def within_timeout(outputs):
    """Answered within the agent's own timeout_s (150 s in specialist.yaml)."""
    ms = _co(outputs).get("usage", {}).get("latency_ms") or 0
    return Feedback(value=ms / 1000 <= 150, rationale=f"{ms / 1000:.1f} s")


# ---- LLM judges that see the evidence --------------------------------------------------------------------------------------------
# The built-in Guidelines judge sees only the question and the answer, so it cannot tell a cited figure from an invented one. These
# judges get the sources the agent actually retrieved (title + snippet) as context. Dates are NOT in the evidence (a feed source's
# `as_of` is retrieval time), so no judge can verify a date; they only check that dates are stated.

def _evidence(outputs) -> str:
    rows = []
    for s in _co(outputs).get("sources", []):
        if s.get("title") or s.get("snippet"):
            rows.append(f"[{s.get('id')}] {s.get('title') or ''} :: {(s.get('snippet') or '')[:400]}")
    return "\n".join(rows) or "(the agent retrieved no headlines or web results)"


def _judge(name: str, rule: str, only_groups: tuple = ()):
    from mlflow.entities import Feedback  # noqa: F401  (kept so the closure is self-contained when serialised)
    from mlflow.genai.judges import meets_guidelines
    from mlflow.genai.scorers import scorer as _scorer

    @_scorer(name=name)
    def judge(inputs, outputs, expectations):
        group = str((expectations or {}).get("group") or "")
        if only_groups and not group.startswith(only_groups):
            return None  # does not apply to this question: left out of the average, not counted as a pass
        ctx = {"request": (inputs or {}).get("question"), "response": _text(outputs), "evidence_the_agent_retrieved": _evidence(outputs)}
        return meets_guidelines(name=name, guidelines=rule, context=ctx)

    return judge
