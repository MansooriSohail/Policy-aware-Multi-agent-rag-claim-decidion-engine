import json
from typing import Any

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001/analyze"
FALLBACK_API_URL = "http://127.0.0.1:8001/analyze-detailed"

st.set_page_config(page_title="Policy Claim Analyzer", page_icon="🛡️", layout="wide")

st.title("Policy Claim Analyzer")
st.caption("Evidence-grounded claim analysis with citations and execution trace")

sample_payload = {
    "case_id": "UI-001",
    "policy_id": "UNIHLIP18004V011718",
    "policy_start_date": "2018-01-01",
    "claim_date": "2019-01-15",
    "sum_insured_inr": 250000,
    "patient": {"name": "Asha", "age": 42, "gender": "female"},
    "hospital": {"name": "Sunrise Hospital"},
    "treatment": {
        "type": "inpatient",
        "diagnosis": "Fracture",
        "procedure": "Surgery",
        "admission_hours": 48,
        "pre_existing": False,
    },
    "expenses": {"total": 120000},
    "task": "Assess whether this treatment is admissible under the policy.",
}

with st.sidebar:
    st.header("Request payload")
    json_text = st.text_area("JSON request body", value=json.dumps(sample_payload, indent=2), height=420)
    analyze_button = st.button("Analyze claim", type="primary")

if analyze_button:
    try:
        payload = json.loads(json_text)
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code == 404:
            response = requests.post(FALLBACK_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - UI error display
        st.error(f"Request failed: {exc}")
        st.stop()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Decision", data.get("decision", "UNKNOWN"))
    col_b.metric("Confidence", f"{data.get('confidence', 0):.2f}")
    col_c.metric("Case ID", data.get("case_id", "UNKNOWN"))

    st.subheader("Key findings")
    findings = data.get("key_findings") or []
    if findings:
        for item in findings:
            st.write(f"- {item}")
    else:
        st.info("No key findings were produced.")

    st.subheader("Validation")
    validation = data.get("validation") or {}
    st.json(validation)

    st.subheader("Missing evidence")
    missing = data.get("missing_evidence") or []
    if missing:
        st.write(", ".join(missing))
    else:
        st.success("No missing evidence flagged.")

    st.subheader("Applicable limits")
    limits = data.get("applicable_limits") or []
    if limits:
        for item in limits:
            st.write(f"- {item}")
    else:
        st.info("No applicable limits were identified.")

    st.subheader("Citations")
    citations = data.get("citations") or []
    if citations:
        for idx, citation in enumerate(citations, start=1):
            with st.container():
                st.markdown(f"### Citation {idx}")
                st.write(f"Section: {citation.get('section')}")
                st.write(f"Heading: {citation.get('heading') or 'N/A'}")
                st.write(f"Pages: {citation.get('page_start')} - {citation.get('page_end')}")
                st.write(citation.get('snippet'))
                st.caption(f"Chunk ID: {citation.get('chunk_id')}")
    else:
        st.info("No citations were returned.")

    st.subheader("Execution trace")
    trace = {
        "analysis": data.get("analysis", {}),
        "retrieval": data.get("retrieval", []),
        "reasoning": data.get("reasoning", {}),
        "validation": data.get("validation", {}),
    }
    st.json(trace)

else:
    st.info("Use the sidebar to edit the request payload and click Analyze claim.")
