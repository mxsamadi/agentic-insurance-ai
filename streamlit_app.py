
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from guardrails.authentication import require_authentication
from initialize_platform import initialize_platform


st.set_page_config(
    page_title="Agentic AI Insurance Platform",
    page_icon="🛡️",
    layout="wide"
)


# Authentication
if not require_authentication():
    st.stop()


# Configure OpenAI credentials
if "OPENAI_API_KEY" in st.secrets:

    os.environ["OPENAI_API_KEY"] = str(
        st.secrets["OPENAI_API_KEY"]
    )


# Automatic platform initialization
@st.cache_resource(show_spinner=False)
def prepare_platform():

    return initialize_platform()


try:

    with st.spinner("Initializing AI platform..."):

        prepare_platform()

except Exception:

    st.error(
        "Platform initialization failed. "
        "Please check the application configuration."
    )

    st.stop()


from api import investigate, InvestigationRequest
from tools.data_utils import load_claims




st.title("🛡️ Agentic AI Insurance Platform")

st.markdown(
    "### Intelligent Risk Assessment & Fraud Investigation"
)

st.caption(
    "LangGraph | XGBoost | Isolation Forest | "
    "OpenAI | RAG | ChromaDB"
)

st.info(
    "Portfolio demonstration using synthetic insurance data. "
    "Risk scores and anomalies are not proof of fraud."
)


# Sidebar
with st.sidebar:

    st.header("Investigation Settings")

    claims = load_claims()

    claim_id = st.selectbox(
        "Select Claim",
        claims["claim_id"].tolist()
    )

    investigation_type = st.selectbox(
        "Investigation Type",
        [
            "Comprehensive Investigation",
            "Fraud Risk Assessment",
            "Anomaly Detection",
            "Claim Risk Indicators"
        ]
    )

    questions = {
        "Comprehensive Investigation":
            f"Conduct a comprehensive fraud investigation for {claim_id}.",

        "Fraud Risk Assessment":
            f"Assess the fraud risk score of {claim_id}.",

        "Anomaly Detection":
            f"Detect anomalies for claim {claim_id}.",

        "Claim Risk Indicators":
            f"Identify risk indicators for {claim_id}."
    }

    question = st.text_area(
        "Investigation Request",
        value=questions[investigation_type],
        max_chars=2000
    )

    run_button = st.button(
        "Run Investigation",
        type="primary",
        width="stretch"
    )

    st.divider()

    st.subheader("Platform Components")

    st.markdown("""
    - Supervisor Agent
    - Risk Agent
    - Anomaly Agent
    - Investigation Agent
    - Validation Agent
    - RAG Knowledge Base
    - AI Guardrails
    """)


# Dataset overview
st.subheader("Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Synthetic Claims",
    f"{len(claims):,}"
)

col2.metric(
    "Total Claim Amount",
    f"${claims['claim_amount'].sum():,.0f}"
)

col3.metric("AI Agents", "5")

col4.metric("ML Models", "2")


# Execute investigation
if run_button:

    st.session_state.pop("result", None)

    with st.spinner(
        "AI agents are analyzing the claim. Please wait..."
    ):

        try:

            request = InvestigationRequest(
                question=question
            )

            response = investigate(request)

            st.session_state["result"] = (
                response.model_dump()
            )

        except Exception:

            st.error(
                "Investigation failed. Verify the claim ID "
                "and application configuration."
            )


# Display investigation results
if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.header(
        f"Investigation Results: {result['claim_id']}"
    )

    if result["validation_passed"]:

        st.success("Validation passed")

    else:

        st.error(
            "Validation failed. Investigation withheld."
        )

    if result["human_review_required"]:

        st.warning("Human review required")

    else:

        st.info("Human review flag not triggered")


    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    risk_score = result.get("risk_score")
    anomaly_score = result.get("anomaly_score")

    col1.metric(
        "Fraud Risk Score",
        f"{risk_score:.4f}"
        if risk_score is not None
        else "N/A"
    )

    col2.metric(
        "Risk Level",
        result.get("risk_level") or "N/A"
    )

    col3.metric(
        "Anomaly Score",
        f"{anomaly_score:.4f}"
        if anomaly_score is not None
        else "N/A"
    )

    col4.metric(
        "Execution Time",
        f"{result['latency_seconds']:.2f}s"
    )


    # Risk gauge
    if risk_score is not None:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={"valueformat": ".4f"},
                title={
                    "text": "Uncalibrated Fraud Risk Score"
                },
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {
                            "range": [0, 0.4],
                            "color": "#BBF7D0"
                        },
                        {
                            "range": [0.4, 0.7],
                            "color": "#FDE68A"
                        },
                        {
                            "range": [0.7, 1],
                            "color": "#FECACA"
                        }
                    ]
                }
            )
        )

        fig.update_layout(height=300)

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.caption(
            "This score is not a calibrated probability "
            "of fraudulent activity."
        )


    # Investigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Investigation Report",
        "Agent Execution",
        "Knowledge Sources",
        "System Details"
    ])


    with tab1:

        st.subheader("AI Investigation Report")

        st.markdown(result["final_report"])

        st.download_button(
            "Download Investigation Report",
            data=result["final_report"],
            file_name=(
                f"{result['claim_id']}_report.txt"
            ),
            mime="text/plain"
        )


    with tab2:

        st.subheader("Multi-Agent Execution Trace")

        for index, entry in enumerate(
            result["execution_log"],
            start=1
        ):

            st.write(f"{index}. {entry}")

        st.subheader("Selected Agents")

        for agent in result["selected_agents"]:

            st.write(
                agent.replace("_", " ").title()
            )

        st.subheader("Completed Agents")

        for agent in result["completed_agents"]:

            st.write(
                "Completed: "
                + agent.replace("_", " ").title()
            )


    with tab3:

        st.subheader("Retrieved Knowledge Base Sources")

        if result["sources"]:

            for source in result["sources"]:

                with st.expander(source["title"]):

                    st.write(
                        "Citation:",
                        source["citation"]
                    )

                    st.write(
                        "Cosine similarity:",
                        source["similarity_score"]
                    )

        else:

            st.info(
                "No knowledge base sources available "
                "for this investigation."
            )


    with tab4:

        st.subheader("Structured System Response")

        st.json(result)


else:

    st.markdown("""
    ### How to get started

    1. Select an insurance claim from the sidebar.
    2. Choose an investigation type.
    3. Click **Run Investigation**.
    4. Review the AI-generated results and supporting evidence.
    """)


st.divider()

st.caption(
    "Agentic AI Insurance Risk & Fraud Detection Platform | "
    "Synthetic Data | Human-Centered Decision Support"
)
