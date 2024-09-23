import streamlit as st
import pandas as pd
import networkx as nx
import json
from pyvis.network import Network
from typing import Optional, Dict, Union, List, Any
from fc import (
    extracted_claimed_facts,
    search_context,
    build_kg,
    verify_facts,
    add_fact_check_to_text,
    Chat,
    MODEL_NAME,
    ddg_search,
)
import re
import io
import tempfile

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")


# Monkey-patch the pyvis library to allow StringIO objects
original_write_html = Network.write_html

def patched_write_html(self, name, notebook=False):
    if isinstance(name, io.StringIO):
        html = self.generate_html()
        name.write(html)
    else:
        original_write_html(self, name, notebook)

Network.write_html = patched_write_html

def visualize_kg(kg):
    G = nx.Graph()

    def add_node_safe(node):
        if not isinstance(node, str):
            return json.dumps(node)
        return node

    for entity, relations in kg.items():
        G.add_node(add_node_safe(entity))
        for relation, value in relations.items():
            safe_value = add_node_safe(value)
            G.add_node(safe_value)
            G.add_edge(add_node_safe(entity), safe_value, title=relation)

    net = Network(
        notebook=True,
        width="100%",
        height="500px",
        bgcolor="#222222",
        font_color="white",
    )
    net.from_nx(G)
    net.repulsion(node_distance=200, spring_length=200)
    
    # Use StringIO to capture the HTML string
    html_io = io.StringIO()
    net.write_html(html_io)
    html_string = html_io.getvalue()

    st.components.v1.html(html_string, height=500)


def add_fact_check_to_text(text: str, results: Dict[str, Dict[str, Union[str, float]]]) -> str:
    """
    Annotates the original text with fact-checking results.

    Args:
        text (str): The original input text.
        results (Dict[str, Dict[str, Union[str, float]]]): The verification results.

    Returns:
        str: Annotated text with fact checks.
    """
    for fact_id, result in results.items():
        claimed = result["claimed"]
        status = result["status"]
        confidence = result["confidence"]
        # Safeguard against '|' characters in claimed facts
        claimed_safe = claimed.replace("|", "\\|")
        # Escape special regex characters in claimed fact
        escaped_claimed = re.escape(claimed_safe)
        # Construct the annotation
        annotation = f"[Fact: {claimed} | status: {status} | confidence: {confidence:.2f}]"
        # Replace only the first occurrence to prevent multiple replacements
        text = re.sub(escaped_claimed, annotation, text, count=1)
    return text


def fc_streamlitet(
    text: str,
    verify_sources: bool = True,
    confidence_threshold: float = 0.7,
    llm: Optional[Chat] = None,
) -> Dict[str, Dict[str, Union[str, float]]]:
    st.write("--- Starting Fact Checking Process ---")
    st.write(f"Input text: {text}")

    if llm is None:
        llm = Chat(model=MODEL_NAME)
        st.info(f"Using default language model: {MODEL_NAME}")
    else:
        st.info("Using provided language model")

    with st.spinner("Step 1: Extracting claimed facts"):
        claimed_facts = extracted_claimed_facts(text, llm)
        st.write(f"Extracted {len(claimed_facts)} claimed facts:")
        for i, fact in enumerate(claimed_facts):
            st.write(f"  {i+1}. {fact['entity']} {fact['relation']} {fact['value']}")

    with st.spinner("Step 2: Searching for relevant context"):
        context = search_context(text, claimed_facts, ddg_search, llm)
        st.write(f"Retrieved context (first 100 characters): {context[:100]}...")

    with st.spinner("Step 3: Building knowledge graph"):

        kg = build_kg(claimed_facts, context, llm)

        st.write(f"Built knowledge graph with {len(kg)} entities")
        st.subheader("Knowledge Graph Visualization")
        visualize_kg(kg)

    with st.spinner("Step 4: Verifying facts"):
        verified_facts = verify_facts(
            claimed_facts, context, kg, confidence_threshold, llm
        )
        st.write(f"Verified {len(verified_facts)} facts")

    st.success("--- Fact Checking Process Completed ---")

    # Display all verified facts
    st.subheader("Fact Checking Results")
    for fact_id, result in verified_facts.items():
        status = result["status"]
        confidence = result["confidence"]
        explanation = result["explanation"]

        # Define color and icon based on status
        if status == "true":
            color = "#90EE90"  # Light green
            status_display = "✅ True"
        elif status == "false":
            color = "#FFB3BA"  # Light red
            status_display = "❌ False"
        elif status == "probably true":
            color = "#ADD8E6"  # Light blue
            status_display = "ℹ️ Probably True"
        elif status == "probably false":
            color = "#FFD700"  # Gold
            status_display = "⚠️ Probably False"
        else:
            color = "#D3D3D3"  # Light gray
            status_display = "❓ Not Sure"

        with st.expander(f"Fact {fact_id}: {result['claimed'][:50]}...", expanded=True):
            st.markdown(
                f"<p style='background-color: {color}; padding: 10px;'>"
                f"<strong>Status:</strong> {status_display} (Confidence: {confidence:.2f})<br>"
                f"<strong>Claimed:</strong> {result['claimed']}<br>"
                f"<strong>Explanation:</strong> {explanation}"
                "</p>",
                unsafe_allow_html=True,
            )

    return verified_facts


st.title("🔍 Fact Checker")

default_text = """Sung Kim is CEO of Upstage.AI and Lucy Park is CPO of the company."""

text = st.text_area("Enter the text to fact-check:", value=default_text, height=150)
verify_sources = st.checkbox("Verify sources", value=True)
confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.1)

if st.button("Check Facts"):
    if text:
        process_tab, results_tab = st.tabs(["Fact-Checking Process", "Results"])

        with process_tab:
            results = fc_streamlitet(
                text=text,
                verify_sources=verify_sources,
                confidence_threshold=confidence_threshold,
            )

        with results_tab:
            st.subheader("Fact-Checking Results Summary")

            # Create a DataFrame for the results
            df = pd.DataFrame.from_dict(results, orient="index")
            df.reset_index(inplace=True)
            df = df.rename(columns={
                "index": "Fact ID",
                "claimed": "Claimed",
                "status": "Status",
                "confidence": "Confidence",
                "explanation": "Explanation"
            })

            # Define color mapping for statuses
            color_mapping = {
                "true": "#90EE90",
                "false": "#FFB3BA",
                "probably true": "#ADD8E6",
                "probably false": "#FFD700",
                "not sure": "#D3D3D3"
            }

            # Apply color based on status
            def color_status(status):
                return f'background-color: {color_mapping.get(status.lower(), "#D3D3D3")}'

            # Corrected styling: applymap receives the cell value, not the Series
            styled_df = df.style.applymap(
                color_status,
                subset=["Status"]
            ).format({
                "Confidence": "{:.2f}"
            })

            # Display the DataFrame
            st.dataframe(styled_df)

            # Add fact-check annotations to the original text
            annotated_text = add_fact_check_to_text(text, results)

            # Debugging Step: Display the raw annotated text to verify format
            st.write("Annotated Text (Raw):")
            st.text(annotated_text)

            # Highlight the fact-check parts
            def highlight_fact(match):
                fact = match.group(0)
                # Extract status using regex: look for 'status: <STATUS>'
                status_search = re.search(r'status:\s*([a-zA-Z\s]+)', fact, re.IGNORECASE)
                if status_search:
                    status = status_search.group(1).strip().lower()
                else:
                    status = "not sure"

                color = color_mapping.get(status, "#D3D3D3")
                return f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">{fact}</span>'

            # Adjust the regex pattern based on how `add_fact_check_to_text` formats the annotations
            highlighted_text = re.sub(r"\[Fact:.*?\]", highlight_fact, annotated_text)

            st.write("Annotated Text:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

    else:
        st.warning("Please enter some text to fact-check.")
