import streamlit as st
import pandas as pd
import networkx as nx
import json
from pyvis.network import Network
from typing import Optional, Dict, Union
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

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")


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
    net.save_graph("kg_graph.html")

    with open("kg_graph.html", "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=500)


def fc_streamlitet(
    text: str, verify_sources: bool = True, confidence_threshold: float = 0.7, llm=None
) -> Dict[str, Dict[str, Union[str, float, bool]]]:
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
        status = "✅ True" if result["verified"] else "❌ False"
        confidence = result["confidence"]
        color = "#90EE90" if result["verified"] else "#FFB3BA"

        with st.expander(f"Fact {fact_id}: {result['claimed'][:50]}...", expanded=True):
            st.markdown(
                f"<p style='background-color: {color}; padding: 10px;'>"
                f"<strong>Status:</strong> {status} (Confidence: {confidence:.2f})<br>"
                f"<strong>Claimed:</strong> {result['claimed']}<br>"
                f"<strong>Explanation:</strong> {result['explanation']}"
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
            df = pd.DataFrame(results).T
            df["confidence"] = df["confidence"].apply(lambda x: f"{x:.2f}")
            df["verified"] = df["verified"].apply(
                lambda x: "✅ True" if x else "❌ False"
            )

            # Display the DataFrame
            st.dataframe(
                df.style.apply(
                    lambda x: [
                        (
                            "background-color: #90EE90"
                            if v == "✅ True"
                            else "background-color: #FFB3BA"
                        )
                        for v in x
                    ],
                    subset=["verified"],
                )
            )

            # Add fact-check annotations to the original text
            annotated_text = add_fact_check_to_text(text, results)

            # Highlight the fact-check parts
            def highlight_fact(match):
                fact = match.group(0)
                if "True" in fact:
                    color = "#90EE90"  # Light green for true facts
                elif "False" in fact:
                    color = "#FFB3BA"  # Light red for false facts
                else:
                    color = "#FFFFE0"  # Light yellow for uncertain facts
                return f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px; font-size: 0.8em;">{fact}</span>'

            highlighted_text = re.sub(r"\[Fact:.*?\]", highlight_fact, annotated_text)

            st.write("Annotated Text:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

    else:
        st.warning("Please enter some text to fact-check.")
