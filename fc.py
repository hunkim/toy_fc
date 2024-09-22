from typing import List, Dict, Optional, Any, Union
import json
from langchain_groq import ChatGroq as Chat
from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "llama-3.1-70b-versatile"
llm = Chat(model=MODEL_NAME)
ddg_search = DuckDuckGoSearchResults()


from typing import List, Dict, Any, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def verify_facts(
    claimed_facts: List[Dict[str, Any]],
    context: str,
    kg: Dict[str, Any],
    confidence_threshold: float,
    llm: Optional[Chat] = None,
) -> Dict[str, Dict[str, Union[str, float, bool]]]:
    """
    Verify the claimed facts against the knowledge graph and context.

    Args:
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        context (str): The context information retrieved from the search.
        kg (Dict[str, Any]): The constructed knowledge graph.
        confidence_threshold (float): The confidence threshold for fact verification.
        llm (Optional[Chat]): The language model to use for verification, if needed.

    Returns:
        Dict[str, Dict[str, Union[str, float, bool]]]: Verified facts with confidence scores.
        The structure is: {fact_id: {"claimed": str, "verified": bool, "confidence": float, "explanation": str}}
    """
    if llm is None:
        llm = Chat(model=MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert fact-checker. Your task is to verify a claimed fact against a knowledge graph and context information. Provide a verification result, confidence score, and explanation for the fact.",
            ),
            (
                "human",
                """Verify the following claimed fact using the provided knowledge graph and context. Determine if it's verified, assign a confidence score (0.0 to 1.0), and provide a brief explanation.

Claimed Fact: {entity} {relation} {value}

Knowledge Graph:
{kg}

Context:
{context}

Provide a JSON object with the following structure:
{{
  "verified": bool,
  "confidence": float,
  "explanation": string
}}

Ensure that:
1. The verification is based on the information in the knowledge graph and context.
2. The confidence score reflects the certainty of the verification (1.0 for absolute certainty, lower for less certainty).
3. The explanation briefly justifies the verification decision and confidence score.

Provide the verification result:""",
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    kg_str = json.dumps(kg, indent=2)
    verified_facts = {}

    for i, fact in enumerate(claimed_facts):
        verification_result = chain.invoke(
            {
                "entity": fact["entity"],
                "relation": fact["relation"],
                "value": fact["value"],
                "kg": kg_str,
                "context": context,
            }
        )

        verified_facts[str(i)] = {
            "claimed": f"{fact['entity']} {fact['relation']} {fact['value']}",
            **verification_result,
        }

    return verified_facts


# Example usage in fc function:
# verified_facts = verify_facts(claimed_facts, context, kg, confidence_threshold, llm)

import json
from typing import Optional, Dict, Union, List, Any
from langchain_groq import ChatGroq as Chat


def fc(
    text: str,
    context: Optional[str] = None,
    kg: Optional[Dict] = None,
    verify_sources: bool = True,
    confidence_threshold: float = 0.7,
    llm=None,
) -> Dict[str, Dict[str, Union[str, float, bool]]]:
    """
    Function to perform fact checking on a given text using a knowledge graph.

    Args:
        text (str): The text to be checked.
        context (Optional[str]): Additional context to be used for fact checking.
        kg (Optional[Dict]): The knowledge graph to be used for fact checking.
        verify_sources (bool): Whether to verify the sources of the information.
        confidence_threshold (float): The confidence threshold for the fact checking.
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        Dict[str, Dict[str, Union[str, float, bool]]]: The fact checked information.
        The structure is: {fact_id: {"claimed": str, "verified": bool, "confidence": float, "explanation": str}}
    """

    print("\n--- Starting Fact Checking Process ---")
    print(f"Input text: {text}")

    if llm is None:
        llm = Chat(model=MODEL_NAME)
        print(f"Using default language model: {MODEL_NAME}")
    else:
        print("Using provided language model")

    print("\nStep 1: Extracting claimed facts")
    claimed_facts = extracted_claimed_facts(text, llm)
    print(f"Extracted {len(claimed_facts)} claimed facts:")
    for i, fact in enumerate(claimed_facts):
        print(f"  {i+1}. {fact['entity']} {fact['relation']} {fact['value']}")

    if context is None:
        print("\nStep 2: Searching for relevant context")
        context = search_context(text, claimed_facts, ddg_search, llm)
        print(f"Retrieved context (first 100 characters): {context[:100]}...")
    else:
        print("\nStep 2: Using provided context")

    if kg is None:
        print("\nStep 3: Building knowledge graph")
        kg = build_kg(claimed_facts, context, llm)
        print(f"Built knowledge graph with {len(kg)} entities")
    else:
        print("\nStep 3: Using provided knowledge graph")

    print("\nStep 4: Verifying facts")
    verified_facts = verify_facts(claimed_facts, context, kg, confidence_threshold, llm)
    print(f"Verified {len(verified_facts)} facts:")
    for fact_id, result in verified_facts.items():
        print(f"  Fact {fact_id}:")
        print(f"    Claimed: {result['claimed']}")
        print(f"    Verified: {result['verified']}")
        print(f"    Confidence: {result['confidence']}")
        print(
            f"    Explanation: {result['explanation'][:100]}..."
        )  # Truncate long explanations

    print("\n--- Fact Checking Process Completed ---")

    # Final step 
    print("\nStep 5: Adding fact-check annotations to the original text")
    fact_checked_text = add_fact_check_to_text(text, verified_facts, llm)
    print("Fact-checked text generated")

    return verified_facts, fact_checked_text


def extracted_claimed_facts(
    text: str, llm: Optional[Chat] = None
) -> List[Dict[str, Any]]:
    """
    Extract claimed facts from the given text, including entities and their relationships.

    Args:
        text (str): The input text to extract facts from.
        llm (Optional[Chat]): The language model to use for extraction, if needed.

    Returns:
        List[Dict[str, Any]]: A list of extracted facts, where each fact is represented as a dictionary.
    """
    if llm is None:
        llm = Chat(model=MODEL_NAME)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert fact extractor. Your task is to analyze the given text and extract a list of claimed facts, focusing on entities and their relationships. Extract precise and specific relations without categorizing them into predefined types.",
            ),
            (
                "human",
                """Extract the claimed facts from the following text, providing a list of dictionaries. Each dictionary should represent a fact and include keys for 'entity', 'relation', and 'value'. Be specific and precise with the relations.

Examples:
Input: "Albert Einstein developed the theory of relativity in 1915."
Output: [
    {{"entity": "Albert Einstein", "relation": "developed", "value": "theory of relativity"}},
    {{"entity": "theory of relativity", "relation": "developed in", "value": "1915"}}
]

Input: "The Eiffel Tower, completed in 1889, stands at a height of 324 meters."
Output: [
    {{"entity": "Eiffel Tower", "relation": "completed in", "value": "1889"}},
    {{"entity": "Eiffel Tower", "relation": "height", "value": "324 meters"}}
]

Now, extract facts from the following text:
{input_text}""",
            ),
            (
                "human",
                "Respond with a JSON array of fact dictionaries only, without any additional text.",
            ),
        ]
    )

    # Create the output parser
    output_parser = JsonOutputParser()

    # Create the chain
    chain = prompt | llm | output_parser

    # Run the chain
    result = chain.invoke({"input_text": text})

    return result


# Example usage:
# facts = extracted_claimed_facts("Albert Einstein developed the theory of relativity in 1915.")


# Add this test case at the end of the file
def test_extracted_claimed_facts():
    test_text = "Albert Einstein developed the theory of relativity in 1915. He was born in Ulm, Germany in 1879."
    llm = Chat(model=MODEL_NAME)

    print("Extracting facts from the text...")
    result = extracted_claimed_facts(test_text, llm)

    print("Extracted facts:")
    for fact in result:
        print(fact)

    # Basic property checks
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"

    for fact in result:
        assert isinstance(fact, dict), "Each fact should be a dictionary"
        assert "entity" in fact, "Each fact should have an 'entity' key"
        assert "relation" in fact, "Each fact should have a 'relation' key"
        assert "value" in fact, "Each fact should have a 'value' key"

    # Check for specific facts (more flexible now)
    assert any(
        "Einstein" in f["entity"] and "relativity" in f["value"] for f in result
    ), "Fact about Einstein and relativity not found"
    assert any(
        "Einstein" in f["entity"] and "born" in f["relation"] for f in result
    ), "Fact about Einstein's birth not found"
    assert any(
        "Einstein" in f["entity"] and "Ulm" in f["value"] for f in result
    ), "Fact about Einstein's birthplace not found"
    assert any(
        "Einstein" in f["entity"] and "1879" in f["value"] for f in result
    ), "Fact about Einstein's birth year not found"

    print("All checks passed!")


def search_context(
    text: str,
    claimed_facts: List[Dict[str, Any]],
    search_tool: Any,
    llm: Optional[Chat] = None,
) -> str:
    """
    Search for relevant information using claimed facts.

    Args:
        text (str): The original input text.
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        search_tool (Any): The search tool to use for finding information (e.g., DuckDuckGoSearchResults).
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        str: The relevant context information found from the search.
    """
    if llm is None:
        llm = Chat(model=MODEL_NAME)

    # Step 1: Generate search keywords
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at generating concise and relevant search keywords. Your task is to analyze the given text and extracted facts, then produce a list of 3-5 search keywords or short phrases that would be most effective for finding additional context and verification information.",
            ),
            (
                "human",
                """Given the following text and extracted facts, generate a list of 3-5 search keywords or short phrases:

Text: {text}

Extracted Facts:
{facts}

Provide only the keywords or short phrases, separated by commas.""",
            ),
        ]
    )

    facts_str = "\n".join(
        [
            f"- {fact['entity']} {fact['relation']} {fact['value']}"
            for fact in claimed_facts
        ]
    )
    keywords_response = llm.invoke(prompt.format(text=text, facts=facts_str))

    # Parse the keywords from the response
    keywords = [kw.strip() for kw in keywords_response.content.split(",") if kw.strip()]

    # Step 2: Perform search using the generated keywords
    search_query = " ".join(keywords)
    search_results = search_tool.run(search_query)

    # Step 3: Return the search results
    return search_results


# Example usage in fc function:
# if context is None:
#     context = search_context(text, claimed_facts, ddg_search, llm)


# Add this test case at the end of the file
def test_search_context():
    test_text = "Albert Einstein developed the theory of relativity in 1915."
    claimed_facts = [
        {
            "entity": "Albert Einstein",
            "relation": "developed",
            "value": "theory of relativity",
        },
        {"entity": "theory of relativity", "relation": "developed in", "value": "1915"},
    ]
    llm = Chat(model=MODEL_NAME)
    ddg_search = DuckDuckGoSearchResults()

    print("Searching for context...")
    context = search_context(test_text, claimed_facts, ddg_search, llm)

    print("Generated context:")
    print(context)

    # Basic property checks
    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"

    # Check for relevant content
    relevant_terms = ["Einstein", "relativity", "1915", "physics", "theory"]
    assert any(
        term.lower() in context.lower() for term in relevant_terms
    ), "Context should contain relevant information"

    print("All checks passed!")


def build_kg(
    claimed_facts: List[Dict[str, Any]], context: str, llm: Optional[Chat] = None
) -> Dict[str, Any]:
    """
    Build a knowledge graph from claimed facts and context information.

    Args:
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        context (str): The context information retrieved from the search.
        llm (Optional[Chat]): The language model to use for processing, if needed.

    Returns:
        Dict[str, Any]: The constructed knowledge graph with source information.
    """
    if llm is None:
        llm = Chat(model=MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in building knowledge graphs. Your task is to analyze the given context and construct a knowledge graph, using the claimed facts only as inspiration for the schema without assuming their truth. Include source information for each fact.",
            ),
            (
                "human",
                """Given the following context and claimed facts, construct a knowledge graph. Assume all information in the context is true, but use the claimed facts only as hints for the types of relations to look for.

Context:
{context}

Claimed Facts (use only as schema hints):
{claimed_facts}

Construct the knowledge graph as a JSON object where keys are entities and values are dictionaries of relations. Each relation should have a "value" and a "source" (a relevant quote from the context).

Example format:
{{
  "Entity1": {{
    "relation1": {{
      "value": "Value1",
      "source": "Relevant quote from context"
    }},
    "relation2": {{
      "value": "Value2",
      "source": "Another relevant quote"
    }}
  }},
  "Entity2": {{
    ...
  }}
}}

Ensure that:
1. All information comes from the context, not the claimed facts.
2. Each fact has a source quote from the context.
3. The schema is inspired by, but not limited to, the relations in the claimed facts.

Construct the knowledge graph:""",
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    facts_str = "\n".join(
        [
            f"- {fact['entity']} {fact['relation']} {fact['value']}"
            for fact in claimed_facts
        ]
    )

    kg = chain.invoke({"context": context, "claimed_facts": facts_str})

    return kg


# Example usage in fc function:
# if kg is None:
#     kg = build_kg(claimed_facts, context, llm)


# Add this test case at the end of the file
def test_build_kg():
    claimed_facts = [
        {
            "entity": "Albert Einstein",
            "relation": "developed",
            "value": "theory of relativity",
        },
        {"entity": "theory of relativity", "relation": "developed in", "value": "1915"},
    ]
    context = "Albert Einstein, born in 1879, was a renowned physicist. He published his theory of general relativity in 1915, which revolutionized our understanding of gravity. Einstein's work on the photoelectric effect earned him the Nobel Prize in Physics in 1921."
    llm = Chat(model=MODEL_NAME)

    kg = build_kg(claimed_facts, context, llm)

    print("Generated Knowledge Graph:")
    print(json.dumps(kg, indent=2))

    assert isinstance(kg, dict), "KG should be a dictionary"
    assert len(kg) > 0, "KG should not be empty"
    assert "Albert Einstein" in kg, "KG should contain information about Einstein"
    assert any(
        "relativity" in str(v) for v in kg.values()
    ), "KG should contain information about relativity"
    assert all(
        isinstance(v, dict) and "value" in v and "source" in v
        for relations in kg.values()
        for v in relations.values()
    ), "All facts should have 'value' and 'source'"
    assert all(
        v["source"] in context for relations in kg.values() for v in relations.values()
    ), "All sources should be from the context"

    print("All checks passed!")


def test_verify_facts():
    claimed_facts = [
        {
            "entity": "Albert Einstein",
            "relation": "developed",
            "value": "theory of relativity",
        },
        {"entity": "theory of relativity", "relation": "developed in", "value": "1915"},
        {"entity": "Albert Einstein", "relation": "born in", "value": "1879"},
    ]
    context = "Albert Einstein, born in 1879, was a renowned physicist. He published his theory of general relativity in 1915, which revolutionized our understanding of gravity. Einstein's work on the photoelectric effect earned him the Nobel Prize in Physics in 1921."
    kg = {
        "Albert Einstein": {
            "born in": {
                "value": "1879",
                "source": "Albert Einstein, born in 1879, was a renowned physicist.",
            },
            "developed": {
                "value": "theory of general relativity",
                "source": "He published his theory of general relativity in 1915, which revolutionized our understanding of gravity.",
            },
            "received": {
                "value": "Nobel Prize in Physics",
                "source": "Einstein's work on the photoelectric effect earned him the Nobel Prize in Physics in 1921.",
            },
        },
        "theory of general relativity": {
            "published in": {
                "value": "1915",
                "source": "He published his theory of general relativity in 1915, which revolutionized our understanding of gravity.",
            }
        },
    }
    llm = Chat(model=MODEL_NAME)
    confidence_threshold = 0.7

    verified_facts = verify_facts(claimed_facts, context, kg, confidence_threshold, llm)

    print("Verified Facts:")
    print(json.dumps(verified_facts, indent=2))

    assert isinstance(verified_facts, dict), "Verified facts should be a dictionary"
    assert len(verified_facts) <= len(
        claimed_facts
    ), "Should have results for at most all claimed facts"

    for fact_id, result in verified_facts.items():
        assert "verified" in result, "Each result should have a 'verified' field"
        assert "confidence" in result, "Each result should have a 'confidence' field"
        assert "explanation" in result, "Each result should have an 'explanation' field"
        assert isinstance(result["verified"], bool), "'verified' should be a boolean"
        assert isinstance(result["confidence"], float), "'confidence' should be a float"
        assert isinstance(
            result["explanation"], str
        ), "'explanation' should be a string"
        assert (
            result["confidence"] >= confidence_threshold
        ), f"Confidence for fact {fact_id} should be above the threshold"

    print("All checks passed!")


def add_fact_check_to_text(text, verified_facts, llm=None):
    if llm is None:
        llm = Chat(model=MODEL_NAME)

    # First, let's create a mapping of claimed facts to their verifications
    fact_map = {fact['claimed']: fact for fact in verified_facts.values()}
    
    # Now, let's ask the LLM to annotate the original text
    system_message = HumanMessage(content="""
    You are an AI assistant tasked with adding fact-check annotations to a given text.
    For each fact in the text that has been verified, add an inline annotation 
    right after the fact, using the following format:
    [Fact: <STATUS> (Confidence: <CONFIDENCE>) - <BRIEF_EXPLANATION>]
    Where <STATUS> is True, False, or Unsure, <CONFIDENCE> is the confidence score,
    and <BRIEF_EXPLANATION> is a very short explanation.
    """)

    human_message = HumanMessage(content=f"""
    Original text:
    {text}

    Verified facts:
    {fact_map}

    Please add fact-check annotations to the original text based on the verified facts.
    """)

    response = llm([system_message, human_message])
    
    return response.content


def test_add_fact_check_to_text():
    # Sample text
    text = "The Earth is flat and orbits around the Moon. The Sun is cold."

    # Sample verified facts
    verified_facts = {
        "0": {
            "claimed": "Earth is flat",
            "verified": False,
            "confidence": 0.99,
            "explanation": "The Earth is actually an oblate spheroid."
        },
        "1": {
            "claimed": "Earth orbits around the Moon",
            "verified": False,
            "confidence": 0.99,
            "explanation": "The Moon orbits around the Earth, not vice versa."
        },
        "2": {
            "claimed": "Sun is cold",
            "verified": False,
            "confidence": 0.99,
            "explanation": "The Sun is extremely hot, with surface temperatures around 5,500°C."
        }
    }

    # Create a mock LLM function
    class MockLLM:
        def __call__(self, messages):
            return AIMessage(content="The Earth is flat [Fact: False (Confidence: 0.99) - The Earth is actually an oblate spheroid.] and orbits around the Moon [Fact: False (Confidence: 0.99) - The Moon orbits around the Earth, not vice versa.]. The Sun is cold [Fact: False (Confidence: 0.99) - The Sun is extremely hot, with surface temperatures around 5,500°C.].")

    mock_llm = MockLLM()

    # Call the function
    result = add_fact_check_to_text(text, verified_facts, mock_llm)

    # Print the result
    print("Annotated text:")
    print(result)

    # Assertions
    assert "[Fact: False" in result, "Annotation for false fact not found"
    assert "Confidence: 0.99" in result, "Confidence score not found in annotation"
    assert "The Earth is actually an oblate spheroid" in result, "Explanation for Earth's shape not found"
    assert "The Moon orbits around the Earth" in result, "Explanation for Earth-Moon relationship not found"
    assert "The Sun is extremely hot" in result, "Explanation for Sun's temperature not found"

    print("All assertions passed!")

# Run the test if this script is executed directly
if __name__ == "__main__":
    test_add_fact_check_to_text()
