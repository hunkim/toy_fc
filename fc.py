from typing import List, Dict, Optional, Any, Union
import json

# from langchain_groq import ChatGroq as Chat

from langchain_upstage import ChatUpstage as Chat
from langchain_community.tools import DuckDuckGoSearchResults
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
from typing import Optional, Dict, Union, List, Any


MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "llama-3.1-70b-versatile"
MODEL_NAME = "solar-pro"
ddg_search = DuckDuckGoSearchResults()


from typing import List, Dict, Any, Union, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def verify_facts(
    claimed_facts: List[Dict[str, Any]],
    context: str,
    kg: Dict[str, Any],
    confidence_threshold: float,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
) -> Dict[str, Dict[str, Any]]:
    """
    Verify the claimed facts against the knowledge graph and context.

    Args:
        claimed_facts (List[Dict[str, Any]]): The list of extracted claimed facts.
        context (str): The context information retrieved from the search.
        kg (Dict[str, Any]): The constructed knowledge graph.
        confidence_threshold (float): The confidence threshold for fact verification.
        llm (Optional[Chat]): The language model to use for verification, if needed.

    Returns:
        Dict[str, Dict[str, Any]]: Verified facts with status, confidence, and explanation.
        The structure is: {fact_id: {"claimed": str, "status": str, "confidence": float, "explanation": str}}
    """

    kg_str = json.dumps(kg, indent=2)
    verified_facts = {}

    valid_statuses = {"true", "false", "probably true", "probably false", "not sure"}

    for i, fact in enumerate(claimed_facts):
        verification_result = verify_one_fact(context, kg_str, fact, llm)

        status = verification_result.get("status", "not sure").lower()
        confidence = verification_result.get("confidence", 0.0)
        explanation = verification_result.get("explanation", "")

        # Validate status
        if status not in valid_statuses:
            status = "not sure"

        # Validate confidence score
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            confidence = 0.0

        # Apply confidence threshold
        if confidence < confidence_threshold:
            status = "not sure"

        verified_facts[str(i)] = {
            "claimed": f"{fact['entity']} {fact['relation']} {fact['value']}",
            "status": status,
            "confidence": confidence,
            "explanation": explanation,
        }

    return verified_facts

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def verify_one_fact(context, kg_str, fact, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert fact-checker. Your task is to verify a claimed fact against a knowledge graph and context information. Categorize the verification result into one of the following five categories: true, false, probably true, probably false, or not sure.",
            ),
            (
                "human",
                """Verify the following claimed fact using the provided knowledge graph and context. Categorize the verification result into one of the following:

1. **true**: Supporting evidence found in context.
2. **false**: Contradicting evidence found in context.
3. **probably true**: No context available, but based on your knowledge and common sense, the fact is likely true.
4. **probably false**: No contradicting evidence in context, but based on your knowledge and common sense, the fact is likely false.
5. **not sure**: Cannot judge due to subjectivity or unknowns.

Additionally, assign a confidence score between 0.0 and 1.0 that reflects the certainty of the categorization.

Provide the result in a JSON object with the following structure:
{{
  "status": "<CATEGORY>",
  "confidence": <CONFIDENCE_SCORE>,
  "explanation": "<BRIEF_EXPLANATION>"
}}

Ensure that:
1. The categorization is based on the information in the knowledge graph and context.
2. The confidence score accurately reflects the certainty of the categorization.
3. The explanation briefly justifies the verification decision and confidence score.
    
Claimed Fact: {entity} {relation} {value}

Knowledge Graph:
{kg}

Context:
{context}

Provide the verification result:""",
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    verification_result = chain.invoke(
        {
            "entity": fact["entity"],
            "relation": fact["relation"],
            "value": fact["value"],
            "kg": kg_str,
            "context": context,
        }
    )

    return verification_result


def fc(
    text: str,
    context: Optional[str] = None,
    kg: Optional[Dict] = None,
    verify_sources: bool = True,
    confidence_threshold: float = 0.7,
    llm=Chat(model=MODEL_NAME),
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
        print(f"    Status: {result['status']}")
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
    text: str, llm: Optional[Chat] = Chat(model=MODEL_NAME)
) -> List[Dict[str, Any]]:
    """
    Extract claimed facts from the given text, including entities and their relationships.

    Args:
        text (str): The input text to extract facts from.
        llm (Optional[Chat]): The language model to use for extraction, if needed.

    Returns:
        List[Dict[str, Any]]: A list of extracted facts, where each fact is represented as a dictionary.
    """

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



def search_context(
    text: str,
    claimed_facts: List[Dict[str, Any]],
    search_tool: Any,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
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




@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def build_kg(
    claimed_facts: List[Dict[str, Any]],
    context: str,
    llm: Optional[Chat] = Chat(model=MODEL_NAME),
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




def add_fact_check_to_text(text, verified_facts, llm=Chat(model=MODEL_NAME)):
    # First, let's create a mapping of claimed facts to their verifications
    fact_map = {fact["claimed"]: fact for fact in verified_facts.values()}

    # Now, let's ask the LLM to annotate the original text
    system_message = HumanMessage(
        content="""
    You are an AI assistant tasked with adding fact-check annotations to a given text.
    For each fact in the text that has been verified, add an inline annotation 
    right after the fact, using the following format:
    [Fact: <STATUS> (Confidence: <CONFIDENCE>) - <BRIEF_EXPLANATION>]
    Where <STATUS> is True, False, or Unsure, <CONFIDENCE> is the confidence score,
    and <BRIEF_EXPLANATION> is a very short explanation.
    """
    )

    human_message = HumanMessage(
        content=f"""
    Original text:
    {text}

    Verified facts:
    {fact_map}

    Please add fact-check annotations to the original text based on the verified facts.
    """
    )

    response = llm([system_message, human_message])

    return response.content


