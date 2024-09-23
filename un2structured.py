from langchain_upstage import ChatUpstage as Chat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict, Any  # Add this import
import json  # Add this import

MODEL_NAME = "solar-pro"


def text2kvpairs(
    text: str, llm: Chat = Chat(model_name=MODEL_NAME)
) -> List[Dict[str, str]]:
    """
    Extract key-value pairs from the given text using a language model with high accuracy.

    Args:
        text (str): The input text from which to extract key-value pairs.
        llm (Chat, optional): The language model to use for extraction. Defaults to Chat(model_name=MODEL_NAME).

    Returns:
        List[Dict[str, str]]: A list of dictionaries representing the extracted key-value pairs.
    """

    # Define the prompt template with comprehensive instructions and multiple examples
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are an advanced AI assistant specialized in extracting comprehensive key-value pairs from unstructured text with high accuracy and completeness. Your task is to:"
                "\n1. Identify all significant pieces of information in the text."
                "\n2. Create meaningful and specific keys for each piece of information."
                "\n3. Provide precise and concise values for each key."
                "\n4. Ensure no important information is missed."
                "\n5. Break down complex information into multiple key-value pairs when necessary."
                "\n6. Maintain consistency in key naming and formatting."
                "\n7. Include quantitative data and specific details whenever present in the text."
                "\n8. Capture relationships between entities when relevant."
            ),
            (
                "human",
                """Extract key-value pairs from the following text. Each key-value pair should be represented as a dictionary with 'key' and 'value' fields. Aim for completeness and accuracy.

Examples:

Input: "Apple Inc., headquartered in Cupertino, was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is known for its innovative products like the iPhone and MacBook."
Output: [
    {{"key": "Company Name", "value": "Apple Inc."}},
    {{"key": "Headquarters", "value": "Cupertino"}},
    {{"key": "Founders", "value": "Steve Jobs, Steve Wozniak, Ronald Wayne"}},
    {{"key": "Year Founded", "value": "1976"}},
    {{"key": "Known For", "value": "Innovative products"}},
    {{"key": "Notable Products", "value": "iPhone, MacBook"}}
]

Input: "The 2024 Paris Olympics will feature athletes from over 200 countries competing in 32 different sports. The event is scheduled from July 26 to August 11, with an estimated budget of €7.3 billion."
Output: [
    {{"key": "Event", "value": "2024 Paris Olympics"}},
    {{"key": "Number of Participating Countries", "value": "Over 200"}},
    {{"key": "Number of Sports", "value": "32"}},
    {{"key": "Start Date", "value": "July 26, 2024"}},
    {{"key": "End Date", "value": "August 11, 2024"}},
    {{"key": "Duration", "value": "17 days"}},
    {{"key": "Estimated Budget", "value": "€7.3 billion"}}
]

Now, extract key-value pairs from the following text. Be thorough and capture all relevant information:
{text}""",
            ),
            (
                "human",
                "Respond with a JSON array of key-value pair dictionaries only, without any additional text or explanations. Ensure that all relevant information from the text is captured in the key-value pairs.",
            ),
        ]
    )

    # Initialize the output parser
    output_parser = JsonOutputParser()

    # Create the processing chain
    chain = prompt | llm | output_parser

    # Execute the chain with the provided text
    result = chain.invoke({"text": text})

    return result


def text2kg(
    text: str, kv_pairs: List[Dict[str, str]], llm: Chat = Chat(model_name=MODEL_NAME)
) -> Dict[str, Any]:
    """
    Extract a knowledge graph from the given text and key-value pairs using a language model with high accuracy.

    Args:
        text (str): The input text from which to extract the knowledge graph.
        kv_pairs (List[Dict[str, str]]): The key-value pairs extracted from the text.
        llm (Chat, optional): The language model to use for extraction. Defaults to Chat(model_name=MODEL_NAME).

    Returns:
        Dict[str, Any]: A dictionary representing the extracted knowledge graph.
    """

    # Define the prompt template with comprehensive instructions and multiple examples
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are an advanced AI assistant specialized in extracting comprehensive knowledge graphs from unstructured text and key-value pairs with high accuracy. Your task is to:"
                "\n1. Identify all significant entities, relationships, and attributes in the text and key-value pairs."
                "\n2. Create a hierarchical structure that accurately represents the relationships between entities."
                "\n3. Ensure that all important information is captured, including nested relationships."
                "\n4. Use consistent naming conventions for entities and relationships."
                "\n5. Include all relevant attributes for each entity."
                "\n6. Capture quantitative data and specific details when present."
                "\n7. Represent complex relationships using nested structures when appropriate."
                "\n8. Ensure that the knowledge graph is logically organized and easy to navigate."
            ),
            (
                "human",
                """Extract a knowledge graph from the following text and key-value pairs. The knowledge graph should be represented as a JSON object where keys are entities and values are dictionaries of relationships and attributes. Aim for completeness, accuracy, and proper hierarchical structure.

Examples:

Input Text: "Apple Inc., headquartered in Cupertino, was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is known for its innovative products like the iPhone and MacBook."
Key-Value Pairs: [
    {{"key": "Company Name", "value": "Apple Inc."}},
    {{"key": "Headquarters", "value": "Cupertino"}},
    {{"key": "Founders", "value": "Steve Jobs, Steve Wozniak, Ronald Wayne"}},
    {{"key": "Year Founded", "value": "1976"}},
    {{"key": "Known For", "value": "Innovative products"}},
    {{"key": "Notable Products", "value": "iPhone, MacBook"}}
]
Output Knowledge Graph: {{
    "Apple Inc.": {{
        "type": "Company",
        "attributes": {{
            "headquarters": "Cupertino",
            "yearFounded": 1976,
            "knownFor": "Innovative products"
        }},
        "relationships": {{
            "founders": [
                {{"name": "Steve Jobs", "role": "Co-founder"}},
                {{"name": "Steve Wozniak", "role": "Co-founder"}},
                {{"name": "Ronald Wayne", "role": "Co-founder"}}
            ],
            "products": [
                {{"name": "iPhone", "type": "Product"}},
                {{"name": "MacBook", "type": "Product"}}
            ]
        }}
    }}
}}

Input Text: "The 2024 Paris Olympics will feature athletes from over 200 countries competing in 32 different sports. The event is scheduled from July 26 to August 11, with an estimated budget of €7.3 billion."
Key-Value Pairs: [
    {{"key": "Event", "value": "2024 Paris Olympics"}},
    {{"key": "Number of Participating Countries", "value": "Over 200"}},
    {{"key": "Number of Sports", "value": "32"}},
    {{"key": "Start Date", "value": "July 26, 2024"}},
    {{"key": "End Date", "value": "August 11, 2024"}},
    {{"key": "Duration", "value": "17 days"}},
    {{"key": "Estimated Budget", "value": "€7.3 billion"}}
]
Output Knowledge Graph: {{
    "2024 Paris Olympics": {{
        "type": "SportingEvent",
        "attributes": {{
            "location": "Paris",
            "year": 2024,
            "participatingCountries": "Over 200",
            "numberOfSports": 32,
            "startDate": "2024-07-26",
            "endDate": "2024-08-11",
            "duration": "17 days",
            "estimatedBudget": "€7.3 billion"
        }},
        "relationships": {{
            "host": {{"name": "Paris", "type": "City", "country": "France"}},
            "organizer": {{"name": "International Olympic Committee", "type": "Organization"}}
        }}
    }}
}}

Now, extract the knowledge graph from the following text and key-value pairs:
Text: {text}
Key-Value Pairs: {kv_pairs}""",
            ),
            (
                "human",
                "Respond with a JSON object representing the knowledge graph only, without any additional text or explanations. Ensure that the knowledge graph captures all relevant entities, relationships, and attributes from both the text and the key-value pairs. Use nested structures to represent complex relationships and maintain a logical hierarchy.",
            ),
        ]
    )

    # Initialize the output parser
    output_parser = JsonOutputParser()

    # Create the processing chain
    chain = prompt | llm | output_parser

    # Execute the chain with the provided text and key-value pairs
    result = chain.invoke({"text": text, "kv_pairs": kv_pairs})

    return result


def text2questions(
    text: str, llm: Chat = Chat(model_name=MODEL_NAME)
) -> List[Dict[str, Any]]:
    """
    Break down complex questions or statements into smaller, focused questions.

    Args:
        text (str): The input text containing complex questions or statements.
        llm (Chat, optional): The language model to use for extraction. Defaults to Chat(model_name=MODEL_NAME).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing a sub-question and related information.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are an advanced AI assistant specialized in breaking down complex questions or statements into smaller, more focused questions. Your task is to:"
                "\n1. Analyze the input text and identify the main topics and subtopics."
                "\n2. Break down the complex question or statement into multiple smaller, specific questions."
                "\n3. Ensure that each sub-question is clear, concise, and focused on a single aspect."
                "\n4. Identify key search terms or keywords for each sub-question."
                "\n5. Provide a brief explanation of why each sub-question is relevant to the original input."
                "\n6. Maintain the original context and intent of the input text."
                "\n7. Ensure that the set of sub-questions covers all aspects of the original input."
                "\n8. Prioritize the sub-questions based on their importance to the overall query."
            ),
            (
                "human",
                """Break down the following complex question or statement into smaller, focused questions. For each sub-question, provide key search terms and a brief explanation of its relevance. Format your response as a JSON array of objects.

Examples:

Input: "What are the environmental and economic impacts of renewable energy adoption in developing countries, and how does it compare to traditional fossil fuels?"
Output: [
    {{
        "sub_question": "What are the environmental impacts of renewable energy adoption in developing countries?",
        "search_terms": ["renewable energy", "environmental impacts", "developing countries"],
        "explanation": "This question focuses on the specific environmental effects of adopting renewable energy in the context of developing nations.",
        "priority": 1
    }},
    {{
        "sub_question": "What are the economic impacts of renewable energy adoption in developing countries?",
        "search_terms": ["renewable energy", "economic impacts", "developing countries"],
        "explanation": "This question addresses the financial and economic consequences of implementing renewable energy solutions in developing economies.",
        "priority": 2
    }},
    {{
        "sub_question": "How does the environmental impact of renewable energy compare to that of fossil fuels in developing countries?",
        "search_terms": ["renewable energy vs fossil fuels", "environmental comparison", "developing countries"],
        "explanation": "This question compares the environmental effects of renewable energy and traditional fossil fuels specifically in developing nations.",
        "priority": 3
    }},
    {{
        "sub_question": "How does the economic impact of renewable energy compare to that of fossil fuels in developing countries?",
        "search_terms": ["renewable energy vs fossil fuels", "economic comparison", "developing countries"],
        "explanation": "This question compares the economic implications of adopting renewable energy versus continuing to use fossil fuels in developing economies.",
        "priority": 4
    }}
]

Input: "What are the latest advancements in artificial intelligence for healthcare, particularly in diagnosis and treatment planning, and what ethical considerations should be addressed?"
Output: [
    {{
        "sub_question": "What are the latest advancements in AI for medical diagnosis?",
        "search_terms": ["AI in healthcare", "medical diagnosis", "recent advancements"],
        "explanation": "This question focuses on cutting-edge AI technologies specifically used for diagnosing medical conditions.",
        "priority": 1
    }},
    {{
        "sub_question": "How is AI being used in treatment planning in healthcare?",
        "search_terms": ["AI in healthcare", "treatment planning", "personalized medicine"],
        "explanation": "This question explores the application of AI in developing and optimizing treatment plans for patients.",
        "priority": 2
    }},
    {{
        "sub_question": "What ethical considerations arise from using AI in healthcare diagnosis and treatment?",
        "search_terms": ["AI ethics in healthcare", "patient privacy", "algorithmic bias"],
        "explanation": "This question addresses the important ethical implications of implementing AI in sensitive healthcare applications.",
        "priority": 3
    }},
    {{
        "sub_question": "How does AI improve the accuracy of medical diagnoses compared to traditional methods?",
        "search_terms": ["AI vs traditional diagnosis", "diagnostic accuracy", "healthcare AI performance"],
        "explanation": "This question compares the effectiveness of AI-based diagnosis to conventional diagnostic approaches.",
        "priority": 4
    }}
]

Now, break down the following complex question or statement into smaller, focused questions:
{text}"""
            ),
            (
                "human",
                "Respond with a JSON array of objects representing the sub-questions, without any additional text or explanations. Ensure that each object contains 'sub_question', 'search_terms', 'explanation', and 'priority' fields. The set of sub-questions should comprehensively cover all aspects of the original input while maintaining its context and intent."
            ),
        ]
    )

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser
    result = chain.invoke({"text": text})

    return result


if __name__ == "__main__":

    complex_question = "What's the trend in AI and what are the best AI companies following the trend?"
    sub_questions = text2questions(complex_question)
    print("Sub-questions:")
    print(json.dumps(sub_questions, indent=4))  # Pretty-print the sub-questions

    text = """
    Solar-Proofread achieved 79% accuracy for a major international media company with a daily circulation of over 1.8 million, surpassing both base and fine-tuned GPT-4o mini.
Proofreading thousands of words per day is a highly detailed task, 
requiring a deep understanding of grammar, syntax, and vocabulary in multiple contexts. 
For a leading international media company, reviewing more than 1,800 articles 
per month was taking a toll on their editorial staff, 
cutting into time that could be spent on research and content creation. 
This case study explores how Upstage, a pioneering AI company specializing in building powerful LLMs for enterprises, 
used Predibase to fine-tune our own proprietary model "Solar-Mini" to efficiently and accurately proofread at scale. 
Solar-Mini is available for fine-tuning and serving exclusively on Predibase, and you can try it today for free."""

    kv_pairs = text2kvpairs(text)
    print("Key-Value Pairs:")
    print(json.dumps(kv_pairs, indent=4))  # Pretty-print the key-value pairs

    kg = text2kg(text, kv_pairs)
    print("Knowledge Graph:")
    print(json.dumps(kg, indent=4))  # Pretty-print the knowledge graph

   
