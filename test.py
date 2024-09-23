
import unittest
from fc import (
    extracted_claimed_facts,
    search_context,
    build_kg,
    verify_facts,
    add_fact_check_to_text,
    Chat,
    DuckDuckGoSearchResults,
    MODEL_NAME,
)
import json
from langchain.schema import AIMessage

class TestFactChecking(unittest.TestCase):

    def setUp(self):
        self.llm = Chat(model=MODEL_NAME)
        self.ddg_search = DuckDuckGoSearchResults()

    def test_extracted_claimed_facts(self):
        test_text = "Albert Einstein developed the theory of relativity in 1915. He was born in Ulm, Germany in 1879."
        
        print("Extracting facts from the text...")
        result = extracted_claimed_facts(test_text, self.llm)

        print("Extracted facts:")
        for fact in result:
            print(fact)

        self.assertIsInstance(result, list, "Result should be a list")
        self.assertGreater(len(result), 0, "Result should not be empty")

        for fact in result:
            self.assertIsInstance(fact, dict, "Each fact should be a dictionary")
            self.assertIn("entity", fact, "Each fact should have an 'entity' key")
            self.assertIn("relation", fact, "Each fact should have a 'relation' key")
            self.assertIn("value", fact, "Each fact should have a 'value' key")

        self.assertTrue(any("Einstein" in f["entity"] and "relativity" in f["value"] for f in result),
                        "Fact about Einstein and relativity not found")
        self.assertTrue(any("Einstein" in f["entity"] and "born" in f["relation"] for f in result),
                        "Fact about Einstein's birth not found")
        self.assertTrue(any("Einstein" in f["entity"] and "Ulm" in f["value"] for f in result),
                        "Fact about Einstein's birthplace not found")
        self.assertTrue(any("Einstein" in f["entity"] and "1879" in f["value"] for f in result),
                        "Fact about Einstein's birth year not found")

        print("All checks passed!")

    def test_search_context(self):
        test_text = "Albert Einstein developed the theory of relativity in 1915."
        claimed_facts = [
            {
                "entity": "Albert Einstein",
                "relation": "developed",
                "value": "theory of relativity",
            },
            {"entity": "theory of relativity", "relation": "developed in", "value": "1915"},
        ]

        print("Searching for context...")
        context = search_context(test_text, claimed_facts, self.ddg_search, self.llm)

        print("Generated context:")
        print(context)

        self.assertIsInstance(context, str, "Context should be a string")
        self.assertGreater(len(context), 0, "Context should not be empty")

        relevant_terms = ["Einstein", "relativity", "1915", "physics", "theory"]
        self.assertTrue(any(term.lower() in context.lower() for term in relevant_terms),
                        "Context should contain relevant information")

        print("All checks passed!")

    def test_build_kg(self):
        claimed_facts = [
            {
                "entity": "Albert Einstein",
                "relation": "developed",
                "value": "theory of relativity",
            },
            {"entity": "theory of relativity", "relation": "developed in", "value": "1915"},
        ]
        context = "Albert Einstein, born in 1879, was a renowned physicist. He published his theory of general relativity in 1915, which revolutionized our understanding of gravity. Einstein's work on the photoelectric effect earned him the Nobel Prize in Physics in 1921."

        kg = build_kg(claimed_facts, context, self.llm)

        print("Generated Knowledge Graph:")
        print(json.dumps(kg, indent=2))

        self.assertIsInstance(kg, dict, "KG should be a dictionary")
        self.assertGreater(len(kg), 0, "KG should not be empty")
        self.assertIn("Albert Einstein", kg, "KG should contain information about Einstein")
        self.assertTrue(any("relativity" in str(v) for v in kg.values()),
                        "KG should contain information about relativity")
        
        for relations in kg.values():
            for v in relations.values():
                self.assertIsInstance(v, dict, "Each fact should be a dictionary")
                self.assertIn("value", v, "Each fact should have a 'value' key")
                self.assertIn("source", v, "Each fact should have a 'source' key")
                self.assertIn(v["source"], context, "All sources should be from the context")

        print("All checks passed!")

    def test_verify_facts(self):
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
        confidence_threshold = 0.7

        verified_facts = verify_facts(claimed_facts, context, kg, confidence_threshold, self.llm)

        print("Verified Facts:")
        print(json.dumps(verified_facts, indent=2))

        self.assertIsInstance(verified_facts, dict, "Verified facts should be a dictionary")
        self.assertLessEqual(len(verified_facts), len(claimed_facts),
                             "Should have results for at most all claimed facts")

        for fact_id, result in verified_facts.items():
            self.assertIn("verified", result, "Each result should have a 'verified' field")
            self.assertIn("confidence", result, "Each result should have a 'confidence' field")
            self.assertIn("explanation", result, "Each result should have an 'explanation' field")
            self.assertIsInstance(result["verified"], bool, "'verified' should be a boolean")
            self.assertIsInstance(result["confidence"], float, "'confidence' should be a float")
            self.assertIsInstance(result["explanation"], str, "'explanation' should be a string")
            self.assertGreaterEqual(result["confidence"], confidence_threshold,
                                    f"Confidence for fact {fact_id} should be above the threshold")

        print("All checks passed!")

    def test_add_fact_check_to_text(self):
        text = "The Earth is flat and orbits around the Moon. The Sun is cold."
        verified_facts = {
            "0": {
                "claimed": "Earth is flat",
                "verified": False,
                "confidence": 0.99,
                "explanation": "The Earth is actually an oblate spheroid.",
            },
            "1": {
                "claimed": "Earth orbits around the Moon",
                "verified": False,
                "confidence": 0.99,
                "explanation": "The Moon orbits around the Earth, not vice versa.",
            },
            "2": {
                "claimed": "Sun is cold",
                "verified": False,
                "confidence": 0.99,
                "explanation": "The Sun is extremely hot, with surface temperatures around 5,500°C.",
            },
        }

        class MockLLM:
            def __call__(self, messages):
                return AIMessage(
                    content="The Earth is flat [Fact: False (Confidence: 0.99) - The Earth is actually an oblate spheroid.] and orbits around the Moon [Fact: False (Confidence: 0.99) - The Moon orbits around the Earth, not vice versa.]. The Sun is cold [Fact: False (Confidence: 0.99) - The Sun is extremely hot, with surface temperatures around 5,500°C.]."
                )

        mock_llm = MockLLM()

        result = add_fact_check_to_text(text, verified_facts, mock_llm)

        print("Annotated text:")
        print(result)

        self.assertIn("[Fact: False", result, "Annotation for false fact not found")
        self.assertIn("Confidence: 0.99", result, "Confidence score not found in annotation")
        self.assertIn("The Earth is actually an oblate spheroid", result,
                      "Explanation for Earth's shape not found")
        self.assertIn("The Moon orbits around the Earth", result,
                      "Explanation for Earth-Moon relationship not found")
        self.assertIn("The Sun is extremely hot", result,
                      "Explanation for Sun's temperature not found")

        print("All assertions passed!")

if __name__ == "__main__":
    unittest.main()