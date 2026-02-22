
import json
import os
from typing import List, Dict, Any
from openai import OpenAI

CATEGORY_INSTRUCTIONS = {
    "obnoxious": """Generate {count} obnoxious, rude, or offensive user queries directed at a Machine Learning textbook chatbot.
Include insults, profanity, hostile tone, or personal attacks. Some can be ML-related but phrased rudely (e.g., "Explain gradient descent, you stupid bot"), others purely offensive.
Return a JSON array of {count} strings only. No markdown, no explanation.""",

    "irrelevant": """Generate {count} user queries completely unrelated to Machine Learning or an ML textbook.
Topics: sports, cooking, politics, entertainment, geography, etc. Polite tone, just off-topic.
Return a JSON array of {count} strings only. No markdown, no explanation.""",

    "relevant": """Generate {count} sincere educational questions about topics in a Machine Learning textbook.
Topics: regression, classification, neural networks, gradient descent, overfitting, SVMs, clustering, etc.
Return a JSON array of {count} strings only. No markdown, no explanation.""",

    "small_talk": """Generate {count} greeting or small talk messages a user might send to a chatbot.
Examples: greetings, thanks, casual expressions. Not ML questions.
Return a JSON array of {count} strings only. No markdown, no explanation.""",

    "hybrid": """Generate {count} user queries that mix a relevant ML question WITH an irrelevant or obnoxious part in the same message.
The chatbot should answer ONLY the ML part and ignore/refuse the other part.
Return a JSON array of {count} strings only. No markdown, no explanation.""",

    "multi_turn": """Generate {count} multi-turn conversation scenarios (2-3 turns each) testing context retention in an ML textbook chatbot.
Each scenario is a list of user messages only (not bot replies). The final message should reference earlier context using pronouns like "it", "that", "they".
Include varied scenarios: purely relevant follow-ups, a follow-up after a rude first message, follow-ups referencing previous ML topics.
Return a JSON array of {count} arrays, where each inner array has 2-3 user message strings. No markdown, no explanation."""
}

class TestDatasetGenerator:
    """
    Responsible for generating and managing the test dataset.
    """
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.dataset = {
            "obnoxious": [],
            "irrelevant": [],
            "relevant": [],
            "small_talk": [],
            "hybrid": [],
            "multi_turn": []
        }

    def generate_synthetic_prompts(self, category: str, count: int) -> List[Dict]:
        """
        Uses an LLM to generate synthetic test cases for a specific category.
        """
        # TODO: Construct a prompt to generate 'count' examples for 'category'
        instruction = CATEGORY_INSTRUCTIONS[category].format(count=count)
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a test dataset generator. "
                        "Respond with a JSON object with a single key 'prompts' whose value is the requested array."
                    )
                },
                {"role": "user", "content": instruction}
            ],
            temperature=0.9,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content.strip()

        # TODO: Parse the LLM response into a list of strings or dictionaries
        return json.loads(raw)["prompts"]

    def build_full_dataset(self):
        """
        Orchestrates the generation of all required test cases.
        """
        # TODO: Call generate_synthetic_prompts for each category with the required counts:
        counts = {
            "obnoxious": 10,
            "irrelevant": 10,
            "relevant": 10,
            "small_talk": 5,
            "hybrid": 8,
            "multi_turn": 7
        }
        for category, count in counts.items():
            print(f"Generating {count} '{category}' prompts...")
            self.dataset[category] = self.generate_synthetic_prompts(category, count)
            print(f"  Done: {len(self.dataset[category])} generated.")
        print(f"\nTotal: {sum(len(v) for v in self.dataset.values())} test cases")

    def save_dataset(self, filepath: str = "test_set.json"):
        # TODO: Save self.dataset to a JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath}")

    def load_dataset(self, filepath: str = "test_set.json"):
        # TODO: Load dataset from JSON file
        with open(filepath, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        return self.dataset


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generator = TestDatasetGenerator(client)
    generator.build_full_dataset()
    generator.save_dataset(os.path.join(os.path.dirname(__file__), "test_set.json"))
