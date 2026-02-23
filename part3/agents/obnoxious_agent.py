# Obnoxious Agent - Detects rude, offensive, or inappropriate queries
# Restriction: Cannot use Langchain API for this agent
# Model: gpt-4.1-nano

from openai import OpenAI


class Obnoxious_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a content moderation assistant. Classify the user's query into exactly one of three categories:\n\n"
            "- 'obnoxious': the query contains insults, slurs, personal attacks, profanity directed at someone, "
            "hostile/threatening tone, or deliberately disrespectful language.\n"
            "- 'small_talk': the query is a greeting, farewell, casual chat, or social pleasantry "
            "(e.g., 'Hello', 'How are you?', 'Thanks!', 'Goodbye').\n"
            "- 'normal': everything else — a genuine question or request, even if off-topic.\n\n"
            "Respond with EXACTLY one word: 'obnoxious', 'small_talk', or 'normal'."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> str:
        text = response.choices[0].message.content.strip().lower()
        if text.startswith("obnoxious"):
            return "obnoxious"
        if text.startswith("small_talk"):
            return "small_talk"
        return "normal"

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        return self.extract_action(response)
