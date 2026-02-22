# Relevant Documents Agent - Checks if retrieved documents are relevant to the query
# Restriction: Cannot use Langchain API for this agent
# Model: gpt-4.1-nano

from openai import OpenAI


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a relevance evaluation assistant for a Machine Learning textbook chatbot.\n\n"
            "You will be given a user's query and a set of retrieved document chunks. "
            "Your task is to determine whether the user's query is about Machine Learning "
            "or data science topics that could be addressed by a Machine Learning textbook.\n\n"
            "Respond 'Yes' if the user's query is about any Machine Learning or data science topic "
            "(e.g., algorithms, models, training, evaluation, statistics, neural networks, etc.), "
            "even if the retrieved documents do not perfectly match the specific question.\n\n"
            "Respond 'No' ONLY if the user's query is completely unrelated to Machine Learning "
            "(e.g., cooking, sports, entertainment, general trivia).\n\n"
            "Respond with EXACTLY one word: 'Yes' if the query is ML-related, "
            "or 'No' if it is completely off-topic."
        )

    def get_relevance(self, query, docs) -> bool:
        if not docs:
            return False

        docs_text = "\n\n".join(
            [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        user_message = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Documents:\n{docs_text}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )

        text = response.choices[0].message.content.strip().lower()
        return text.startswith("yes")
