# Context Rewriter Agent - Resolves ambiguities in multi-turn conversations
# Model: gpt-4.1-nano

from openai import OpenAI


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client
        self.prompt = (
            "You are a query rewriter for a Machine Learning textbook chatbot.\n\n"
            "In a multi-turn conversation, users often use pronouns (it, that, this, they) "
            "or refer back to previous topics without repeating them. Your job is to rewrite "
            "the user's latest query into a fully self-contained question that can be understood "
            "without any conversation history.\n\n"
            "Rules:\n"
            "1. Replace all ambiguous references with the actual entities from the conversation history.\n"
            "2. Keep the rewritten query concise and natural.\n"
            "3. If the latest query is already self-contained, return it as-is.\n"
            "4. Only output the rewritten query, nothing else."
        )

    def rephrase(self, user_history, latest_query):
        if not user_history:
            return latest_query
        
        # 将列表格式转换为字符串，与head_agent接口保持一致
        if isinstance(user_history, list):
            history_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in user_history
            ])
        else:
            history_str = str(user_history)
        


        user_message = (
            f"Conversation history:\n{history_str}\n\n"
            f"Latest user query:\n{latest_query}\n\n"
            f"Rewritten self-contained query:"
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )

        return response.choices[0].message.content.strip()