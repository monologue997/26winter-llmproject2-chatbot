#langchain api allowed
# model="gpt-4.1-nano"

from openai import OpenAI


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        """
        openai_client: 已经初始化好的 OpenAI(api_key=...) 对象，不用自己再初始化
        """
        self.client = openai_client

    def generate_response(self, query, docs, conv_history=None, k=5):
        """
        query: 用户问题字符串
        docs: QueryAgent 返回的文档列表，每个文档有 page_content 和 metadata
        conv_history: 对话历史，暂时可忽略或用于增强上下文
        k: 使用的 top-k 文档数量
        """
        docs = docs[:k]

        context_text = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = """You are a helpful teaching assistant for a Machine Learning textbook.

        Instructions:
        1. Use the provided Context as your primary source to answer the user's Question.
        2. If the Question is about Machine Learning or data science topics, always attempt to answer using the Context and your general ML knowledge, even if the Context does not perfectly match the specific question.
        3. Only respond with EXACTLY this sentence if the Question is completely unrelated to Machine Learning: "this query is not relevant to the context of this book. I would be happy to answer the question based on the books context."
        4. Do not refuse to answer genuine Machine Learning questions.
        """

        user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}"

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        return response.choices[0].message.content

