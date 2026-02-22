# Query Agent - Pinecone向量检索代理
# Langchain API allowed for this agent
# Model: gpt-4.1-nano


import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


class Query_Agent:
    """
    Query_Agent负责从Pinecone向量数据库中检索相关文档
    
    主要职责：
    1. 初始化Pinecone向量存储和OpenAI embeddings
    2. 根据用户查询检索最相关的文档
    3. 可选：设置系统提示词
    4. 可选：从响应中提取特定动作
    """
    
    def __init__(self, pinecone_index, openai_client, embeddings):
        """
        初始化Query_Agent
        
        参数:
            pinecone_index: Pinecone索引名称，例如 "machine-learning-textbook"
            openai_client: OpenAI客户端实例（本代理中可能不直接使用，但保留接口一致性）
            embeddings: OpenAI embeddings实例，或者在此方法内部创建
        """
        self.pinecone_index = pinecone_index
        self.openai_client = openai_client
        
        # 如果传入的embeddings是None，则自行创建
        if embeddings is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_key
            )
        else:
            self.embeddings = embeddings
        
        # 初始化Pinecone向量存储
        # 使用命名空间ns2500（根据原代码）
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings,
            namespace="ns2500"
        )
        
        # 可选的提示词，用于更高级的功能
        self.prompt = None
    
    def query_vector_store(self, query, k=5):
        """
        从Pinecone向量存储中查询相关文档
        
        参数:
            query: 用户的查询字符串
            k: 返回的top-k相关文档数量，默认为5
            
        返回:
            results: 文档列表，每个文档包含page_content和metadata
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        return results
    
    def set_prompt(self, prompt):
        """
        设置Query_Agent的系统提示词
        
        参数:
            prompt: 系统提示词字符串
            
        说明:
            这个方法允许在需要时动态调整代理的行为
            例如，可以设置特定的检索策略或过滤条件
        """
        self.prompt = prompt
    
    def extract_action(self, response, query=None):
        """
        从响应中提取特定的动作或信息
        
        参数:
            response: LLM或系统的响应
            query: 可选的原始查询，用于上下文
            
        返回:
            提取的动作或处理后的信息
            
        说明:
            这个方法可以用于解析LLM响应，提取结构化信息
            例如，判断是否需要重新检索、调整参数等
            具体实现取决于你的agent架构设计
        """
        # 示例实现：可以根据需要扩展
        # 例如，检查响应中是否包含特定的关键词或指令
        if self.prompt and "ACTION:" in str(response):
            # 提取ACTION标记后的内容
            action_start = str(response).find("ACTION:") + 7
            action = str(response)[action_start:].strip().split('\n')[0]
            return action
        
        # 默认返回None表示没有特殊动作
        return None


