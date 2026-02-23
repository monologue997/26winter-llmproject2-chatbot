# Query Agent - Pinecone vector retrieval agent
# Langchain API allowed for this agent
# Model: gpt-4.1-nano


import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


class Query_Agent:
    """
    Query_Agent retrieves relevant documents from the Pinecone vector database.

    Responsibilities:
    1. Initialize the Pinecone vector store and OpenAI embeddings
    2. Retrieve the most relevant documents for a given user query
    3. Optional: set a system prompt for advanced retrieval behavior
    4. Optional: extract specific actions from LLM responses
    """

    def __init__(self, pinecone_index, openai_client, embeddings):
        """
        Initialize the Query_Agent.

        Args:
            pinecone_index: Pinecone index name, e.g. "ml-mp2"
            openai_client: OpenAI client instance (kept for interface consistency)
            embeddings: OpenAIEmbeddings instance, or None to create one internally
        """
        self.pinecone_index = pinecone_index
        self.openai_client = openai_client

        # Create embeddings internally if none are provided
        if embeddings is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_key
            )
        else:
            self.embeddings = embeddings

        # Initialize Pinecone vector store using namespace ns2500
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings,
            namespace="ns2500"
        )

        # Optional prompt for advanced retrieval features
        self.prompt = None

    def query_vector_store(self, query, k=5):
        """
        Query the Pinecone vector store for relevant documents.

        Args:
            query: User query string
            k: Number of top-k documents to return (default 5)

        Returns:
            results: List of documents, each with page_content and metadata
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        return results

    def set_prompt(self, prompt):
        """
        Set the system prompt for this agent.

        Args:
            prompt: Prompt string to guide retrieval behavior

        Note:
            This allows dynamic adjustment of the agent's behavior at runtime,
            e.g. applying custom retrieval strategies or filters.
        """
        self.prompt = prompt

    def extract_action(self, response, query=None):
        """
        Extract a specific action or structured information from an LLM response.

        Args:
            response: LLM or system response
            query: Optional original query for context

        Returns:
            Extracted action string, or None if no action found

        Note:
            This method can parse LLM output to extract structured information,
            e.g. determining whether a re-retrieval or parameter adjustment is needed.
        """
        # Example: extract content after an "ACTION:" marker if present
        if self.prompt and "ACTION:" in str(response):
            action_start = str(response).find("ACTION:") + 7
            action = str(response)[action_start:].strip().split('\n')[0]
            return action

        # Default: no special action
        return None
