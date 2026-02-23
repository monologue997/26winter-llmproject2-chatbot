# Head Agent - Controller agent that orchestrates all sub-agents
# Coordinates the multi-agent pipeline for modular decision-making
# Model: gpt-4.1-nano

import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# Import all sub-agents
from agents.obnoxious_agent import Obnoxious_Agent
from agents.context_rewriter_agent import Context_Rewriter_Agent
from agents.query_agent import Query_Agent
from agents.relevant_agent import Relevant_Documents_Agent
from agents.answering_agent import Answering_Agent


class Head_Agent:
    """
    Head_Agent is the controller for the multi-agent system.

    Responsibilities:
    1. Initialize and manage all sub-agents
    2. Decide which agents to invoke based on query type
    3. Orchestrate the agent workflow in the correct order
    4. Manage multi-turn conversation history
    5. Return the final response to the user

    Workflow:
    User query -> Context rewrite (multi-turn) -> Obnoxious check -> Vector retrieval
    -> Relevance check -> Answer generation -> Return
    """

    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        """
        Initialize the Head_Agent.

        Args:
            openai_key: OpenAI API key
            pinecone_key: Pinecone API key
            pinecone_index_name: Name of the Pinecone index to query
        """
        # Store configuration
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name

        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_key)

        # Initialize embeddings (used by Query_Agent)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key
        )

        # Conversation history for multi-turn support
        self.conversation_history = []

        # Sub-agent placeholders (initialized to None)
        self.obnoxious_agent = None
        self.context_rewriter_agent = None
        self.query_agent = None
        self.relevant_agent = None
        self.answering_agent = None

        # Initialize all sub-agents
        self.setup_sub_agents()

    def setup_sub_agents(self):
        """
        Instantiate and configure each sub-agent.
        Each agent is initialized independently, reflecting modular design.
        """
        self.obnoxious_agent = Obnoxious_Agent(
            openai_client=self.client
        )

        self.context_rewriter_agent = Context_Rewriter_Agent(
            openai_client=self.client
        )

        self.query_agent = Query_Agent(
            pinecone_index=self.pinecone_index_name,
            openai_client=self.client,
            embeddings=self.embeddings
        )

        self.relevant_agent = Relevant_Documents_Agent(
            openai_client=self.client
        )

        self.answering_agent = Answering_Agent(
            openai_client=self.client
        )

        print("Initialized agents finished")

    def process_query(self, user_query: str, use_history: bool = True) -> dict:
        """
        Main method for processing a user query.

        Args:
            user_query: The user's input string
            use_history: Whether to use conversation history (multi-turn mode)

        Returns:
            A dict containing:
            - response: Final response text
            - agent_path: Agents used during processing (for debugging/evaluation)
            - status: success / error / rejected status
        """
        agent_path = []  # Records which agents were invoked during this query

        try:
            # ===== Step 1: Rewrite query for multi-turn context (if applicable) =====
            # Resolve pronoun references and incomplete phrasing from prior turns
            processed_query = user_query
            if use_history and len(self.conversation_history) > 0:
                agent_path.append("Context_Rewriter_Agent")
                processed_query = self.context_rewriter_agent.rephrase(
                    user_history=self.conversation_history,
                    latest_query=user_query
                )

            # ===== Step 2: Check for obnoxious content or small talk =====
            # Returns "obnoxious" / "small_talk" / "normal"
            agent_path.append("Obnoxious_Agent")
            query_class = self.obnoxious_agent.check_query(processed_query)

            if query_class == "obnoxious":
                response = "I'm sorry, but I can't respond to that type of query. Please ask respectfully."
                agent_path.append("REJECTED_OBNOXIOUS")
                return {
                    "response": response,
                    "agent_path": " → ".join(agent_path),
                    "status": "rejected_obnoxious"
                }

            if query_class == "small_talk":
                response = "Hello! Great to hear from you! I'm doing well and ready to help. Whenever you have questions about Machine Learning — algorithms, models, or anything from the textbook — feel free to ask!"
                agent_path.append("SMALL_TALK")
                return {
                    "response": response,
                    "agent_path": " → ".join(agent_path),
                    "status": "success"
                }

            # ===== Step 3: Retrieve relevant documents from Pinecone =====
            agent_path.append("Query_Agent")
            retrieved_docs = self.query_agent.query_vector_store(
                query=processed_query,
                k=5
            )

            # ===== Step 4: Validate document relevance =====
            agent_path.append("Relevant_Documents_Agent")

            # relevance_result is a bool: True = relevant, False = off-topic
            relevance_result = self.relevant_agent.get_relevance(
                query=processed_query,
                docs=retrieved_docs
            )

            if not relevance_result:  # Query is off-topic
                response = "This query is not relevant to the context of this book. I would be happy to answer questions based on the book's content."
                agent_path.append("REJECTED_IRRELEVANT")
                return {
                    "response": response,
                    "agent_path": " → ".join(agent_path),
                    "status": "rejected_irrelevant"
                }

            # ===== Step 5: Generate the final answer =====
            agent_path.append("Answering_Agent")
            final_response = self.answering_agent.generate_response(
                query=processed_query,
                docs=retrieved_docs,
                conv_history=self.conversation_history if use_history else None,
                k=5
            )

            # ===== Update conversation history =====
            if use_history:
                self.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })

            return {
                "response": final_response,
                "agent_path": " → ".join(agent_path),
                "status": "success"
            }

        except Exception as e:
            return {
                "response": f"An error occurred: {str(e)}",
                "agent_path": " → ".join(agent_path) + " → ERROR",
                "status": "error"
            }

    def reset_conversation(self):
        """
        Clear the conversation history.
        Call this to start a fresh conversation or reset context.
        """
        self.conversation_history = []
        print("Conversation history reset.")

    def get_conversation_history(self):
        """
        Return the current conversation history.

        Returns:
            List of conversation turns (dicts with 'role' and 'content')
        """
        return self.conversation_history

    def main_loop(self):
        """
        Command-line interactive loop for manual testing.
        Provides a simple CLI interface to test the chatbot.
        """
        print("=" * 60)
        print("  Multi-Agent Chatbot - Command Line Interface")
        print("=" * 60)
        print("Commands:")
        print("  - Type your question to get an answer")
        print("  - Type 'reset' to clear conversation history")
        print("  - Type 'history' to view conversation")
        print("  - Type 'quit' or 'exit' to exit")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue

                if user_input.lower() == 'history':
                    print("\n Conversation History:")
                    for msg in self.conversation_history:
                        print(f"  {msg['role']}: {msg['content'][:100]}...")
                    continue

                # Process query
                print("\n Processing...")
                result = self.process_query(user_input)

                print(f"\n Assistant: {result['response']}")
                print(f"\n Agent Path: {result['agent_path']}")
                print(f" Status: {result['status']}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n Error: {str(e)}")


if __name__ == "__main__":
    # Load API keys from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key or not pinecone_key:
        print("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        exit(1)

    # Create Head Agent
    head_agent = Head_Agent(
        openai_key=openai_key,
        pinecone_key=pinecone_key,
        pinecone_index_name="machine-learning-textbook"
    )

    # Run interactive loop
    head_agent.main_loop()
