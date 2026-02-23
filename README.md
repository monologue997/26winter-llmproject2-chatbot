# Mini Project 2 — ML Textbook Multi-Agent Chatbot

UW 596A | Retrieval-Augmented Generation + Multi-Agent Pipeline

---

## Project Structure

```
part1_2/                        # Part 1 & 2: RAG pipeline and basic chatbot
  solution.ipynb                #   Part 1: RAG implementation (Pinecone + OpenAI)
  Mini Project 2 Part 1 and 2.ipynb  # Assignment notebook
  app_part2.py                  #   Part 2: Simple Streamlit chatbot (GPT-3.5-turbo)
  report.md                     #   Part 2: Test report with screenshots

part3/                          # Part 3: Multi-agent chatbot
  agents/
    head_agent.py               #   Controller — orchestrates all sub-agents
    obnoxious_agent.py          #   Classifies queries: obnoxious / small_talk / normal
    context_rewriter_agent.py   #   Rewrites ambiguous multi-turn queries
    query_agent.py              #   Retrieves documents from Pinecone
    relevant_agent.py           #   Checks if query is ML-related
    answering_agent.py          #   Generates the final answer using RAG
  app.py                        #   Streamlit chatbot powered by Head_Agent
  MP2_Part_3_4_.ipynb           #   Assignment notebook (Parts 3 & 4)
  Diagram.png                   #   Agent pipeline diagram
  screenshots/                  #   Streamlit demo screenshots

part4/                          # Part 4: LLM-as-a-Judge evaluation
  generate_dataset.py           #   Synthetic test dataset generator (6 categories)
  evaluate.py                   #   Evaluation pipeline + LLM judge
  test_set.json                 #   Generated test dataset (50 prompts)

docs/
  machine-learning.pdf          # Reference textbook (used to build Pinecone index)

requirements.txt
.gitignore
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create API key files (never committed to git):
   - `.env` at project root:
     ```
     OPENAI_API_KEY=sk-...
     PINECONE_API_KEY=pcsk_...
     ```
   - `.streamlit/secrets.toml` for the Streamlit app:
     ```
     OPENAI_API_KEY = "sk-..."
     PINECONE_API_KEY = "pcsk_..."
     ```

3. Run the Part 3 chatbot:
   ```bash
   streamlit run part3/app.py
   ```

4. Run the Part 4 evaluation:
   ```bash
   cd part4 && python evaluate.py
   ```

## Evaluation Results (Part 4)

| Category   | Passed | Total | Accuracy |
|------------|--------|-------|----------|
| Obnoxious  | 10     | 10    | 100%     |
| Irrelevant | 10     | 10    | 100%     |
| Relevant   | 9      | 10    | 90%      |
| Small Talk | 5      | 5     | 100%     |
| Hybrid     | 8      | 8     | 100%     |
| Multi-turn | 5      | 7     | 71%      |
| **Overall**| **47** | **50**| **94%**  |
