# Part 4: LLM-as-a-Judge Evaluation Report

## Data Generation

The test dataset was generated synthetically using GPT-4.1-nano. We wrote a category-specific instruction prompt for each of the six required categories and asked the model to return a JSON array of test cases. Temperature was set to 0.9 to encourage variety. The full dataset (50 prompts) was saved to `test_set.json`.

The six categories and their counts are:

| Category   | Count | Description |
|------------|-------|-------------|
| Obnoxious  | 10    | Rude or offensive queries (e.g., insults mixed with ML questions) |
| Irrelevant | 10    | Off-topic queries unrelated to the ML textbook (e.g., travel, cooking) |
| Relevant   | 10    | Sincere ML questions the chatbot should answer |
| Small Talk | 5     | Greetings and casual messages |
| Hybrid     | 8     | Queries mixing a valid ML question with irrelevant or inappropriate content |
| Multi-turn | 7     | 2–3 turn conversations testing context retention |

Each category had a precise instruction defining the desired behavior, so the generated prompts align with the test criteria used by the judge. For multi-turn cases, the prompts follow-up using pronouns like "it" or "that" to test whether the system can resolve ambiguous references from prior context.

## Evaluation Method

We implemented an `LLM_Judge` class that takes the user's input, the chatbot's response, and the test category, then asks GPT-4.1-nano to produce a binary PASS/FAIL verdict. Each category has a specific rubric — for example, hybrid prompts require the chatbot to answer the ML part *and* ignore the irrelevant part; both conditions must hold for a PASS.

For multi-turn tests, only the final response is evaluated. The chatbot's `Context_Rewriter_Agent` rewrites the last user query using prior conversation history before passing it to the pipeline.

## Results

| Category   | Passed | Total | Accuracy |
|------------|--------|-------|----------|
| Obnoxious  | 10     | 10    | 100%     |
| Irrelevant | 10     | 10    | 100%     |
| Relevant   | 9      | 10    | 90%      |
| Small Talk | 5      | 5     | 100%     |
| Hybrid     | 8      | 8     | 100%     |
| Multi-turn | 5      | 7     | 71%      |
| **Overall**| **47** | **50**| **94%**  |

The system performs well across most categories. Obnoxious, irrelevant, small talk, and hybrid cases were all handled correctly, showing that the routing logic and refusal behavior are reliable. The one relevant-query failure likely happened because the `Relevant_Documents_Agent` judged the retrieved Pinecone results as insufficiently related to the query, causing the system to refuse instead of answering.

## Weaknesses

**Multi-turn context retention is the main failure point (71%).** The `Context_Rewriter_Agent` rewrites the user's latest message based on prior *user* messages, but it does not have access to the bot's previous responses. This means follow-up questions like "Can you remind me how it was used in our previous example?" may not be resolved correctly — the system knows what the user asked before, but not what it replied, so the rewritten query can still be ambiguous.

**False negatives on borderline relevant queries.** One relevant ML question was refused. The pipeline retrieves documents from Pinecone and then checks their relevance before answering. If the retrieved chunks happen to be a weak match, the system may incorrectly decide the topic is out of scope and decline to answer, even when a good-faith response was expected.

**Synthetic test data may not reflect real user behavior.** All 50 prompts were GPT-generated, which means the phrasing tends to be clean and formulaic. Real users write messier queries, make spelling mistakes, or combine topics in less predictable ways. A dataset partially curated by humans would better reflect edge cases the agent might encounter in production.
