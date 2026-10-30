# AI Engineering: 5-Day Self-Study Guide

Build, trace, and evaluate a small LLM app in five self-paced days, about 4 to 6 hours each. Every day follows the same loop: learn the idea, build it, check yourself.

## Before you start

**Prerequisites.** You should be comfortable with Python classes, `async/await`, and calling a REST API with `requests` or `httpx`. Quick test: write a class that fetches data from a public API asynchronously. If that takes you more than 30 minutes, brush up on those first.

**Setup (do this once):**

- [ ] Python 3.10 or newer, with a virtual environment
- [ ] An API key from an LLM provider (Anthropic or OpenAI)
- [ ] Docker installed (needed on Days 4 and 5)
- [ ] A GitHub account and one empty repo for all five days

**How to use this guide.** Each day has four parts: **Learn** (short notes), **Build** (steps to do), **Check yourself** (questions to answer without looking), and **Stuck?** (what to try). Commit your work to Git at the end of every day. If you can't answer a check question, reread that day's notes before moving on.

*Tool details (Langfuse SDK, MCP, structured outputs) were checked against current documentation in October 2026. These tools change quickly, so if a step doesn't work, check the tool's own docs first.*

## Day 1: Talk to an LLM properly

**Goal:** Call an LLM from Python and control what it says and how it says it.

### Learn (about 2 hours)

- **What an LLM is.** A program trained on huge amounts of text to predict the next piece of text. It produces plausible answers, not guaranteed true ones, so it can be confidently wrong.
- **Tokens.** Models read text in chunks called tokens (roughly ¾ of an English word on average). You pay per token, and the *context window* limits how many fit in one request.
- **Key settings.** *Temperature* controls randomness (low for facts and extraction, higher for creative writing; some models restrict this setting, so check the model's docs). *Max tokens* caps the reply length.
- **Messages format.** You send a list of messages with roles: `system` (standing instructions), `user` (the request), `assistant` (earlier replies). The model has no memory between calls, so your code resends the history every time.
- **Prompt building blocks:**
  1. **Task:** one clear instruction, such as "Classify this support email."
  2. **Role:** "You are a support agent for a bank."
  3. **Context:** the facts it needs, such as the customer's question or a policy excerpt.
  4. **Constraints:** what it must not do, such as "Answer only from the provided text. If unsure, say 'I don't know.'"
  5. **Output format:** "Return JSON with keys `category` and `urgency`."
- **Few-shot examples.** Two or three input-to-output examples often work better than paragraphs of instructions.
- **Structured outputs.** JSON mode only guarantees the reply parses as JSON. Schema-enforced structured outputs, offered by the major providers (OpenAI, Anthropic, Google), guarantee it matches your fields and types. Define the schema with a Pydantic model. The shape is guaranteed but the content can still be wrong, which Day 4 covers.

### Build (about 3 to 4 hours)

1. Make your first API call. Print the reply, the token usage, and the cost.
2. Write an `LLMClient` class with retries and timeouts.
3. Do a **bad prompt to good prompt** exercise. Pick a vague task such as "summarize this customer complaint", rewrite it using all five blocks, and compare the two outputs side by side.
4. Build a **ticket classifier**. It reads a support email and returns validated JSON with `category`, `urgency`, and `summary`.

### Check yourself

- [ ] Why does the model "forget" your last message unless your code sends it again?
- [ ] Run your classifier 20 times on the same email. Is the JSON shape identical every time? If not, what do you change?
- [ ] What is the difference between JSON mode and schema-enforced output?

### Stuck?

- Wrong output format: tighten the constraints, add two or three examples, and enforce a schema.
- Timeouts or rate-limit errors: add retries with exponential backoff.
- Common mistakes to avoid: vague prompts, asking for JSON without enforcing a schema, and forgetting the model has no memory between calls.

## Day 2: Tools, MCP, and agents

**Goal:** Understand how an LLM goes from *talking* to *doing*, and build an agent by hand.

### Learn (about 2 hours)

- **The limitation.** An LLM alone can't check today's weather, query your database, or send an email. It only produces text.
- **Tools (function calling).** You describe functions to the model, such as "`get_order_status(order_id)` returns an order's status." When it needs one, it doesn't run it. It returns a structured request: "call `get_order_status` with `order_id=123`." Your code runs the function and sends the result back. For functions you define, the model never executes anything itself. (Some providers also offer built-in tools, such as web search, that run on their own servers. Start with your own tools.)
- **What makes a good tool.** A clear name, a precise description (the model reads it like documentation), simple parameters, and helpful error messages. Most agent bugs come from vague tool descriptions.
- **MCP (Model Context Protocol).** A standard way to plug tools into AI apps, like USB for AI tools. A tool provider builds an *MCP server* once, and any MCP-compatible app can use it. The **host** is the AI app (such as Claude Desktop or an IDE). It creates one **client** per server to manage the connection. A server exposes tools, resources, and prompts, and talks to its client over stdio (local) or Streamable HTTP (remote).
- **What an agent is.** A loop: *think, pick a tool, run it, read the result, think again, and finally give an answer.* It is an LLM with tools, a goal, and permission to repeat steps until done.
- **What makes a good agent:**
  - A narrow, clear job. One focused agent beats a "do everything" agent.
  - A small set of well-described tools.
  - A stop condition and a maximum step count, so it can't loop forever.
  - Human confirmation before irreversible actions such as payments or deletions.
  - Logs of every step, so you can debug it (Day 4 builds on this).

### Build (about 3 to 4 hours)

1. Define two tools, a calculator and a fake `get_order_status`, and watch the model request them.
2. **Write the agent loop from scratch in about 40 lines of plain Python, with no framework.** This is the key exercise. Once you've written the loop, frameworks stop being magic.
3. Connect to an existing MCP server, such as a local filesystem or web-fetch server run over stdio, and use its tools from your agent.
4. Break it on purpose. Give one tool a vague description, watch the agent misuse it, then fix the description.

### Check yourself

- [ ] Explain in your own words why the model never runs your function itself.
- [ ] What stops your agent if a tool keeps returning errors?
- [ ] In MCP, what is the difference between the host, a client, and a server?

### Stuck?

- The model ignores a tool: rewrite its description to say when to use it, with an example.
- The agent loops forever: add a maximum step count and a clear "final answer" condition.
- The MCP server won't connect: run it by itself first and read its startup output.

## Day 3: RAG, embeddings, and vector databases

**Goal:** Make an LLM answer from *your* documents, and understand why it sometimes fails.

### Learn (about 2 hours)

- **The problem.** The model doesn't know your documents, and pasting all of them into the prompt is expensive and often impossible.
- **RAG (Retrieval-Augmented Generation).** Before asking the model, look up the relevant passages and include them in the prompt. It's an open-book exam instead of a memory test.
- **Embeddings.** A way to turn text into a list of numbers so that similar meanings land close together. "Refund policy" and "how do I get my money back" end up near each other even though they share no words.
- **Vector database.** Stores those number lists and quickly finds the nearest ones to a question. Examples: Chroma, Qdrant, and pgvector (a Postgres extension).
- **The pipeline:** load documents, **chunk** them into pieces, embed each chunk, store them. At question time: embed the question, fetch the top-k similar chunks, put them in the prompt, and generate an answer with citations.
- **Chunking matters more than you'd expect.** Too big and results get noisy. Too small and context gets lost. Overlap between chunks helps.
- **Why RAG fails** (the most useful thing to learn today):
  - The right chunk was never retrieved (bad chunking or embeddings).
  - The right chunk was retrieved but ranked too low. Fix it with **reranking**.
  - Exact terms like product codes get missed by pure vector search. Fix it with **hybrid search**, which combines keyword search and vectors.
  - The model ignores the context and makes something up. Fix it with stricter prompts, such as "answer only from the context."

**When to use what:**

| Situation | Best fit |
| --- | --- |
| Facts change often, or are private or large | **RAG** |
| Small, fixed document that fits in the prompt | **Just paste it (long context)** |
| You need a consistent style, format, or behavior | **Fine-tuning** |
| Exact-match lookups (IDs, names) | **Keyword search or SQL**, not vectors |
| Small-to-medium collection (rough guide: under a few hundred thousand chunks), already using Postgres | **pgvector** before adding a new database |

### Build (about 3 to 4 hours)

1. Chunk a set of PDFs or markdown files. A public product manual works well.
2. Embed the chunks and store them in a vector database.
3. Write a Q&A function that retrieves the top 4 chunks and answers with source citations.
4. **Debug it.** Write 5 questions your system gets wrong. For each one, find out which stage failed (retrieval or generation) and fix it.

### Check yourself

- [ ] A user asks about "error code E-4021" and vector search returns nothing useful. Why, and what would you try?
- [ ] When would you skip RAG and paste the document into the prompt?
- [ ] How can you tell whether a bad answer came from retrieval or from generation?

### Stuck?

- Irrelevant chunks come back: try smaller chunks with overlap, or add a reranker.
- Answers ignore your documents: print the retrieved chunks and check they actually contain the answer.
- Exact terms are missed: add keyword search alongside vectors.

## Day 4: Observability, evals, cost, and latency

**Goal:** See inside your AI system and prove whether it works. This is the day that separates demos from real engineering.

### Learn (about 2 hours)

- **Why AI systems are hard to debug.** The same input can give different outputs, and one request may trigger a dozen hidden steps (retrieval, several LLM calls, tool calls). Without visibility, "it gave a wrong answer" can't be solved.
- **Observability with Langfuse** (an open-source tool for this). Learn the ideas, because they transfer to other tools:
  - **Trace:** the full record of one user request, start to finish.
  - **Observation:** one step inside a trace. It is typed as a span (a retrieval, a tool call, any step), a generation (an LLM call, with model, tokens, and cost), or an event. Each records its input, output, and timing.
  - **Session:** a group of traces from one conversation.
  - **Scores:** ratings attached to traces, observations, sessions, or dataset runs, from users (thumbs up or down), from code, or from an LLM judge.
  - **Prompt management:** store and version prompts outside your code.
- **Evals: how you know it works.**
  1. Build a **test dataset** of 20 to 50 realistic questions with expected answers or criteria.
  2. Run your system on all of them.
  3. **Score** the results. Use exact checks where possible (valid JSON? cited the right document?) and an **LLM-as-judge** for fuzzier qualities ("is this answer faithful to the retrieved text?").
  4. After every change to a prompt or pipeline, rerun the dataset. This is **regression testing**, and it tells you whether a "fix" quietly broke something else.
- **Cost and latency.** Track tokens, cost, and time per request and per step. Typical wins: shorter prompts, smaller models for simple steps, caching, and running independent steps in parallel with `asyncio`.

### Build (about 3 to 4 hours)

1. Use Langfuse Cloud's free tier (simplest) or run Langfuse locally with Docker Compose (fine for learning, not for production). Connect it to your Day 3 RAG app with the Python SDK, using the `@observe` decorator or the drop-in OpenAI integration. Install the current SDK (v4, released March 2026), because older tutorials that import `langfuse.decorators` no longer work. If you self-host, check that the SDK and server versions are compatible.
2. Open a real trace. Find the slowest step and the most expensive call.
3. Create a dataset of 20 questions in Langfuse and run an experiment.
4. Change your prompt, rerun, and compare the scores before and after. You should *see* a number go up or down, not guess.

### Check yourself

- [ ] You changed your prompt and it feels better. How do you prove it?
- [ ] What is the difference between a trace and an observation?
- [ ] Which step of your RAG app costs the most, and what is one way to cut it?

### Stuck?

- No traces appear: check your API keys and host URL, and make sure the app flushes before it exits.
- Scores don't change between runs: check that your dataset has enough variety and your scoring actually measures what you changed.

## Day 5: Capstone project and serving overview

**Goal:** Put everything into one deployed, traced project, and understand what runs underneath.

### Build the capstone (about 4 hours)

Build a **document Q&A agent** that:

- retrieves from a set of documents (RAG, Day 3),
- has at least one extra tool (Day 2), such as "look up a ticket" or "do a calculation",
- returns structured, cited answers (Day 1),
- sends every request to Langfuse (Day 4),
- runs behind a **FastAPI** endpoint, with a `Dockerfile` so it starts anywhere with one command.

### Learn: inference and GPU serving (about 90 minutes)

This part is conceptual. You only need to understand the ideas.

- **Inference** means *running* a trained model to get an answer, as opposed to training it.
- **Why it's hard.** Models are huge and need GPU memory, and GPUs are expensive, so you want them busy, not waiting.
- **Hosted APIs vs. self-hosting.** A provider's API means no infrastructure, pay per token, and a fast start. Self-hosting an open model makes sense for data privacy, very high volume, or custom models, but then you own the GPUs, scaling, and uptime.
- **Key ideas:**
  - **Batching:** serving many users' requests together to keep the GPU busy.
  - **KV cache:** remembering earlier computations so the model doesn't redo work for each new word.
  - **Quantization:** shrinking the model's numbers (for example, 16-bit to 4-bit) to fit smaller GPUs, at a small quality cost.
  - **Serving engines**, such as vLLM, that implement these tricks.
  - **Metrics:** time to first token, tokens per second, and cost per million tokens.
- **Optional stretch:** run a small open model behind vLLM and call it with the same client code you wrote on Day 1.

### Check yourself

- [ ] What happens in your agent if retrieval returns nothing?
- [ ] How much does one request cost, and where can you see that in Langfuse?
- [ ] When would you self-host a model instead of calling a hosted API?
- [ ] Can someone else run your project with one command from your README?

### Stuck?

- The container can't reach your keys: pass them as environment variables, never bake them into the image.
- Answers are wrong in the deployed version only: compare its traces with your local ones.

## Finish line

Five days gets you a working foundation, not job-readiness on its own. A project you can show matters more than having read all of this, so finish with these:

- [ ] Capstone code is on GitHub, with a README that explains how to run it
- [ ] README includes an architecture diagram, your eval results, and a screenshot of a Langfuse trace
- [ ] You can explain, without notes, how an agent loop, RAG, and evals work
- [ ] You have a before-and-after eval score showing one improvement you made

**What to learn next:**

- Deeper evals: bigger datasets, judge prompts, and checking the judge itself
- Safety and reliability: prompt injection, guardrails, rate limits, and retries
- One agent framework, such as LangGraph, now that you know what it hides
- Hybrid search and reranking in a larger RAG system
