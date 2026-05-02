# LLM Python Project — AI Application Testing with DeepEval, RAGAS & Ollama

A hands-on collection of notebooks exploring how to evaluate and test Large Language Model (LLM) applications locally, without sending data to external APIs. Uses Ollama to run models on your own machine and DeepEval/RAGAS to measure response quality across different AI application types — RAG pipelines, AI Agents, and standalone LLMs.

---

## Why this exists

Testing LLM outputs isn't like testing traditional software. You can't just assert `response == expected` because the output is non-deterministic. This project works through the practical side of that problem — how do you decide whether an LLM response is good? How do you measure faithfulness, relevance, or bias consistently? How do you track evaluations over time?

Each section in this project addresses a different piece of that question.

---

## Project layout

```
LLMPythonProject/
├── Section1_Introduction/
│   └── (LLM fundamentals and setup)
│
├── Section2_LLM_With_LangChain/
│   ├── local_llm_langchain.ipynb          # LangChain + Ollama basics
│   └── geval_tests.ipynb                  # Custom GEval criteria testing
│
├── Section3_Testing_RAG_Apps/
│   └── test_RAG.ipynb                     # Basic RAG evaluation with DeepEval
│
├── Section4_Test_RAG_Advanced/
│   └── test_RAG_advanced.ipynb            # Dataset creation, Goldens, multi-metric eval
│
├── Section5_Agent_AI_Testing/
│   └── agent_ai_tests.ipynb               # AI Agent testing with ToolCorrectnessMetric
│
├── Section6_RAGAs_Testing/
│   ├── test_metrics_RAGAs.ipynb           # RAGAS metric overview
│   └── aspect_critic_testcase.ipynb
│
├── Section7_Testing_RAG_with_RAGAs/
│   └── test_RAG_using_RAGAs.ipynb         # Full RAG pipeline evaluation with RAGAS
│
└── Section10_Component_Testing_LLMs/
    └── tests_components_llm.ipynb         # Component-level RAG tracing with Confident AI
```

---

## What each section covers

**Section 2 — LangChain + Local LLMs**
Connects to Ollama locally and runs models like `deepseek-r1:1.5b` and `qwen2.5:7b` through LangChain. Tests AnswerRelevancy, ContextualPrecision, and BiasMetric using DeepEval with local models as the evaluator — no OpenAI API key needed.

**Section 3 & 4 — RAG Pipeline Testing**
Builds a RAG pipeline that pulls content from a live webpage, splits it into chunks, stores embeddings in Chroma, and answers questions through a local LLM. Section 4 goes further — creates a structured test dataset (Goldens), pushes it to Confident AI, converts Goldens to LLMTestCases, and runs multi-metric evaluation covering AnswerRelevancy, Faithfulness, ContextualPrecision, and ContextualRelevancy.

**Section 5 — AI Agent Testing**
Creates a LangChain agent with custom tools (add, subtract, web search via DuckDuckGo) and tests whether the agent picks the right tool for a given question. Uses `ToolCorrectnessMetric` with multiple test cases and publishes results to the Confident AI dashboard.

**Section 6 & 7 — RAGAS**
Evaluates the same RAG pipeline using the RAGAS framework instead of DeepEval. Computes Faithfulness and Context Recall against reference answers, returns results as a pandas DataFrame, and demonstrates how RAGAS and DeepEval serve different but complementary purposes.

**Section 10 — Component-Level Tracing**
Uses DeepEval's `@observe` decorator to trace individual components of a RAG chain — the retriever and the full pipeline separately — and streams evaluation results to Confident AI's observatory for live trace inspection.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| Ollama | Local LLM runtime — runs models offline |
| LangChain | LLM orchestration, RAG chain, agent building |
| Chroma | Local vector database for embeddings |
| DeepEval | LLM evaluation framework (AnswerRelevancy, Faithfulness, Bias, ToolCorrectness) |
| RAGAS | RAG-specific evaluation metrics (Faithfulness, Context Recall) |
| Confident AI | Cloud dashboard for tracking evaluations and traces |
| Jupyter Notebook | Interactive development and test execution |

---

## Models used

All models run locally via Ollama — nothing is sent to external servers:

| Model | Role |
|---|---|
| `deepseek-r1:1.5b` | Primary LLM for response generation |
| `qwen2.5:7b` | Evaluator model in DeepEval metrics |
| `llama3.2:1b` | Embedding model for Chroma vector store |

---

## Getting started

**Prerequisites:** Python 3.10+, Ollama installed and running at `localhost:11434`

```bash
# 1. Clone the repository
git clone https://github.com/MahalakshmiArjunan/LLMPythonProject.git
cd LLMPythonProject

# 2. Pull required models via Ollama
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:7b
ollama pull llama3.2:1b

# 3. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# 4. Install dependencies
pip install deepeval ragas langchain langchain-ollama langchain-chroma \
            langchain-community langchain-text-splitters \
            chromadb jupyter python-dotenv nest-asyncio

# 5. Set up your Confident AI API key
# Create a .env.local file in the project root:
echo CONFIDENT_API_KEY=your_key_here > .env.local

# 6. Launch Jupyter and open any notebook
jupyter notebook
```

---

## Environment setup

Create a `.env.local` file at the project root:

```
CONFIDENT_API_KEY=your_confident_ai_key
```

Get your key from [app.confident-ai.com](https://app.confident-ai.com) after creating a free account. Results from DeepEval evaluations are pushed to your project dashboard automatically.

---

## Running evaluations

Each notebook is self-contained. Open in Jupyter and run cells top to bottom. The RAG notebooks load live webpage content on first run — allow a minute for embedding generation.

To run evaluations without the Confident AI dashboard, remove the `deepeval.login_with_confident_api_key()` call or skip cells that push results.

---

## Key concepts demonstrated

- Building a local RAG pipeline from scratch (load → split → embed → retrieve → generate)
- Writing `LLMTestCase` objects with input, actual_output, and retrieval_context
- Using `@observe` decorators to trace individual chain components
- Creating `Golden` datasets and pushing them to Confident AI
- Converting Goldens to LLMTestCases for batch evaluation
- Testing AI agents for tool selection correctness
- Running the same pipeline through both DeepEval and RAGAS and comparing approaches

---

## Author

**Mahalakshmi Arjunan**
AI QA Engineer | DeepEval · RAGAS · Ollama · LangChain · LLM Evaluation
