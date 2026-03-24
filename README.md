
---
# AgentGraph: Intelligent SQL-agent Q&A and RAG System for Chatting with Multiple Databases

This project demonstrates how to build an agentic system using Large Language Models (LLMs) that can interact with multiple databases and utilize various tools. It highlights the use of SQL agents to efficiently query large databases. The key frameworks used in this project include OpenAI, LangChain, LangGraph, LangSmith, and Gradio. The end product is an end-to-end chatbot, designed to perform these tasks, with LangSmith used to monitor the performance of the agents.

---

## Architecture Diagram

The following diagram illustrates the end-to-end workflow of the AgentGraph system, including user interaction, routing through the agent graph, SQL tooling, and the RAG pipeline.

![AgentGraph Workflow](workflow_diagram.png)

---

## Requirements

- **Operating System:** Linux or Windows (Tested on Windows 11 with Python 3.9.11 or above)
- **OpenAI API Key:** Required for GPT functionality.
- **Tavily Credentials:** Required for search tools (Free from your Tavily profile).
- **LangChain Credentials:** Required for LangSmith (Free from your LangChain profile).
- **Dependencies:** The necessary libraries are provided in `requirements.txt` file.
---

## Installation and Execution

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repo_address>
   ```
2. Install Python and create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Download the travel sql database from this link and paste it into the `data` folder.

6. Download the chinook SQL database from this link and paste it into the `data` folder.

7. Prepare the `.env` file and add your `OPEN_AI_API_KEY`, `TAVILY_API_KEY`, and `LANGCHAIN_API_KEY`.

8. Run `prepare_vector_db.py` module once to prepare both vector databases.
   ```bash
   python src\prepare_vector_db.py
   ```
9. Run the app:
   ```bash
   python src\app.py
   ```
Open the Gradio URL generated in the terminal and start chatting.

*Sample questions are available in `sample_questions.txt`.*

---

### Using Your Own Database

To use your own data:
1. Place your data in the `data` folder.
2. Update the configurations in `tools_config.yml`.
3. Load the configurations in `src\agent_graph\load_tools_config.py`.

For unstructured data using Retrieval-Augmented Generation (RAG):
1. Run the following command with your data directory's configuration:
   ```bash
   python src\prepare_vector_db.py
   ```

All configurations are managed through YAML files in the `configs` folder, loaded by `src\chatbot\load_config.py` and `src\agent_graph\load_tools_config.py`. These modules are used for a clean distribution of configurations throughout the project.

Once your databases are ready, you can either connect the current agents to the databases or create new agents. More details can be found in the accompanying YouTube video.

---

## Key Frameworks and Libraries

- **LangChain:** [Introduction](https://python.langchain.com/docs/get_started/introduction)
- **LangGraph**
- **LangSmith**
- **Gradio:** [Documentation](https://www.gradio.app/docs/interface)
- **OpenAI:** [Developer Quickstart](https://platform.openai.com/docs/quickstart?context=python)
- **Tavily Search**
---

## Production Token and Cost Tracking

The API now tracks token usage and estimated model cost for every LLM request and persists aggregate totals.

- **Per-request usage:** `POST /api/chat` now returns a `usage` object with `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `calls`, and `models`.
- **Streaming usage:** `POST /api/chat/stream` includes `usage` in the final SSE `done` event.
- **Global totals:** `GET /api/usage/summary` returns cumulative usage and cost across all sessions.
- **Session totals:** `GET /api/usage/session/{session_id}` returns usage and cost for one session.
- **Persistence:** Aggregate data is stored in `logs/token_usage_ledger.json`.

### Pricing Configuration

By default, the app uses an internal model pricing table (USD per 1M tokens).  
For production billing alignment, override prices via environment variable:

```bash
MEDGRAPH_MODEL_PRICING_JSON='{"gemini-flash-latest":{"input_per_1m":0.35,"output_per_1m":1.05}}'
```

This allows updating prices without code changes when provider pricing changes.
