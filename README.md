# LangChain Academy

## Introduction

Welcome to LangChain Academy!
This is a growing set of modules focused on foundational concepts within the LangChain ecosystem.
Module 0 is basic setup and Modules 1 - 4 focus on LangGraph, progressively adding more advanced themes.
In each module folder, you'll see a set of notebooks. A LangChain Academy accompanies each notebook
to guide you through the topic. Each module also has a `studio` subdirectory, with a set of relevant
graphs that we will explore using the LangGraph API and Studio.

## Setup

### Python version

To get the most out of this course, please ensure you're using Python 3.11 or later.
This version is required for optimal compatibility with LangGraph. If you're on an older version,
upgrading will ensure everything runs smoothly.

```
python3 --version
```

### Clone repo

```
git clone https://github.com/langchain-ai/langchain-academy.git
$ cd langchain-academy
```

### Create an environment and install dependencies

#### Mac/Linux/WSL

```
python3 -m venv lc-academy-env
source lc-academy-env/bin/activate
pip install -r requirements.txt
```

#### Windows Powershell

```
PS> python3 -m venv lc-academy-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> lc-academy-env\scripts\activate
PS> pip install -r requirements.txt
```

#### Windows

```
CMD$> python -m venv lc-academy-env
CMD$> source lc-academy-env/Scripts/activate
CMD$> pip install -r requirements.txt
```

### Running notebooks

If you don't have Jupyter set up, follow installation instructions [here](https://jupyter.org/install).

```
jupyter notebook
```

### Setting up env variables

Briefly going over how to set up environment variables. You can also
use a `.env` file with `python-dotenv` library.

```
LANGCHAIN_API_KEY=lsv2********
LANGCHAIN_TRACING_V2=true
GEMINI_API_KEY=AI********
TAVILY_API_KEY=tvly-********
```

### Gemini API key libraries and usage

```
%%capture --no-stderr
%pip install --quiet -U langchain_core
%pip install -U langchain-google-genai
%pip install python-dotenv
%pip install --quiet -U langgraph langgraph_sdk
%pip install -U  tavily-python wikipedia langchain_community
---
import os, getpass
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from os import environ

dotenv.load_dotenv()

os.environ["TAVILY_API_KEY"]=environ.get("TAVILY_API_KEY")

GEMINI_API_KEY = environ.get('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)
```

#### Mac/Linux/WSL

```
export API_ENV_VAR="your-api-key-here"
```

#### Windows Powershell

```
PS> $env:API_ENV_VAR = "your-api-key-here"
```

### Set OpenAI API key

- If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
- Set `OPENAI_API_KEY` in your environment

### Sign up and Set LangSmith API

- Sign up for LangSmith [here](https://smith.langchain.com/), find out more about LangSmith
- and how to use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/)!
- Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` in your environment

### Set up Tavily API for web search

- Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient,
  quick, and persistent search results.
- You can sign up for an API key [here](https://tavily.com/).
  It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily.

- Set `TAVILY_API_KEY` in your environment.

### Set up LangGraph Studio

- Currently, Studio only has macOS support and needs Docker Desktop running.
- Download the latest `.dmg` file [here](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download)
- Install Docker desktop for Mac [here](https://docs.docker.com/engine/install/)

### Running Studio

Graphs for LangGraph Studio are in the `module-x/studio/` folders.

- To use Studio, you will need to create a .env file with the relevant API keys
- Run this from the command line to create these files for module 1 to 4, as an example:

```
$ for i in {1..4}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
$ echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```

## Detailed Analysis of LangGraph Libraries and Their Usage

### Module 0: Setup and Basics

#### Key Imports:

- `from langchain_google_genai import ChatGoogleGenerativeAI`: Integration with Google's Gemini AI models
- `import dotenv`: Loads environment variables from a .env file for API keys and configuration
- `import os`: Interacts with the operating system, used for environment variables

#### Purpose:

The Google Generative AI integration provides an alternative to OpenAI models, allowing you to use Gemini models for your agents. The dotenv setup simplifies managing API keys securely.

### Module 1: Core Concepts

#### Key Imports:

- `from typing import TypedDict, Annotated, Sequence`: Advanced type hinting to define structure of data
- `from langgraph.graph import StateGraph, END`: Core building blocks for creating agent workflows
- `from pydantic import BaseModel, Field`: Schema definition for structured data validation
- `from langchain_core.messages import HumanMessage, AIMessage`: Message types for conversation structure
- `from langchain.tools import BaseTool`: Base class for creating custom tools for agents

#### Annotated Types Explained:

`Annotated[Type, metadata]` is a Python typing feature that adds metadata to a type. In LangGraph, it's often used to define the structure of states and add runtime information. For example:

```python
State = Annotated[
    Dict[str, Any],
    TypeVar("State")
]
```

This indicates that `State` is a dictionary with string keys and any values, plus additional metadata for the type system.

#### StateGraph Usage:

StateGraph is a fundamental concept that defines the workflow of an agent. It creates a directed graph where:

- Nodes represent operations or functions
- Edges represent transitions between states
- The graph maintains state throughout execution

### Module 2: Advanced Interactions

#### Key Imports:

- `from operator import itemgetter`: Extracts specific items from collections/dictionaries
- `from langgraph.checkpoint import MemorySaver`: Saves agent state for persistence
- `from langchain.retrievers import TimeWeightedVectorStoreRetriever`: Retrieves documents with time-based weighting
- `from langchain_core.runnables import RunnablePassthrough, RunnableLambda`: Functional components for processing pipelines

#### Reducers Explained:

Reducers in LangGraph are functions that combine multiple outputs into a single result. They're commonly used in:

1. Map-reduce patterns where multiple parallel tasks generate results that need to be combined
2. Aggregating responses from multiple agents or tools
3. Summarizing information from various sources

Example:

```python
def reduce_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

This reducer takes multiple document objects and combines their content into a single string.

### Module 3: Debugging & User Interaction

#### Key Imports:

- `from langgraph.checkpoint import LocalStateCheckpointSaver`: Saves checkpoints of agent state locally
- `from IPython.display import display`: Creates interactive elements in notebooks
- `from langchain.callbacks.base import BaseCallbackHandler`: Creates custom callback handlers for debugging
- `from langchain_core.messages import SystemMessage`: Defines system instructions for the agent

#### Breakpoints and Debugging:

LangGraph implements breakpoints similar to traditional programming:

```python
# Define a breakpoint condition
def should_break(state):
    return "debug" in state["input"].lower()

# Add breakpoint to graph
builder.add_conditional_edges(
    "agent",
    should_break,
    {True: "human_intervention", False: "output"}
)
```

This creates a dynamic breakpoint that pauses execution when a condition is met, allowing human intervention.

### Module 4: Performance & Architecture

#### Key Imports:

- `from langchain.chains.summarize import load_summarize_chain`: Pre-built chain for text summarization
- `import asyncio`: Python's asynchronous programming library
- `from tavily import TavilyClient`: Integration with Tavily search API
- `from langchain_core.documents import Document`: Represents documents for processing
- `from langchain.text_splitter import RecursiveCharacterTextSplitter`: Chunks text intelligently

#### Asynchronous Processing:

```python
# Async function definition
async def fetch_all(query):
    search_task = asyncio.create_task(search_client.search(query))
    wiki_task = asyncio.create_task(wiki_client.search(query))
    # Run both tasks concurrently
    search_results, wiki_results = await asyncio.gather(search_task, wiki_task)
    return {"search": search_results, "wiki": wiki_results}
```

This pattern allows multiple time-consuming operations (like API calls) to run simultaneously, improving overall performance.

#### Sub-graph Architecture:

LangGraph allows you to compose smaller graphs into larger workflows:

```python
# Create specialized sub-graphs
research_graph = create_research_graph()
summary_graph = create_summary_graph()

# Compose them in the main graph
main_graph.add_node("research", research_graph)
main_graph.add_node("summarize", summary_graph)
main_graph.add_edge("research", "summarize")
```

This modular approach allows complex agents to be built from simpler, reusable components.

### Key Python Features for LangGraph Development:

1. **Type Annotations**: Ensure code correctness and provide better IDE support

   ```python
   def process_message(message: str) -> Dict[str, Any]:
       # Type-safe processing
   ```

2. **Pydantic Models**: Validate data and ensure it matches expected schemas

   ```python
   class AgentState(BaseModel):
       messages: List[BaseMessage] = Field(default_factory=list)
       next_steps: List[str] = Field(default_factory=list)
   ```

3. **Functional Programming**: Process data through transformation pipelines

   ```python
   result = (
       RunnablePassthrough.assign(
           documents=lambda x: retriever.get_relevant_documents(x["query"])
       )
       | format_docs
       | llm
   ).invoke({"query": user_query})
   ```

4. **Async/Await**: Handle concurrent operations efficiently
   ```python
   async def process_batch(items):
       tasks = [process_item(item) for item in items]
       return await asyncio.gather(*tasks)
   ```
