# 🦜🔗 LangChain & RAG Comprehensive Toolkit with Ask WebPage Project

A deep dive into Large Language Model (LLM) orchestration, Retrieval-Augmented Generation (RAG), and custom AI application development using LangChain. 

This repository serves as a comprehensive toolkit and learning workspace demonstrating advanced AI engineering concepts. It includes integrations with multiple AI providers (OpenAI, Google Gemini, Anthropic, Hugging Face), structured output parsing, dynamic prompt engineering, vector databases, and full-stack AI applications.

---

## 📁 Repository Structure

```text
├── ask-webpage-backend/     # FastAPI server for processing web-page queries
├── ask-webpage-extension/   # Chrome extension files (Manifest, Popup UI/JS)
├── Chains/                  
│   ├── Runnable/            # LCEL components (parallel, branch, lambda, passthrough)
│   └── *_chain.py           # Pre-built chain orchestrations (conditional, sequential)
├── Models/                  
│   ├── Embeddings/          # OpenAI and Hugging Face embedding scripts
│   ├── Language/            # Chat models (OpenAI, Anthropic, Gemini, Hugging Face)
│   └── Project-1 Document Similarity.py
├── Output (of LLM)/         
│   ├── Output_Parser/       # JSON, Pydantic, and Structured Output Parsers
│   └── Structured_Output/   # .with_structured_output() implementations
├── Prompts/                 
│   ├── ListOfMessage/       # Context-aware chat histories
│   ├── SingleMessage/       # Streamlit Q&A and Research Summarizer
│   └── UI_Chatbot.py        # Streamlit Chatbot UI
├── RAG Components/          
│   ├── Document Loaders/    # PDF, Text, Directory, and Web loaders
│   ├── Retrievers/          # External API integration (Wikipedia)
│   ├── Text Splitters/      # Various chunking algorithms
│   └── Vector Stores/       # ChromaDB implementation
├── requirements.txt         # Project dependencies
└── README.md

````
---

## 🚀 Key Features & Implementations

### 1. Advanced Chain Orchestration (LCEL)
Located in `Chains/` and `Chains/Runnable/`, demonstrating complex workflows using LangChain Expression Language:
* **Parallel Processing (`parallel.py`, `parallel_chain.py`)**: Running multiple LLM chains simultaneously. For example, taking a source document and simultaneously generating study notes and a quiz, then merging them into a single final output.
* **Conditional Branching (`runnableBranch.py`, `conditional_chain.py`)**: Routing logic based on dynamic input. Includes a feedback analyzer that classifies user feedback sentiment (positive/negative) and dynamically routes it to the appropriate response generation chain.
* **Lambda & Passthrough (`runnableLambda.py`, `runnablePassthrough.py`)**: Integrating custom Python functions (like text length counters) directly into LCEL chains, and passing unmodified data through complex pipelines.
* **Sequential Chains (`sequential_chain.py`, `sequence.py`)**: Multi-step pipelines where the output of one prompt (e.g., "Write a detailed report") seamlessly feeds into another (e.g., "Summarize the report").

### 2. Multi-Provider Model & Embedding Integration
Located in `Models/`, showing seamless switching between various AI providers:
* **Language Models**: Implementations for `ChatOpenAI` (GPT-4), `ChatAnthropic` (Claude-2), `ChatGoogleGenerativeAI` (Gemini-1.5-Flash), and Open Source models via `HuggingFaceEndpoint` (e.g., Mistral, TinyLlama, Gemma).
* **Embeddings (`1_huggingFace_embeddings.py`, `2_openai_embedding.py`)**: Text vectorization using OpenAI (`text-embedding-3-large`) and Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`).
* **Document Similarity Project**: A standalone script calculating cosine similarity between query embeddings and document embeddings using `scikit-learn` and Hugging Face.

### 3. Structured Data & Output Parsing
Located in `Output (of LLM)/`, focusing on enforcing strict schemas for LLM responses:
* **Pydantic Validation (`pydanticOutputParser.py`)**: Using Pydantic `BaseModel` and `Field` validations to enforce strict schemas (e.g., ensuring a fictional person's profile always returns a string `name`, integer `age`, and string `city`).
* **Structured Output (`1_type_dict_structured_output.py` to `5_json_struture_op.py`)**: Utilizing the `.with_structured_output()` method with `TypedDict`, `Annotated`, `Pydantic`, and JSON Schemas to reliably extract structured review data (key points, sentiment, pros, cons) from raw text.
* **Format Parsers**: Implementations of `JsonOutputParser` and `StructuredOutputParser`.

### 4. RAG (Retrieval-Augmented Generation) Components
Located in `RAG Components/`, providing a modular architecture for document QA:
* **Document Loaders**: Scripts for ingesting standard text (`TextLoader.py`), directories (`DirectoryLoader.py`), PDFs (`PyPDFLoader.py`), and web content (`WebBasedLoader.py`). Includes lazy loading examples.
* **Text Splitters**: Chunking strategies including Length-based, Semantic Meaning-based, and Document/Text Structure-based splitting to optimize context windows.
* **Vector Stores**: Implementation of vector search and storage using ChromaDB (`langchain_chroma.py`).
* **Retrievers**: External knowledge retrieval utilizing the Wikipedia API (`wikipedia_retriever.py`).

### 5. Interactive Chatbots & UIs (Streamlit)
Located in `Prompts/`, moving beyond CLI to interactive web apps:
* **Conversational Chatbots (`UI_Chatbot.py`)**: A Streamlit-based chat interface utilizing `SystemMessage`, `HumanMessage`, and `AIMessage` to maintain conversational context and history.
* **Research Paper Summarizer (`Research_paper_summarizer_Dynamic_Prompt.py`)**: A dynamic Streamlit tool where users can select academic papers, explanation styles (e.g., Beginner-Friendly, Mathematical, Code-Oriented), and lengths, which dynamically updates the LLM prompt.
* **Static Q&A (`QnA_Static_Prompt.py`)**: A straightforward interface for asking questions using Hugging Face's Mistral model.

### 6. 🌐 Highlight Project: "Ask Webpage" Full-Stack Application
Located in `ask-webpage-backend/` and `ask-webpage-extension/`, this is a practical, end-to-end AI product that allows you to chat with any active webpage you are currently viewing.

**How It Works:**
1. **Capture:** The user clicks the Chrome extension and types a question. The extension automatically grabs the active tab's URL.
2. **Process:** The URL and query are sent to the local API. LangChain dynamically scrapes the webpage, chunks the text, and feeds it into the LLM as context.
3. **Answer:** The LLM streams a context-aware answer back to the extension UI.

**The Tech Stack & Development Approach:**
* **FastAPI Backend (Manually Engineered)**: Built from scratch utilizing Python, **FastAPI**, and **LangChain** (`WebBaseLoader`, Prompt Templates, and LLM chains). This handles the heavy lifting, routing, and prompt execution.
* **Chrome Extension (Vibecoded)**: The frontend (`popup.html`, `popup.js`, `manifest.json`) was **vibecoded** (rapidly prototyped using AI generation). This approach allowed for a clean, functional UI to capture DOM elements and handle API requests without getting bogged down in frontend boilerplate.




### 🛠️ Installation & Setup

### Clone the repository
```bash
git clone https://github.com/ShauryaPundirGraphicEra/Langchain-Learning
cd Langchain-Learning
````

### Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install the required dependencies

```bash
pip install -r requirements.txt
```

### Set up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
GOOGLE_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

---

## 🖥️ Usage Examples

### Running the Streamlit Chatbot UI

Experience the context-aware chatbot powered by Hugging Face's Mistral model:

```bash
streamlit run Prompts/UI_Chatbot.py
```

### Running the Research Paper Summarizer

Launch the dynamic prompt UI for summarizing academic papers:

```bash
streamlit run Prompts/SingleMessage/Research_paper_summarizer_Dynamic_Prompt.py
```

### Running the Ask-Webpage Backend

Start the FastAPI server required for the Chrome extension:

```bash
cd ask-webpage-backend
uvicorn backend:app --reload
```

> **Note:** Ensure the Chrome extension is loaded via `chrome://extensions/` → "Load unpacked" → select `ask-webpage-extension`.

### Executing a LangChain Script

Run any individual script directly to see console output:

```bash
python Chains/conditional_chain.py
```


