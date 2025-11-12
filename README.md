# LangGraph-Examples
## exp-01-langgraph_persistent_memory_agent.ipynb

# LangGraph Persistent Memory Agent Tutorial

## 📚 Overview

This tutorial will guide you through building a **LangGraph agent with persistent vector-store memory**. Your agent will remember conversations across sessions, even after program restarts, using FAISS for efficient similarity-based retrieval.

### What You'll Build

- An intelligent agent that routes queries to specialized tools
- Persistent memory using vector embeddings
- Integration with search and calculator tools
- Context-aware responses based on conversation history

---

## 🎯 Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher installed
- An OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Basic understanding of Python and LLMs
- Terminal/command line access

---

## 📦 Step 1: Installation

First, install all required dependencies:

```bash
pip install -U langchain langgraph langchain-openai duckduckgo-search faiss-cpu
```

### Package Breakdown

| Package | Purpose |
|---------|---------|
| `langchain` | Core framework for building LLM applications |
| `langgraph` | State machine and workflow orchestration |
| `langchain-openai` | OpenAI integration for LLMs and embeddings |
| `duckduckgo-search` | Web search functionality |
| `faiss-cpu` | Vector similarity search for memory |

---

## 🏗️ Step 2: Project Structure

Create a new directory for your project:

```bash
mkdir langgraph-memory-agent
cd langgraph-memory-agent
```

Your project will generate these files:
- `agent.py` - Main agent code
- `chat_memory.faiss` - Persistent vector store (auto-generated)
- `chat_memory.faiss.pkl` - FAISS index metadata (auto-generated)

---

## 🔧 Step 3: Import Dependencies

Create `agent.py` and start with imports:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langgraph.graph import StateGraph, END
import os
import re
```

### Import Explanation

- **ChatOpenAI**: GPT model for generating responses
- **OpenAIEmbeddings**: Convert text to vector embeddings
- **FAISS**: Facebook AI Similarity Search for vector storage
- **DuckDuckGoSearchRun**: Web search tool
- **StateGraph**: LangGraph's state machine builder
- **tool**: Decorator for creating custom tools

---

## 🧠 Step 4: Initialize Persistent Vector Store

The vector store is the heart of the memory system:

```python
# Vector store file path
VECTOR_DB_PATH = "chat_memory.faiss"

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load existing vector store or create new one
if os.path.exists(VECTOR_DB_PATH):
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("✅ Loaded existing memory from disk")
else:
    vectorstore = FAISS.from_texts(
        ["Conversation initialized."], 
        embeddings
    )
    print("🆕 Created new memory store")
```

### How It Works

1. **First run**: Creates a new FAISS index with an initialization message
2. **Subsequent runs**: Loads existing conversations from disk
3. **Embeddings**: Each conversation turn is converted to a 1536-dimensional vector
4. **Similarity search**: Finds relevant past conversations using cosine similarity

---

## 🛠️ Step 5: Define Agent Tools

Tools extend your agent's capabilities beyond text generation:

```python
# Web search tool
search_tool = DuckDuckGoSearchRun()

# Custom calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2+2' or '10*5'."""
    try:
        # Clean the expression for safety
        safe_expr = re.sub(r'[^0-9+\-*/.() ]', '', expression)
        result = eval(safe_expr)
        return f"The result is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"
```

### Tool Design Principles

- **Clear docstrings**: Help the LLM understand when to use each tool
- **Error handling**: Gracefully handle invalid inputs
- **Security**: Sanitize inputs (especially for `eval()`)
- **Focused functionality**: Each tool does one thing well

---

## 🧩 Step 6: Create Agent Nodes

LangGraph uses nodes to represent different processing steps:

### 6.1 Controller Node (Router)

```python
def controller(state):
    """Route queries to appropriate specialized nodes."""
    query = state["query"].lower()
    
    # Check for mathematical operations
    if any(op in query for op in ["+", "-", "*", "/", "multiply", "divide", "calculate"]):
        return {"next": "calc"}
    
    # Check for information retrieval needs
    elif any(word in query for word in ["who", "what", "when", "where", "news", "latest", "search"]):
        return {"next": "search"}
    
    # Default to reasoning for conversational queries
    else:
        return {"next": "reason"}
```

**Purpose**: Analyzes the query and routes it to the most appropriate node.

---

### 6.2 Reasoning Node (Memory-Enhanced Chat)

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def reasoning(state):
    """Handle conversational queries with memory context."""
    query = state["query"]
    
    # Retrieve similar past conversations
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    memory_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Build prompt with context
    prompt = f"""You are a helpful AI assistant with memory of past conversations.

Previous relevant context:
{memory_context}

Current user question: {query}

Respond clearly and naturally. If the previous context is relevant, refer to it. 
If not, answer the question directly."""
    
    # Generate response
    response = llm.invoke(prompt)
    
    # Save this exchange to memory
    conversation_entry = f"User: {query}\nAssistant: {response.content}"
    vectorstore.add_texts([conversation_entry])
    vectorstore.save_local(VECTOR_DB_PATH)
    
    return {"answer": response.content}
```

**Key Features**:
- Retrieves 3 most similar past conversations
- Provides context to the LLM
- Saves new conversations to memory
- Persists memory to disk after each interaction

---

### 6.3 Calculator Node

```python
def calc(state):
    """Handle mathematical calculations."""
    query = state["query"]
    
    # Extract mathematical expression
    expr = re.sub(r"[^0-9+\-*/(). ]", "", query)
    
    # Calculate result
    result = calculator.invoke(expr)
    
    # Save to memory
    conversation_entry = f"User: {query}\nAssistant: {result}"
    vectorstore.add_texts([conversation_entry])
    vectorstore.save_local(VECTOR_DB_PATH)
    
    return {"answer": result}
```

---

### 6.4 Search Node

```python
def search(state):
    """Handle web search queries."""
    query = state["query"]
    
    # Perform web search
    search_results = search_tool.run(query)
    
    # Summarize results using LLM
    summary_prompt = f"""Summarize the following search results concisely:

{search_results}

Provide a clear, informative summary."""
    
    summary = llm.invoke(summary_prompt)
    
    # Save to memory
    conversation_entry = f"User: {query}\nAssistant: {summary.content}"
    vectorstore.add_texts([conversation_entry])
    vectorstore.save_local(VECTOR_DB_PATH)
    
    return {"answer": summary.content}
```

---

## 🔗 Step 7: Build the LangGraph Workflow

Now we assemble all nodes into a state machine:

```python
# Initialize the graph with dictionary state
graph = StateGraph(dict)

# Add all nodes
graph.add_node("controller", controller)
graph.add_node("reason", reasoning)
graph.add_node("calc", calc)
graph.add_node("search", search)

# Set entry point
graph.set_entry_point("controller")

# Add conditional routing from controller
graph.add_conditional_edges(
    "controller",
    lambda x: x["next"],
    {
        "reason": "reason",
        "calc": "calc",
        "search": "search"
    }
)

# All specialized nodes end the workflow
graph.add_edge("reason", END)
graph.add_edge("calc", END)
graph.add_edge("search", END)

# Compile the graph
app = graph.compile()
```

### Workflow Visualization

```
         START
           ↓
      [controller] ──→ decides route
           ↓
     ┌─────┼─────┐
     ↓     ↓     ↓
  [calc] [reason] [search]
     ↓     ↓     ↓
     └─────┼─────┘
           ↓
          END
```

---

## 💬 Step 8: Create Interactive Chat Interface

```python
def chat():
    """Run an interactive chat session with persistent memory."""
    print("🤖 Persistent AI Agent ready!")
    print("💾 Memory is automatically saved after each message")
    print("Type 'exit' or 'quit' to end the conversation\n")
    print("-" * 50)
    
    while True:
        # Get user input
        query = input("\n😊 You: ").strip()
        
        # Check for exit command
        if query.lower() in ["exit", "quit", "bye"]:
            print("\n🧠 All conversations saved to memory. Goodbye! 👋")
            break
        
        # Skip empty inputs
        if not query:
            continue
        
        # Run the agent
        try:
            result = app.invoke({"query": query})
            print(f"\n🤖 AI: {result['answer']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different query.")

# Run the chat interface
if __name__ == "__main__":
    chat()
```

---

## 🚀 Step 9: Run Your Agent

### First Run

```bash
python agent.py
```

**Example conversation:**

```
🤖 Persistent AI Agent ready!
💾 Memory is automatically saved after each message
Type 'exit' or 'quit' to end the conversation
--------------------------------------------------

😊 You: Who is the ruler of Dubai?

🤖 AI: Sheikh Mohammed bin Rashid Al Maktoum is the ruler of Dubai. 
He has been the Vice President and Prime Minister of the UAE and 
the Emir of Dubai since 2006.

😊 You: What is 25 * 4?

🤖 AI: The result is 100

😊 You: exit

🧠 All conversations saved to memory. Goodbye! 👋
```

---

### Second Run (After Restart)

```bash
python agent.py
```

**The agent remembers previous context:**

```
😊 You: Tell me more about his recent initiatives

🤖 AI: Referring to Sheikh Mohammed bin Rashid Al Maktoum, whom we 
discussed earlier, he has launched several significant initiatives 
including the Dubai AI Strategy, Smart City programs, and various 
innovation-focused projects to make Dubai a global tech hub.
```

**Notice**: The agent remembered "his" refers to Sheikh Mohammed from the previous session!

---

## 🎓 Understanding How Memory Works

### The Memory Cycle

1. **User sends query** → "Tell me about his projects"
2. **Embed query** → Convert to 1536-dimensional vector
3. **Similarity search** → Find 3 most similar past conversations
4. **Context injection** → Include retrieved memories in LLM prompt
5. **Generate response** → LLM uses context to answer
6. **Save new memory** → Store this exchange as new vector
7. **Persist to disk** → Save FAISS index to `chat_memory.faiss`

### Vector Similarity Example

```
Query: "his projects"
↓ (embedding)
[0.23, -0.45, 0.67, ...] (1536 dimensions)
↓ (cosine similarity)
Most similar memories:
1. "User: Who is the ruler of Dubai?\nAssistant: Sheikh Mohammed..." (similarity: 0.82)
2. "User: Tell me about Dubai\nAssistant: Dubai is..." (similarity: 0.71)
3. "Conversation initialized." (similarity: 0.12)
```

---

## 🔍 Testing Different Query Types

### Math Queries
```
You: What is 144 divided by 12?
AI: The result is 12.0
```

### Search Queries
```
You: What are the latest developments in AI?
AI: Recent developments include advances in multimodal AI, 
improved reasoning capabilities, and widespread adoption...
```

### Conversational Queries
```
You: What did we talk about earlier?
AI: We discussed Sheikh Mohammed bin Rashid Al Maktoum, 
the ruler of Dubai, and some mathematical calculations.
```

---

## 🛡️ Best Practices

### 1. Memory Management

```python
# Periodically clean old/irrelevant memories
def clean_old_memories(days_to_keep=30):
    # Implementation for removing outdated entries
    pass
```

### 2. Error Handling

```python
# Wrap vector operations in try-catch
try:
    vectorstore.add_texts([conversation_entry])
    vectorstore.save_local(VECTOR_DB_PATH)
except Exception as e:
    print(f"Warning: Could not save to memory: {e}")
```

### 3. API Key Security

```python
# Use environment variables
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"  # Don't hardcode!

# Or use .env file with python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

---

## 🎯 Advanced Extensions

### Multi-User Memory Isolation

```python
def get_user_vectorstore(user_id):
    """Get or create a separate vector store for each user."""
    path = f"chat_memory_{user_id}.faiss"
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return FAISS.from_texts([f"User {user_id} conversation started."], embeddings)
```

### Memory Search Tool

```python
@tool
def search_memory(query: str) -> str:
    """Search through conversation history."""
    results = vectorstore.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in results])
```

### Conversation Summaries

```python
def summarize_session():
    """Create a summary of the current session."""
    all_docs = vectorstore.similarity_search("", k=100)
    conversations = "\n".join([doc.page_content for doc in all_docs])
    summary = llm.invoke(f"Summarize these conversations:\n{conversations}")
    return summary.content
```

---

## 🐛 Troubleshooting

### Issue: "No such file or directory"
**Solution**: Ensure you're running from the correct directory where `chat_memory.faiss` should be saved.

### Issue: "OpenAI API key not found"
**Solution**: Set your API key:
```bash
export OPENAI_API_KEY='sk-...'
```

### Issue: Memory not persisting
**Solution**: Check that `vectorstore.save_local()` is being called and has write permissions.

### Issue: Poor memory retrieval
**Solution**: 
- Increase `k` in similarity_search (retrieve more memories)
- Use more descriptive conversation entries
- Implement conversation summarization

---

## 📊 Performance Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Embedding model** | `text-embedding-3-small` (good balance) |
| **Memory retrieval** | k=3 to 5 (avoid overwhelming context) |
| **LLM model** | `gpt-4o-mini` (fast + affordable) |
| **Vector store** | FAISS (excellent for < 1M vectors) |
| **Disk usage** | ~1KB per conversation turn |

---

## 🎉 Conclusion

You've built a sophisticated LangGraph agent with:

✅ Persistent vector-store memory  
✅ Multi-tool integration (search, calculator)  
✅ Intelligent query routing  
✅ Context-aware responses  
✅ Cross-session memory retention  

### Next Steps

1. Add more specialized tools (email, calendar, etc.)
2. Implement conversation summarization
3. Add user authentication for multi-user scenarios
4. Deploy as a web API with FastAPI
5. Integrate with other vector stores (Pinecone, Weaviate)

---

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

**Happy building! 🚀**
