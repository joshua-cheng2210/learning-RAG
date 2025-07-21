# Langchain RAG Tutorial

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

> You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work. -- no longer needed. i used a free model from huggingFace for the word embeddings, and a model from microsoft for the question answering prompting.

project video reference: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).

# things learnt
- comparing gemini and openAI token prices for each model
- what is chroma --> a vector database specifically designed for storing and querying embeddings
- you need to break up all your text data into chunks and then use a word to vector embedding model to convert the text into vectors, and save those vectors in the vector db
- word embeddings models dont have preset vectors for each of the words, they generate them in real time through the trained neural network


# future works to check out
- how does vector db work? and what other uses for it?
- other RAG approaches
- other langchain tools
- more types of vector db searches
    - semantic_results = db.similarity_search(query_text, k=3)
    - keyword_results = db.max_marginal_relevance_search(query_text, k=2)

# how does notebook LLM gets more accurate results?
## Is Google Notebook LLM using RAG? How does it get accurate results?

**Yes, Google Notebook LLM likely uses advanced RAG techniques** combined with much larger, more sophisticated models. Here's the breakdown:

### Google Notebook LLM advantages:

#### 1. Larger, more powerful models
- **Your setup**: `google/flan-t5-base` (~250M parameters)
- **Google's likely setup**: Much larger models (70B+ parameters)
- **Difference**: Larger models = better reasoning and language understanding

#### 2. Advanced RAG techniques
```
# Your current RAG (basic)
query → embedding → search top-k → simple prompt → generate

# Advanced RAG (what Google likely uses)
query → query expansion → multiple search strategies → 
reranking → context optimization → advanced prompting → generate
```

#### 3. Better embedding models
- **Your model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Google's likely models**: Custom/proprietary embeddings (1536+ dimensions)

#### 4. Advanced prompting techniques
```python
# Your current prompt (basic)
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

# Advanced prompting (what Google likely uses)
ADVANCED_PROMPT = """
You are an expert analyst. Based on the following context, provide a comprehensive answer.

Context: {context}

Instructions:
- Be specific and cite relevant details
- If uncertain, acknowledge limitations
- Provide step-by-step reasoning
- Use examples from the context

Question: {question}

Detailed Answer:
"""
```

### Professional RAG techniques Google likely uses:

#### 1. Query expansion and refinement
```python
# Expand queries for better retrieval
expanded_query = f"{query_text} related concepts themes"
refined_query = semantic_query_expansion(query_text)
```

#### 2. Multiple search strategies
```python
# Combine different search approaches
semantic_results = db.similarity_search(query_text, k=3)
keyword_results = db.max_marginal_relevance_search(query_text, k=2)
hybrid_results = combine_and_rerank(semantic_results, keyword_results)
```

#### 3. Advanced context preparation
```python
def prepare_context(results, max_tokens=2000):
    # Remove duplicates, optimize length, add relevance scores
    context_chunks = []
    for doc, score in results:
        if score > 0.8:  # Only high-relevance chunks
            context_chunks.append(f"[Relevance: {score:.2f}] {doc.page_content}")
    return "\n\n".join(context_chunks[:5])  # Best 5 chunks
```

#### 4. Reranking and filtering
- Cross-encoder reranking for better relevance
- Diversity filtering to avoid redundant chunks
- Context compression to fit more relevant information

### Comparison: Your RAG vs Professional RAG

| **Aspect** | **Your Current Setup** | **Professional RAG (Google)** |
|------------|----------------------|------------------------------|
| **Model Size** | FLAN-T5-base (250M) | GPT-4, Claude-3, Custom LLMs (70B+) |
| **Embeddings** | MiniLM (384D) | ada-002, custom (1536D+) |
| **Retrieval** | Simple similarity | Multi-strategy + reranking |
| **Context** | Raw chunks | Optimized, summarized, hierarchical |
| **Prompting** | Basic template | Advanced reasoning prompts |
| **Query Processing** | Direct search | Query expansion + refinement |
| **Post-processing** | None | Reranking + filtering |

### Key factors for Google's accuracy:

1. **Model Quality**: Much larger models with better reasoning capabilities
2. **Advanced Retrieval**: Multiple search strategies combined with reranking
3. **Context Optimization**: Smart chunk selection and context preparation
4. **Prompt Engineering**: Sophisticated prompts that guide better reasoning
5. **Query Enhancement**: Query expansion and refinement for better retrieval
6. **Post-processing**: Reranking and filtering for higher quality results

### Bottom line:
Google's accuracy comes from combining **larger models** + **advanced RAG techniques** + **sophisticated prompting** + **extensive engineering**. Your basic RAG is working correctly - it's just using smaller, simpler components. The core principles are the same, but the execution is much more sophisticated.

### How you could improve (if you had resources):
- Upgrade to larger models (flan-t5-large, flan-t5-xl)
```
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # Much better than base
    max_length=1024,
    max_new_tokens=300,
)
```
- Implement query expansion
- Add reranking mechanisms
- Use better embedding models
- Implement advanced prompting techniques
```
IMPROVED_PROMPT = """
Based on the following context from Alice in Wonderland, provide a detailed and accurate answer.

Context:
{context}

Question: {question}

Please provide a comprehensive answer with specific details from the text. If the answer isn't fully clear from the context, explain what information is available.

Answer:
"""
```
- Add context optimization
