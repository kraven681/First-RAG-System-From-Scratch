# 🧠 First RAG System From Scratch

> A complete, beginner-friendly implementation of **Retrieval-Augmented Generation (RAG)** using FAISS, Sentence Transformers, and Google Flan-T5 — built and debugged step by step in Google Colab.

---

## 📌 What is RAG?

**RAG (Retrieval-Augmented Generation)** solves a fundamental problem with LLMs: *they only know what they were trained on.*

Instead of retraining a model on your custom data, RAG lets you:

1. Store your own documents in a **vector database**
2. When a question comes in, **retrieve** the most relevant chunks
3. **Inject** those chunks into the prompt as context
4. Let the LLM **generate** a grounded answer from that context

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Embed the  │────▶│  Search Vector   │────▶│  Build RAG   │
│   Query     │     │  Database (FAISS)│     │   Prompt     │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                             ┌──────────────┐
                                             │  LLM Answer  │
                                             │  (Flan-T5)   │
                                             └──────────────┘
```

---

## 🗂️ Project Structure

```
First-RAG-System/
│
├── First_RAG_System_From_Scratch.py   # Main notebook (exported .py)
├── my_knowledge.txt                   # Custom knowledge base
└── README.md                          # You are here
```

---

## 🔧 Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Text Splitting | `langchain-text-splitters` | Break documents into chunks |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Convert text to vectors |
| Vector Store | `FAISS` (`IndexFlatL2`) | Store & search vectors |
| Generator | `google/flan-t5-small` (T5ForConditionalGeneration) | Answer questions |
| Runtime | Google Colab | Free GPU/CPU environment |

---

## 🚀 Setup & Installation

### Prerequisites
- Google Colab account (free)
- `my_knowledge.txt` uploaded to Colab files

### Install Dependencies

```python
!pip install -q transformers sentence-transformers faiss-cpu
!pip install -q -U sentence-transformers huggingface_hub
!pip install -q -U langchain-text-splitters
```

> ⚠️ **Important:** After installing, go to **Runtime → Restart session** before running any import cells. This prevents version conflict errors.

---

## 📖 Step-by-Step Walkthrough

### Step 0 — The Knowledge Base

Create a plain text file `my_knowledge.txt` with your custom data. This is the information the LLM has **no idea** about from its training:

```
Company Policy Manual:
- WFH Policy: All employees are eligible for a hybrid WFH schedule.
  Employees must be in the office on Tuesdays, Wednesdays, and Thursdays.
  Mondays and Fridays are optional remote days.
- PTO Policy: Full-time employees receive 20 days of Paid Time Off (PTO)
  per year. PTO accrues monthly.
- Tech Stack: The official backend language is Python, and the official
  frontend framework is React. For mobile development, we use React Native.
```

Upload this file to Colab via the 📁 Files panel on the left sidebar.

---

### Step 1 — Chunking

We cannot feed an entire document to the model at once. We split it into small, overlapping **chunks** (like index cards).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("my_knowledge.txt") as f:
    knowledge_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,    # Max characters per chunk
    chunk_overlap=20,  # Overlap so context isn't lost at boundaries
    length_function=len
)

chunks = text_splitter.split_text(knowledge_text)
```

**Why `RecursiveCharacterTextSplitter`?**

It splits intelligently — first tries `\n\n` (paragraphs), then `\n` (lines), then spaces — so it never cuts a sentence in half. A fixed `\n` split would lose meaning at boundaries.

**Output:**
```
We have 4 chunks:
--- Chunk 1 ---
Company Policy Manual: - WFH Policy: All employees are eligible for a hybrid WFH schedule. Employees must be in the office on Tuesdays, Wednesdays,

--- Chunk 2 ---
Wednesdays, and Thursdays. Mondays and Fridays are optional remote days. - PTO Policy: Full-time employees receive 20 days of Paid Time Off (PTO) per

--- Chunk 3 ---
Time Off (PTO) per year. PTO accrues monthly. - Tech Stack: The official backend language is Python, and the official frontend framework is React.

--- Chunk 4 ---
framework is React. For mobile development, we use React Native.
```

Notice the **20-character overlap** — "Wednesdays" appears in both Chunk 1 and Chunk 2, so no context is lost at the boundary.

---

### Step 2 — Embeddings

We convert each text chunk into a **vector** (a list of numbers) that captures its meaning. Semantically similar sentences end up as mathematically close vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)

print(f"Shape of our embeddings: {chunk_embeddings.shape}")
# Output: (4, 384)
# 4 chunks × 384 dimensions each
```

**How embeddings work conceptually:**

```
"WFH Policy text..."   ──▶  [0.12, -0.45, 0.88, ... 384 numbers]
"PTO Policy text..."   ──▶  [0.09, -0.41, 0.79, ... 384 numbers]
"React frontend..."    ──▶  [0.55,  0.23, 0.11, ... 384 numbers]
```

A query like *"work from home days"* will produce a vector mathematically **close** to the WFH chunk, and far from the React chunk. That's what makes retrieval work.

**Why `all-MiniLM-L6-v2`?**  
It's fast, small (80MB), and runs entirely on CPU — perfect for Colab's free tier. It produces 384-dimensional vectors with solid semantic understanding.

> ⚠️ **Common Error Fix:** If you get `ImportError: cannot import name 'cached_download'`, run:
> ```python
> !pip install -q -U sentence-transformers huggingface_hub
> ```
> Then restart the runtime. This is a version mismatch between `sentence-transformers` and `huggingface_hub`.

---

### Step 3 — Vector Store with FAISS

We store the embeddings in **FAISS** — a library by Meta for fast similarity search. Think of it as a database you search by *meaning*, not by exact keywords.

```python
import faiss
import numpy as np

d = chunk_embeddings.shape[1]            # 384 dimensions

index = faiss.IndexFlatL2(d)             # L2 = Euclidean distance
index.add(np.array(chunk_embeddings).astype('float32'))

print(f"FAISS index created with {index.ntotal} vectors.")
# Output: FAISS index created with 4 vectors.
```

**`IndexFlatL2` explained:**  
This is the simplest FAISS index — it computes the exact Euclidean distance between the query vector and *every* stored vector. It's perfect for small datasets (< 10,000 chunks). For larger datasets, you'd switch to approximate indexes like `IndexHNSWFlat` for speed.

---

### Step 4 — Load the Generator Model

We use Google's **Flan-T5-Small** — an encoder-decoder model fine-tuned on instruction-following tasks.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model  = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

print("Model loaded successfully.")
```

**Why call it directly instead of `pipeline()`?**

Newer versions of `transformers` removed `text2text-generation` from the pipeline task registry. Calling `T5ForConditionalGeneration` directly:
- Works on **all versions** of transformers
- Gives you full control over `max_new_tokens` vs `max_length`
- Avoids the prompt-echoing bug where the pipeline returns the input + output instead of just the answer

---

### Step 5 — The Full RAG Pipeline

This is where everything connects. For each query:

```
Query → Embed → FAISS Search → Retrieve Chunks → Build Prompt → Generate Answer
```

```python
def answer_question(query):

    # ── 1. RETRIEVE ──────────────────────────────────────────────
    query_embedding = model.encode([query]).astype('float32')

    distances, indices = index.search(query_embedding, k=2)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # ── 2. AUGMENT ───────────────────────────────────────────────
    prompt_template = f"""
    Answer the following question using *only* the provided context.
    If the answer is not in the context, say "I don't have that information."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # ── 3. GENERATE ──────────────────────────────────────────────
    inputs = tokenizer(prompt_template, return_tensors="pt",
                       truncation=True, max_length=512)

    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_new_tokens=100)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"--- CONTEXT ---\n{context}\n")
    return answer
```

**Three stages explained:**

| Stage | What happens |
|---|---|
| **Retrieve** | Query is embedded → FAISS finds 2 nearest chunks by vector distance |
| **Augment** | Retrieved chunks are injected into a structured prompt as "Context" |
| **Generate** | Flan-T5 reads the prompt and produces a grounded answer |

---

### Step 6 — Run Queries & Results

```python
query_1 = "What is the WFH policy?"
print(f"Query: {query_1}")
print(f"Answer: {answer_question(query_1)}\n")
```

**Output:**
```
Query: What is the WFH policy?

--- CONTEXT ---
Company Policy Manual: - WFH Policy: All employees are eligible for a hybrid
WFH schedule. Employees must be in the office on Tuesdays, Wednesdays,

Wednesdays, and Thursdays. Mondays and Fridays are optional remote days. ...

Answer: All employees are eligible for a hybrid WFH schedule. Employees must
be in the office on Tuesdays, Wednesdays, and Thursdays. Mondays and Fridays
are optional remote days.
```

```python
query_2 = "What is the company's dental plan?"
print(f"Query: {query_2}")
print(f"Answer: {answer_question(query_2)}\n")
```

**Output:**
```
Query: What is the company's dental plan?

--- CONTEXT ---
Company Policy Manual: - WFH Policy: ...

Answer: I don't have that information.
```

The model correctly answers from context when the answer exists, and gracefully refuses when it doesn't — without hallucinating.

---

## 🐛 Errors Encountered & Fixes

This project was built through real debugging. Here are the errors hit and how they were resolved:

### Error 1 — `ImportError: cannot import name 'cached_download'`

```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**Cause:** `sentence-transformers` was outdated. It called `cached_download` which was removed from `huggingface_hub >= 0.24`.

**Fix:**
```python
!pip install -q -U sentence-transformers huggingface_hub
# Then: Runtime → Restart session
```

---

### Error 2 — `KeyError: Unknown task text2text-generation`

```
KeyError: "Unknown task text2text-generation, available tasks are [...]"
```

**Cause:** Newer `transformers` removed `text2text-generation` from the pipeline task registry.

**Fix:** Stop using `pipeline()` for Flan-T5. Call `T5ForConditionalGeneration` directly:
```python
# ❌ Broken on new transformers:
generator = pipeline('text2text-generation', model='google/flan-t5-small')

# ✅ Works on all versions:
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model  = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
```

---

### Error 3 — Model echoing the prompt instead of answering

**Symptom:** The `Answer:` field returned the entire prompt text, not a real answer.

**Cause:** Using `pipeline('text-generation')` with a causal LM (like TinyLlama) treats Flan-T5 as a decoder-only completion model — it continues the prompt instead of answering it.

**Fix:** Flan-T5 is an **encoder-decoder** model. It must be called with `.generate()`, which reads the full input and produces a new sequence — not a continuation.

---

### Error 4 — `max_length` / `max_new_tokens` conflict warning

```
Both `max_new_tokens` (=256) and `max_length` (=100) seem to have been set.
```

**Cause:** `max_length` counts total tokens (input + output). With a long prompt this left 0 tokens for the answer.

**Fix:** Use `max_new_tokens` on `.generate()`, and `max_length` only on the tokenizer input:
```python
inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = t5_model.generate(**inputs, max_new_tokens=100)
```

---

## 💡 How to Extend This Project

| Upgrade | How |
|---|---|
| Better embedding model | Replace `all-MiniLM-L6-v2` with `BAAI/bge-m3` |
| Larger knowledge base | Load multiple `.txt` or `.pdf` files |
| Persist the index | `faiss.write_index(index, "my_index.faiss")` |
| Better generator | Use `google/flan-t5-large` or the Claude API |
| Hybrid search | Combine FAISS with BM25 keyword search |
| Evaluation | Add [RAGAS](https://github.com/explodinggradients/ragas) to measure accuracy |

---

## 📚 Key Concepts Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A small piece of your document (150 chars here) |
| **Embedding** | A vector (list of numbers) representing the meaning of text |
| **Vector Store** | A database searched by vector similarity, not keywords |
| **FAISS** | Meta's library for fast vector similarity search |
| **L2 Distance** | Euclidean distance between two vectors — smaller = more similar |
| **Encoder-Decoder** | Model architecture where input is fully read before output is generated (Flan-T5) |
| **Hallucination** | When an LLM confidently states something not in its context |
| **Grounded answer** | Answer that comes directly from retrieved context, not model memory |

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — text splitting utilities
- [Sentence Transformers](https://www.sbert.net/) — embedding model
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [Hugging Face](https://huggingface.co/) — Flan-T5 model hosting
- [Google Flan-T5](https://huggingface.co/google/flan-t5-small) — the generator model

---

*Built with 💻 on Google Colab | Debugged with patience 🐛*
