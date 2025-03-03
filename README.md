# weightgain

**Fine-tune _any_ embedding model in under a minute. Even closed-source models (OpenAI, Cohere, Voyage, etc.).**

It works by training an adapter that sits _on top_ of the model, instead of modifying the model itself. This adapter transforms your embeddings after they're generated to boost retrieval accuracy and overall RAG performance.

Weightgain lets you train an adapter in just a couple lines of code, even if you don't have a dataset.

## Installation

```bash
> pip install weightgain
```

## Quickstart

```python
from weightgain import Dataset, Adapter, Model

# Choose an embedding model
model = Model("openai/text-embedding-3-large")

# Generate synthetic data (or supply your own)
dataset = Dataset.from_synthetic_chunks(
    prompt="Chunks of code from an arbitrary Python codebase.",
    llm="openai/gpt-4o-mini",
    n_chunks=25,
    n_queries_per_chunk=5
)

# Train the adapter
adapter = Adapter.train(dataset, model)

# Generate a new embedding
old_embedding = model.get_embedding("Embed this sentence")
new_embedding = adapter @ old_embedding # matrix multiplication
```

## Usage

### Choosing an Embedding Model

Weightgain wraps LiteLLM to get access to model APIs. You can see the full list of supported embedding models [here.](https://docs.litellm.ai/docs/embedding/supported_embedding)

<!--TODO: You can also define your own-->

### Building the Dataset

You need a dataset of `[query, chunk]` pairs to get started. A chunk is a retrieval result, e.g. a code snippet or excerpt from a document. You can either generate a synthetic dataset or supply your own.

**If you already have chunks:**

```python
from weightgain import Dataset

chunks = [...] # list of strings
model = Model("openai/text-embedding-3-large")
dataset = Dataset.from_chunks(
    chunks,
    model,
    llm="openai/gpt-4o-mini",
    n_queries_per_chunk=1
)
```

This will use `gpt-4o-mini` (or whatever LiteLLM model you want) to generate `1` query per chunk.

**If you don't have chunks:**

```python
dataset = Dataset.from_synthetic_chunks(
    prompt="Chunks of code from an arbitrary Python codebase.",
    llm="openai/gpt-4o-mini",
    n_chunks=25,
    n_queries_per_chunk=1
)
```

This will generate chunks using the prompt, and then generate `1` query per chunk.

**If you have queries and chunks:**

```python
qa_pairs = [...] # list of (str, str) tuples
dataset = Dataset.from_pairs(qa_pairs, model)
```

### Training the Adapter

```python
from weightgain import Adapter

adapter = Adapter.train(
    dataset,
    model,
    batch_size=25,
    max_epochs=50,
    learning_rate=100.0,
    dropout=0.0
)
```

After training, you can generate a report with various plots (training loss, cosine similarity distributions before/after training, etc.):

```python
adapter.show_report()
```

![Example report](./report.png)

### Using the Adapter

Behind the scenes, the adapter is just a `numpy` matrix that you can multiply your embeddings with:

```python
old_embedding = model.get_embedding("Embed this sentence")
new_embedding = adapter @ old_embedding
```

You can access this matrix directly too:

```python
adapter.matrix # returns numpy.ndarray
```

## Roadmap

1. Add option to train an MLP instead of a linear layer
2. Add a method for easy hyperparameter search
