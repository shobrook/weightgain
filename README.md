# weightgain

**Fine-tune _any_ embedding model in under a minute. Even closed-source models from OpenAI, Cohere, Voyage, etc.**

It works by training an [adapter](https://research.trychroma.com/embedding-adapters) that sits on top of the model, transforming the embeddings _after_ they're generated. This produces task-specific embeddings optimized for your specific RAG/retrieval use case. 

With weightgain, you can train an adapter in just a couple lines of code –– even if you don't have a dataset.

## Installation

```bash
> pip install weightgain
```

## Quickstart

```python
from weightgain import Dataset, Adapter

# Generate a dataset (or supply your own)
dataset = Dataset.from_synthetic_chunks(
    prompt="Chunks of code from an arbitrary Python codebase.",
    llm="openai/gpt-4o-mini",
)

# Train the adapter
adapter = Adapter("openai/text-embedding-3-large")
adapter.fit(dataset)

# Apply the adapter
new_embeddings = adapter.transform(old_embeddings)
```

## Usage

### Choosing an Embedding Model

Weightgain wraps LiteLLM. You can fine-tune any embedding model supported by LiteLLM, e.g. models from OpenAI, Cohere, Voyage, etc. [Here's](https://docs.litellm.ai/docs/embedding/supported_embedding) the full list of supported models.

<!--TODO: You can also define your own-->

### Building the Dataset

You need a dataset of `[query, chunk]` pairs to get started. A chunk is a retrieval result, e.g. a code snippet or excerpt from a document. And the query is a string that's _similar_ to the chunk and should match in a vector search. You can either generate a synthetic dataset or supply your own.

**If you already have chunks:**

```python
from weightgain import Dataset

chunks = [...] # list of strings
dataset = Dataset.from_chunks(
    chunks,
    llm="openai/gpt-4o-mini",
    n_queries_per_chunk=1
)
```

This will use OpenAI's `gpt-4o-mini` (or whatever LiteLLM model you want) to generate `1` query per chunk.

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

adapter = Adapter.fit(
    dataset,
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

### Applying the Adapter

```python
old_embeddings = [...] # list of vectors
new_embeddings = adapter.transform(old_embeddings)
```

Behind the scenes, an adapter is just a matrix of weights that you can multiply your embeddings with. You can access this matrix like so:

```python
adapter.matrix # returns numpy.ndarray
```

## Roadmap

1. Add option to train an MLP instead of a linear layer
2. Add a method for easy hyperparameter search
