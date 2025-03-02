# weightgain

**Train an adapter for _any_ embedding model in under a minute.**

The best embedding models are locked behind an API (OpenAI, Cohere, Voyage, etc.) and can't be fine-tuned. To get around this, you can train an adapter. This is a matrix of weights you can multiply your embeddings by to optimize them for retrieval over your specific data.

`weightgain` lets you train an adapter in a couple lines of code, even if you don't have a dataset.

## Installation

```bash
> pip install weightgain
```

## Usage

```python
from weightgain import Dataset, Adapter, Model

# Choose an embedding model
model = Model("openai/text-embedding-3-large")

# Generate synthetic data (or supply your own)
dataset = Dataset.from_synthetic_chunks(
    prompt="Chunks of code from an arbitrary Python codebase.",
    model=model,
    llm="openai/gpt-4o-mini",
    n_chunks=25,
    n_queries_per_chunk=5
)

# Train the adapter
adapter = Adapter.train(dataset)

# Apply adapter to the model
model.set_adapter(adapter)

# Generate a new embedding
embeddings = model.get_embeddings(["Embed this sentence"])
```

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
    model=model,
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

<!--TODO: Show example plots-->

### Using the Adapter

```python
model.set_adapter(adapter)
embeddings = model.get_embeddings(["Embed this sentence"])
```

You can also use the `numpy` matrix directly:

```python
embedding = model.get_embedding("Embed this sentence")
new_embedding = embedding @ adapter.matrix # @ is matrix multiplication
```

<!--TODO: Flesh this out more.-->
<!--TODO: How you use the adapter is kinda jank. Maybe should be a function that wraps any LiteLLM embedding call-->

## Roadmap

1. Add option to train an MLP instead of a linear layer
2. Add a method for easy hyperparameter search
3. Move the embedding step to `Adapter.train` instead of dataset creation
