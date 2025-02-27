# weightgain

**Train an adapter for any embedding model in under a minute.**

TODO: Write README

- Problem: You can't fine-tune embedding models behind an API
- Solution: Train an adapter –– matrix of weights you multiply an embedding by to make certain features more or less salient
- `weightgain` lets you train an adapter in under a minute, even if you don't have a dataset. Takes a few lines of code and works with any black-box embedding model.

**Features:**

- Built on top of LiteLLM (therefore supports any model, including OpenAI, Voyage, Cohere, and local ollama models)
- Uses on-disk caching, so you can run the training script multiple times without recomputing embeddings

## Installation

```bash
> pip install weightgain
```

## Usage

1. Provide a dataset of (text_1, text_2) pairs, where text_1 and text_2 are positive examples (e.g. question and answer, logically connected statements, similar sentences, etc.). If you don't have one, you can generate a synthetic dataset using an LLM of your choice.
2. Provide an embedding model.
3. Train the adapter with one line of code.

```python
model = EmbeddingModel.from_litellm("openai/text-ada-003")
dataset = Dataset.from_llm(model, llm="openai/gpt-4o")
adapter = Adapter.train(dataset)
model.set_adapter(adapter)
embedding = model.run("Please embed this sentence")
```

You can also plot things using these objects. E.g. cosine similarities in the dataset before and after the adapter, training loss, etc.

## Citation

If this repository is useful for academic work, please remember to cite it:

```
CITATION THINGIE GOES HERE
```
