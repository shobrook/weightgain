# Third party
from litellm import embedding


class Model(object):
    def __init__(self, model: str, batch_size: int = 100, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.adapter = None

    def set_adapter(self, adapter):
        self.adapter = adapter

    def get_embeddings(self, inputs: list[str]) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(inputs), self.batch_size):
            response = embedding(
                self.model, inputs[i : i + self.batch_size], **self.kwargs
            )

            for result in sorted(response["data"], key=lambda d: d["index"]):
                vector = result["embedding"]
                if self.adapter:
                    vector = vector @ self.adapter

                embeddings.append(vector)

        return embeddings

    def get_embedding(self, input: str) -> list[float]:
        return self.get_embeddings([input])[0]
