# Third party
from litellm import embedding

# Local
try:
    from weightgain.Adapter import Adapter
except ImportError:
    from Adapter import Adapter


class Model(object):
    def __init__(self, model: str, batch_size: int = 2048, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.adapter = None

    def set_adapter(self, adapter: Adapter):
        self.adapter = adapter

    def get_embeddings(self, inputs: list[str]) -> list[list[float]]:
        response = embedding(self.model, inputs, **self.kwargs)

        embeddings = []
        for result in sorted(response["data"], key=lambda d: d["index"]):
            vector = result["embedding"]
            if self.adapter:
                vector = vector @ self.adapter

            embeddings.append(vector)

        return embeddings
