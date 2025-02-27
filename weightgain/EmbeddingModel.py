class EmbeddingModel(object):
    def __init__(self, batch_size: int = 1):
        self.adapter = None
        self.batch_size = batch_size

    @classmethod
    def from_name(cls, name: str) -> "EmbeddingModel":
        pass  # TODO: Use LiteLLM?

    def set_adapter(self, adapter: Adapter):
        self.adapter = adapter

    def get_embeddings(self, inputs: list[str]) -> list[list[float]]:
        pass  # TODO
