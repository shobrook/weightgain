# Third party
import numpy as np
import pandas as pd


######
# MAIN
######


class Adapter(object):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def __matmul__(self, embedding: np.ndarray) -> np.ndarray:
        return embedding @ self.matrix

    @classmethod
    def from_file(cls, path: str) -> "Adapter":
        matrix = np.load(path)
        return cls(matrix)

    @classmethod
    def train(
        cls,
        dataset: pd.DataFrame,
        batch_size: int = 100,
        max_epochs: int = 100,
        lr: float = 100.0,
        dropout: float = 0.0,
        verbose: bool = True,
    ) -> "Adapter":
        pass
