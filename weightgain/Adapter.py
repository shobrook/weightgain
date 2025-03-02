# Standard library
import random
from typing import Optional

# Third party
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local
try:
    from weightgain.Dataset import Dataset
    from weightgain.utilities import cosine_similarity, accuracy_and_stderr
except ImportError:
    from .Dataset import Dataset
    from utilities import cosine_similarity, accuracy_and_stderr


#########
# HELPERS
#########


def tensors_from_df(df: pd.DataFrame) -> tuple[torch.tensor]:
    embedding1 = np.stack(np.array(df["text_1_embedding"].values))
    embedding2 = np.stack(np.array(df["text_2_embedding"].values))
    similarity = np.stack(np.array(df["label"].astype("float").values))

    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    similarity = torch.from_numpy(similarity).float()

    return embedding1, embedding2, similarity


def model(
    embedding1: torch.tensor,
    embedding2: torch.tensor,
    matrix: torch.tensor,
    dropout: float,
) -> float:
    embedding1 = torch.nn.functional.dropout(embedding1, p=dropout)
    embedding2 = torch.nn.functional.dropout(embedding2, p=dropout)
    embedding1 = embedding1 @ matrix
    embedding2 = embedding2 @ matrix
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity


def mse_loss(predictions: torch.tensor, targets: torch.tensor) -> torch.tensor:
    difference = predictions - targets
    return torch.sum(difference * difference) / difference.numel()


def biased_embedding(embedding: list[float], matrix: torch.tensor) -> np.array:
    embedding = torch.tensor(embedding).float()
    embedding = embedding @ matrix
    embedding = embedding.detach().numpy()
    return embedding


def apply_matrix_to_df(matrix: torch.tensor, df: pd.DataFrame):
    for column in ["text_1_embedding", "text_2_embedding"]:
        df[f"{column}_custom"] = df[column].apply(lambda x: biased_embedding(x, matrix))

    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["text_1_embedding_custom"], row["text_2_embedding_custom"]
        ),
        axis=1,
    )


######
# MAIN
######


class Adapter(object):
    def __init__(self, matrix: np.ndarray, results: Optional[pd.DataFrame] = None):
        self.matrix = matrix
        self.results = results

    def __matmul__(self, embedding) -> np.ndarray:
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        return embedding @ self.matrix

    def to_csv(self, path: str):
        if self.results:
            raise Exception("Cannot save Adapter without training results")

        self.results.to_csv(path, index=False)

    def plot_sim_dists(self):
        fig1 = px.histogram(
            self.results,
            x="cosine_similarity",
            color="label",
            barmode="overlay",
            width=500,
            facet_row="dataset",
        )
        fig2 = px.histogram(
            self.results,
            x="cosine_similarity_custom",
            color="label",
            barmode="overlay",
            width=500,
            facet_row="dataset",
        )
        fig1.show()
        fig2.show()

    def plot_loss(self):
        px.line(
            self.results,
            # line_group="dataset",
            x="epoch",
            y="loss",
            color="type",
            hover_data=["batch_size", "learning_rate", "dropout"],
            facet_row="learning_rate",
            facet_col="batch_size",
            width=500,
        ).show()

    def plot_accuracy(self):
        px.line(
            self.results,
            # line_group="dataset",
            x="epoch",
            y="accuracy",
            color="type",
            hover_data=["batch_size", "learning_rate", "dropout"],
            facet_row="learning_rate",
            facet_col="batch_size",
            width=500,
        ).show()

    def show_report(self):
        if self.results is None:
            raise Exception("Cannot generate report without training results")

        # Create a 2x2 subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Training and Test Loss",
                "Training and Test Accuracy",
                "Original Cosine Similarity Distribution",
                "Adapted Cosine Similarity Distribution",
            ),
        )

        # Add loss plot
        for dataset_type in self.results["type"].unique():
            data = self.results[self.results["type"] == dataset_type]
            fig.add_trace(
                go.Scatter(
                    x=data["epoch"],
                    y=data["loss"],
                    mode="lines",
                    name=f"{dataset_type} loss",
                ),
                row=1,
                col=1,
            )

        # Add accuracy plot
        for dataset_type in self.results["type"].unique():
            data = self.results[self.results["type"] == dataset_type]
            fig.add_trace(
                go.Scatter(
                    x=data["epoch"],
                    y=data["accuracy"],
                    mode="lines",
                    name=f"{dataset_type} accuracy",
                ),
                row=1,
                col=2,
            )

        # Check if we have the dataset attribute for similarity distributions
        if hasattr(self, "dataset"):
            # Add original similarity distribution
            for dataset_type in ["train", "test"]:
                subset = self.dataset[self.dataset["dataset"] == dataset_type]

                # Add original similarity distribution
                for label_value in [-1, 1]:
                    label_subset = subset[subset["label"] == label_value]
                    if not label_subset.empty:
                        fig.add_trace(
                            go.Histogram(
                                x=label_subset["cosine_similarity"],
                                name=f"Original {dataset_type} (label={label_value})",
                                opacity=0.7,
                                nbinsx=30,
                            ),
                            row=2,
                            col=1,
                        )

                # Add custom similarity distribution
                for label_value in [-1, 1]:
                    label_subset = subset[subset["label"] == label_value]
                    if not label_subset.empty:
                        fig.add_trace(
                            go.Histogram(
                                x=label_subset["cosine_similarity_custom"],
                                name=f"Custom {dataset_type} (label={label_value})",
                                opacity=0.7,
                                nbinsx=30,
                            ),
                            row=2,
                            col=2,
                        )
        else:
            # If we don't have the dataset, add empty plots with a message
            fig.add_annotation(
                text="Similarity data not available",
                xref="x3",
                yref="y3",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=2,
                col=1,
            )

            fig.add_annotation(
                text="Similarity data not available",
                xref="x4",
                yref="y4",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=800, width=1000, title_text="Training Report", barmode="overlay"
        )

        fig.show()

    @classmethod
    def from_file(cls, path: str) -> "Adapter":
        matrix = np.load(path)
        return cls(matrix)

    @classmethod
    def train(
        cls,
        dataset: Dataset,
        batch_size: int = 100,
        max_epochs: int = 100,
        learning_rate: float = 100.0,
        dropout: float = 0.0,
    ) -> "Adapter":
        embedding_len = len(dataset["text_1_embedding"].values[0])

        emb1_train, emb2_train, sim_train = tensors_from_df(
            dataset[dataset["dataset"] == "train"]
        )
        emb1_test, emb2_test, sim_test = tensors_from_df(
            dataset[dataset["dataset"] == "test"]
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(emb1_train, emb2_train, sim_train),
            batch_size=batch_size,
            shuffle=True,
        )
        matrix = torch.randn(embedding_len, embedding_len, requires_grad=True)

        best_acc, best_matrix = 0, matrix
        epochs, types, losses, accuracies = [], [], [], []
        for epoch in range(1, 1 + max_epochs):
            for emb1, emb2, target_sim in train_loader:
                # Generate prediction and calculate loss
                pred_sim = model(emb1, emb2, matrix, dropout)
                loss = mse_loss(pred_sim, target_sim)
                loss.backward()  # Backpropagate loss

                # Update matrix
                with torch.no_grad():
                    matrix -= matrix.grad * learning_rate
                    matrix.grad.zero_()

            # Calculate test loss
            test_preds = model(emb1_test, emb2_test, matrix, dropout)
            test_loss = mse_loss(test_preds, sim_test)

            # Compute new embeddings + similarities
            apply_matrix_to_df(matrix, dataset)

            # Calculate test accuracy
            for dataset_type in ["train", "test"]:
                data = dataset[dataset["dataset"] == dataset_type]
                accuracy, stderr = accuracy_and_stderr(
                    data["cosine_similarity_custom"], data["label"]
                )

                epochs.append(epoch)
                types.append(dataset_type)
                losses.append(
                    loss.item() if dataset_type == "train" else test_loss.item()
                )
                accuracies.append(accuracy)

                if accuracy > best_acc and dataset_type == "test":
                    best_acc = accuracy
                    best_matrix = matrix

                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset_type} accuracy: {accuracy:0.1%} ± {1.96 * stderr:0.1%}"
                )

        results = pd.DataFrame(
            {
                "epoch": epochs,
                "type": types,
                "loss": losses,
                "accuracy": accuracies,
            }
        )
        results["batch_size"] = batch_size
        results["max_epochs"] = max_epochs
        results["learning_rate"] = learning_rate
        results["dropout"] = dropout

        adapter = cls(best_matrix.detach().numpy(), results)
        adapter.dataset = dataset
        return adapter
