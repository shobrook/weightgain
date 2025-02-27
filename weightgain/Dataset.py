from utils.openai import call_gpt

# Third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local
try:
    from weightgain.utilities import cosine_similarity
except ImportError:
    from utilities import cosine_similarity


#########
# HELPERS
#########


TEST_FRACTION = 0.8
RANDOM_SEED = 123
NEGATIVES_PER_POSITIVE = 1

DEFAULT_QA_GENERATE_PROMPT_TMPL = """You are a teacher creating quiz questions on the following CONTEXT\
# START CONTEXT
{context_str}
# END CONTEXT

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""

def generate_dataset(text) -> list[tuple[str, str, int]]:
    return []  # TODO: Return (text_1, text_2, +1 or -1) tuples


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataset,
        test_size=TEST_FRACTION,
        stratify=dataset["label"],
        random_state=RANDOM_SEED,
    )
    train_df.loc[:, "dataset"] = "train"
    test_df.loc[:, "dataset"] = "test"

    return train_df, test_df


def add_negatives(dataset: pd.DataFrame) -> pd.DataFrame:
    texts = set(dataset["text_1"].values) | set(dataset["text_2"].values)
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair) for text_pair in dataset[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    negatives_dataset = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    negatives_dataset["label"] = -1
    negatives_dataset["dataset"] = dataset["dataset"]

    dataset = pd.concat(
        [
            dataset,
            negatives_dataset.sample(
                n=len(dataset) * NEGATIVES_PER_POSITIVE, random_state=RANDOM_SEED
            ),
        ]
    )
    return dataset


def add_embeddings(dataset: pd.DataFrame, model: EmbeddingModel):
    for column in ["text_1", "text_2"]:
        dataset[f"{column}_embedding"] = model.run(dataset[column])

    dataset["cosine_similarity"] = dataset.apply(
        lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
        axis=1,
    )


######
# MAIN
######


class Dataset(object):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def from_llm(cls, model: EmbeddingModel, llm: str) -> "Dataset":
        dataset = generate_dataset()
        df = pd.DataFrame(dataset, columns=["text_1", "text_2", "label"])
        train_df, test_df = split_dataset(df)
        train_df = add_negatives(train_df)
        test_df = add_negatives(test_df)
        df = pd.concat([train_df, test_df])
        add_embeddings(df, model)
        return cls(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, model: EmbeddingModel) -> "Dataset":
        train_df, test_df = split_dataset(df)
        train_df = add_negatives(train_df)
        test_df = add_negatives(test_df)
        df = pd.concat([train_df, test_df])
        add_embeddings(df, model)
        return cls(df)

    def plot_similarities(self):
        pass  # TODO
