# Third party
import json_repair
import pandas as pd
from litellm import completion
from sklearn.model_selection import train_test_split

# Local
try:
    from weightgain.utilities import cosine_similarity
    from weightgain.Model import Model
except ImportError:
    from utilities import cosine_similarity
    from Model import Model


#########
# HELPERS
#########


TEST_FRACTION = 0.8
RANDOM_SEED = 123
NEGATIVES_PER_POSITIVE = 1

PROMPT = """Your task is to generate synthetic data to fine-tune an embedding model. \
The data should consist of pairs of text strings (text_1, text_2). \

You should generate a dataset of {n} pairs of text strings. Pairs should be unique and varied.

The first string, text_1, should be generated from the following prompt:

<prompt_1>
{prompt_1}
</prompt_1>

The second string, text_2, should be generated from the following prompt:

<prompt_2>
{prompt_2}
</prompt_2>

You MUST generate {n} pairs of text strings."""


def generate_dataset(
    prompt_1: str, prompt_2: str, model: str, n: int = 100
) -> list[tuple[str, str, int]]:
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(prompt_1=prompt_1, prompt_2=prompt_2, n=n),
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_data_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text_1": {
                                        "type": "string",
                                        "description": prompt_1,
                                    },
                                    "text_2": {
                                        "type": "string",
                                        "description": prompt_2,
                                    },
                                },
                                "required": ["text_1", "text_2"],
                                "additionalProperties": False,
                            },
                            "description": f"List of {n} pairs of text strings.",
                        }
                    },
                    "required": ["data"],
                    "additionalProperties": False,
                },
            },
        },
    )
    response = response.choices[0].message.content
    response = json_repair.loads(response)

    dataset = []
    for pair in response["data"]:
        dataset.append((pair["text_1"], pair["text_2"], 1))

    # TODO: Generate more if len(dataset) < n.

    print(dataset)

    df = pd.DataFrame(dataset, columns=["text_1", "text_2", "label"])
    return df


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


def add_embeddings(dataset: pd.DataFrame, model: Model):
    for column in ["text_1", "text_2"]:
        dataset[f"{column}_embedding"] = model.get_embeddings(dataset[column])

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

    def __getitem__(self, key):
        return self.df[key]

    def __setitem__(self, key, value):
        self.df[key] = value

    def apply(self, func, axis=0, **kwargs):
        return self.df.apply(func, axis=axis, **kwargs)

    @classmethod
    def from_synthetic(
        cls, prompt_1: str, prompt_2: str, model: Model, llm: str, n: int = 100
    ) -> "Dataset":
        df = generate_dataset(prompt_1, prompt_2, llm, n)
        train_df, test_df = split_dataset(df)
        train_df = add_negatives(train_df)
        test_df = add_negatives(test_df)
        df = pd.concat([train_df, test_df])
        add_embeddings(df, model)
        return cls(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, model: Model) -> "Dataset":
        train_df, test_df = split_dataset(df)
        train_df = add_negatives(train_df)
        test_df = add_negatives(test_df)
        df = pd.concat([train_df, test_df])
        add_embeddings(df, model)
        return cls(df)

    def plot_similarities(self):
        pass  # TODO
