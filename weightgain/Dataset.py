# Standard library
import asyncio

# Third party
import json_repair
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from litellm import completion, acompletion
from sklearn.model_selection import train_test_split

# Local
try:
    from weightgain.Model import Model
    from weightgain.utilities import cosine_similarity
except ImportError:
    from .Model import Model
    from utilities import cosine_similarity


#########
# HELPERS
#########


TEST_FRACTION = 0.8
RANDOM_SEED = 123
NEGATIVES_PER_POSITIVE = 1

CHUNKS_PROMPT = """Your task is to generate synthetic data for fine-tuning an embedding model. \
This data will be used to improve retrieval performance in a question-answering system. \
Your goal is to create diverse, realistic text chunks based on a given prompt.

Here is the prompt you should base your chunks on:

<chunk_prompt>
{prompt}
</chunk_prompt>

You need to generate the following number of chunks:

<chunk_count>
{n}
</chunk_count>

Follow these instructions when generating your chunks:

<instructions>
- Read and understand the chunk prompt carefully.
- Generate <chunk_count> unique text chunks based on the chunk prompt.
- Ensure each chunk is diverse and represents a unique aspect or feature of the prompt.
- Each chunk should be realistic and based on real-world situations related to the prompt.
- Avoid repetition or highly similar chunks.
</instructions>"""

QA_PROMPT = """Your task is to generate synthetic questions based on the provided context. \
These questions will be used to fine-tune an embedding model to improve retrieval against this context.

Here is the context you should base your questions on:

<context>
{context}
</context>

You need to generate the following number of questions:

<question_count>
{n}
</question_count>

Follow these instructions when generating your questions:

<instructions>
- Read and analyze the provided context carefully.
- Generate <question_count> unique questions based on the context.
- Base your questions solely on the information in the context. DO NOT use prior knowledge.
- Ensure that your questions are diverse and cover different aspects of the context.
</instructions>"""


def generate_chunks(prompt: str, model: str, n: int = 100) -> list[str]:
    print(f"Generating chunks")

    response = completion(
        model=model,
        messages=[
            {"role": "user", "content": CHUNKS_PROMPT.format(prompt=prompt, n=n)}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chunks_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "chunks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "A text chunk based on the <chunk_prompt>.",
                            },
                            "description": f"List of text chunks based on the <chunk_prompt>. len(chunks) >= {n}.",
                        }
                    },
                    "required": ["chunks"],
                    "additionalProperties": False,
                },
            },
        },
    )
    response = response.choices[0].message.content
    response = json_repair.loads(response)
    chunks = response["chunks"][:n]

    if len(chunks) < n:
        # TODO: Generate more until we reach n
        print(f"Warning: Generated only {len(chunks)} chunks. Expected {n}.")

    return chunks


async def generate_dataset(
    chunks: list[str], model: str, n_per_chunk: int = 100
) -> pd.DataFrame:
    tasks = []
    for chunk in chunks:
        task = acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": QA_PROMPT.format(context=chunk, n=n_per_chunk),
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "questions_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "A question based on the <context>.",
                                },
                                "description": f"List of questions based on the <context>. len(questions) >= {n_per_chunk}.",
                            },
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        tasks.append(task)

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Generating questions",
    )

    dataset = []
    for chunk_id, (chunk, result) in enumerate(zip(chunks, results)):
        result = json_repair.loads(result.choices[0].message.content)
        questions = result["questions"][:n_per_chunk]

        if len(questions) < n_per_chunk:
            print(
                f"Warning: Chunk {chunk_id} generated only {len(questions)} questions."
            )

        dataset += [(chunk_id, question, chunk, 1) for question in questions]

    df = pd.DataFrame(dataset, columns=["chunk_id", "text_1", "text_2", "label"])
    return df


def sync_generate_dataset(
    chunks: list[str], model: str, n_per_chunk: int = 100
) -> pd.DataFrame:
    try:
        loop = asyncio.get_running_loop()
        df = loop.run_until_complete(generate_dataset(chunks, model, n_per_chunk))
    except RuntimeError:
        df = asyncio.run(generate_dataset(chunks, model, n_per_chunk))

    return df


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "chunk_id" in dataset.columns:
        chunk_ids = dataset["chunk_id"].unique()
        train_ids, test_ids = train_test_split(
            chunk_ids, test_size=TEST_FRACTION, random_state=RANDOM_SEED
        )

        train_df = dataset[dataset["chunk_id"].isin(train_ids)].copy()
        test_df = dataset[dataset["chunk_id"].isin(test_ids)].copy()
    else:
        train_df, test_df = train_test_split(
            dataset,
            test_size=TEST_FRACTION,
            random_state=RANDOM_SEED,
        )

    train_df.loc[:, "dataset"] = "train"
    test_df.loc[:, "dataset"] = "test"

    return train_df, test_df


def add_negatives_helper(dataset: pd.DataFrame) -> pd.DataFrame:
    texts = set(dataset["text_1"].values) | set(dataset["text_2"].values)
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair) for text_pair in dataset[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    negatives_dataset = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    negatives_dataset["label"] = -1
    negatives_dataset["dataset"] = dataset["dataset"].iloc[0]

    dataset = pd.concat(
        [
            dataset,
            negatives_dataset.sample(
                n=len(dataset) * NEGATIVES_PER_POSITIVE, random_state=RANDOM_SEED
            ),
        ]
    )
    return dataset


def add_negatives(dataset: pd.DataFrame) -> pd.DataFrame:
    train_df, test_df = split_dataset(dataset)
    train_df = add_negatives_helper(train_df)
    test_df = add_negatives_helper(test_df)
    dataset = pd.concat([train_df, test_df])
    return dataset


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
    def from_chunks(
        cls, chunks: list[str], model: Model, llm: str, n_queries_per_chunk: int = 1
    ) -> "Dataset":
        df = sync_generate_dataset(chunks, llm, n_queries_per_chunk)
        df = add_negatives(df, model)
        return cls(df)

    @classmethod
    def from_synthetic_chunks(
        cls,
        prompt: str,
        llm: str,
        n_chunks: int = 100,
        n_queries_per_chunk: int = 1,
    ) -> "Dataset":
        chunks = generate_chunks(prompt, llm, n_chunks)
        df = sync_generate_dataset(chunks, llm, n_queries_per_chunk)
        df = add_negatives(df)
        return cls(df)

    @classmethod
    def from_pairs(cls, text_pairs: list[tuple[str, str]]) -> "Dataset":
        text_pairs = [
            (index, text_1, text_2, 1)
            for index, (text_1, text_2) in enumerate(text_pairs)
        ]
        df = pd.DataFrame(text_pairs, columns=["id", "text_1", "text_2", "label"])
        df = add_negatives(df)
        return cls(df)
