# Standard library
import os
import asyncio

# Third party
from openai import AsyncOpenAI, BadRequestError, OpenAIError
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type


ASYNC_CLIENT = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def format_completion_params(
    messages,
    model,
    stream,
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    presence_penalty,
    tools,
    tool_choice,
    parallel_tool_calls,
    response_format,
):
    completion_params = {
        "model": model,
        "messages": [m.to_openai_message() for m in messages],
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
        "response_format": response_format,
    }

    if tools:
        completion_params["tools"] = tools
        completion_params["tool_choice"] = tool_choice
        completion_params["parallel_tool_calls"] = parallel_tool_calls
        del completion_params["response_format"]

    return completion_params

@retry(
    retry=retry_if_not_exception_type(BadRequestError),
    wait=wait_random_exponential(multiplier=1, max=30),
)
async def call_gpt(
    messages,
    model="gpt-4o",
    stream=False,
    max_tokens=750,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    tools=None,
    tool_choice=None,
    parallel_tool_calls=False,
    response_format={"type": "text"},
):
    completion_params = format_completion_params(
        messages,
        model,
        stream,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        tools,
        tool_choice,
        parallel_tool_calls,
        response_format,
    )

    if stream == True:
        # It will time out if the stream wouldn't start in 5 seconds
        timeout_seconds = 5
    else:
        # TODO: This is dubious and should be revisited
        # It will time out if the request wouldn't complete in 60 seconds
        timeout_seconds = 60

    try:
        # Apply a timeout to the API call
        response = await asyncio.wait_for(
            ASYNC_CLIENT.chat.completions.create(**completion_params),
            timeout=timeout_seconds,
        )

        if not response:
            raise BadRequestError("Empty response")

        if not stream:
            return response.choices[0].message

        return response
    except asyncio.TimeoutError as e:
        raise e
    except BadRequestError as e:
        raise e
    except OpenAIError as e:
        raise e
    except Exception as e:
        raise e

