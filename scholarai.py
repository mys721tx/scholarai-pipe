"""
title: ScholarAI API Pipe
id: scholarai_api_pipe
author: Yishen Miao
author_url: https://github.com/mys721tx
funding_url: https://github.com/mys721tx
description: This pipeline integrates ScholarAI's Responses API with
    Open WebUI, enabling academic search, citation, and real-time
    assistant response streaming using the ScholarAI API. It supports
    custom API keys, debug mode, and is designed specifically for
    ScholarAI's API endpoints.
version: 0.1.0
license: GPL-3.0
requirements: httpx>=0.27.0

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
This pipeline integrates ScholarAI's Responses API with Open WebUI,
enabling academic search and citation features. Drop it in to use
ScholarAI's models.

Supported features:
   - Academic search and retrieval from ScholarAI
   - Citations and references in responses
   - Streaming assistant responses and reasoning in real-time
   - Debug mode (DEBUG=True) for inspecting requests and streamed
     events
   - Custom API keys (users can use their own API keys for requests)

Notes:
   - This pipeline is designed for ScholarAI's API and may not work
     with other APIs.
   - This pipeline is still in development and may have bugs. Use at
     your own risk.
"""

import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.scholarai.io/api"


class Pipe:
    class Valves(BaseModel):
        """
        Configuration for ScholarAI API connection and behavior.
        """
        API_KEY: str = Field(
            default=os.getenv("SCHOLARAI_API_KEY") or "",
            description=(
                "Your ScholarAI API key. If left blank, the environment "
                "variable 'SCHOLARAI_API_KEY' will be used instead."
            ),
        )

        BASE_URL: str = Field(
            default=os.getenv("SCHOLARAI_API") or DEFAULT_BASE_URL,
            description=(
                "The base URL for the ScholarAI API. Defaults to the "
                "official ScholarAI endpoint. Supports custom endpoints as "
                "well."
            ),
        )

        STREAM: bool = Field(
            default=True,
            description=(
                "When True, enables streaming mode for responses by default."
            ),
        )

        DEBUG: bool = Field(
            default=False,
            description=(
                "When True, prints debug statements to the console. "
                "Do not leave this on in production."
            ),
        )

    def __init__(self) -> None:
        """
        Initialize the Pipe instance and its valves configuration.
        """
        print("Pipe __init__ called!")
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        """
        Returns a list containing a single dictionary representing the
        ScholarAI model.

        :returns: A list with one dictionary, each containing the 'id' and
                  'name' of the ScholarAI model.
        :rtype: list[dict[str, str]]
        """
        # Only return the ScholarAI model as a hardcoded option
        return [
            {
                "id": "scholarai",
                "name": "ScholarAI",
            }
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
    ) -> AsyncGenerator[Any, None]:
        """
        Sends a chat completion request to the ScholarAI API and yields
        the response.

        If streaming is enabled, yields each line of the streamed
        response with the 'data: ' prefix. Otherwise, yields the full
        JSON response.

        :param body: The request payload.
        :type body: dict
        :param __user__: The user context (unused).
        :type __user__: dict

        :yields: Streamed lines (as strings) or the full response (as
                 dict).
        :rtype: AsyncGenerator[Any, None]
        """
        print(f"pipe:{__name__}")
        headers: dict[str, str] = {
            "x-scholarai-api-key": self.valves.API_KEY,
            "Content-Type": "application/json",
        }

        stream: bool = body.get("stream", self.valves.STREAM)
        model_id: str = body["model"][body["model"].find(".") + 1 :]
        payload = {**body, "model": model_id}
        print(payload)

        if self.valves.DEBUG:
            print(f"Request headers: {headers}")
            print(f"Request payload: {payload}")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if stream:
                    async with client.stream(
                        "POST",
                        f"{self.valves.BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as response:
                        response.raise_for_status()
                        # Use add_prefix to wrap the streaming lines
                        async for line in add_prefix(response.aiter_lines()):
                            yield line
                else:
                    resp: httpx.Response = await client.post(
                        f"{self.valves.BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    yield resp.json()
        except httpx.RequestError as e:
            yield f"Error: {e}"


async def add_prefix(
    generator: AsyncIterator[str],
) -> AsyncIterator[str]:
    """
    Prepends 'data: ' to each yielded line and appends 'data: [DONE]'
    at the end.

    :param generator: An async iterator yielding lines.
    :type generator: AsyncIterator[str]

    :yields: Each line prefixed with 'data: ', and a final
             'data: [DONE]'.
    :rtype: AsyncIterator[str]
    """
    async for line in generator:
        if not line.strip():
            continue
        yield line if line.startswith("data:") else f"data: {line}"
    yield "data: [DONE]"
