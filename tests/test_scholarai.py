import pytest
from scholarai import add_prefix
from scholarai import Pipe
import types
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

# --- Fixtures for mock responses based on openapi.yaml ---


@pytest.fixture
def mock_chat_completion_response():
    # Example response for /chat/completions (application/json)
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1716200000,
        "model": "scholarai",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def mock_abstracts_response():
    # Example response for /abstracts (array of DocumentResponse)
    return [
        {
            "title": "Sample Paper",
            "authors": ["Alice", "Bob"],
            "abstract": "This is a sample abstract.",
            "publication_date": "2024-01-01",
            "cited_by_count": 42,
            "url": "https://example.com/paper",
            "ss_id": "SS123",
            "doi": "10.1234/example",
            "answer": "Sample answer.",
        }
    ]


@pytest.fixture
def mock_fulltext_response():
    # Example response for /fulltext (ChunkResponse)
    return {
        "chunks": [
            {
                "chunk_num": 1,
                "chunk": "Full text chunk.",
                "img_mds": [],
                "pdf_url": "https://example.com/pdf",
            }
        ],
        "total_chunk_num": 1,
        "hint": "Sample hint.",
    }


@pytest.fixture
def mock_question_response():
    # Example response for /question (ChunkResponse)
    return {
        "chunks": [
            {
                "chunk_num": 1,
                "chunk": "Answer chunk.",
                "img_mds": [],
                "pdf_url": "https://example.com/pdf",
            }
        ],
        "total_chunk_num": 1,
        "hint": "Sample hint.",
    }


@pytest.fixture
def mock_save_citation_response():
    # Example response for /save-citation
    return {"message": "Citation saved to Zotero successfully."}


@pytest.fixture
def mock_add_to_project_response():
    # Example response for /add_to_project (array of DocumentResponse)
    return [
        {
            "title": "Sample Paper",
            "authors": ["Alice", "Bob"],
            "abstract": "This is a sample abstract.",
            "publication_date": "2024-01-01",
            "cited_by_count": 42,
            "url": "https://example.com/paper",
            "ss_id": "SS123",
            "doi": "10.1234/example",
            "answer": "Sample answer.",
            "message": "Paper added to project successfully.",
            "project_id": "gpt",
            "paper_id": "DOI:10.1234/example",
        }
    ]


@pytest.fixture
def mock_create_project_response():
    # Example response for /create_project (204 No Content)
    return None


@pytest.fixture
def mock_analyze_project_response():
    # Example response for /analyze_project
    return {
        "response": "# Project Analysis\nThis is a markdown response.",
        "tool_hint": "Use this for project summary.",
    }


@pytest.fixture
def mock_patents_response():
    # Example response for /patents (array of DocumentResponse)
    return [
        {
            "title": "Sample Patent",
            "authors": ["Inventor A"],
            "abstract": "Patent abstract.",
            "publication_date": "2023-12-01",
            "cited_by_count": 10,
            "url": "https://example.com/patent",
            "ss_id": "SSPAT123",
            "doi": "10.5678/patent",
            "answer": "Patent answer.",
        }
    ]


# --- Example mock for httpx.AsyncClient usage ---
class MockAsyncResponse:
    def __init__(self, json_data=None, lines=None, status_code=200):
        self._json = json_data
        self._lines = lines or []
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class MockAsyncClient:
    def __init__(self, response=None, stream_lines=None):
        self._response = response
        self._stream_lines = stream_lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, *args, **kwargs):
        return MockAsyncResponse(json_data=self._response)

    def stream(self, *args, **kwargs):
        return MockAsyncResponse(lines=self._stream_lines)


class DummyAsyncIterator:
    def __init__(self, lines):
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_lines,expected",
    [
        (
            ["hello", "world"],
            ["data: hello", "data: world", "data: [DONE]"],
        ),
        (
            ["data: already", "prefixed"],
            ["data: already", "data: prefixed", "data: [DONE]"],
        ),
        (
            ["", "   ", "foo", "bar"],
            ["data: foo", "data: bar", "data: [DONE]"],
        ),
        (
            [],
            ["data: [DONE]"],
        ),
        (
            ["data: test", "", "baz"],
            ["data: test", "data: baz", "data: [DONE]"],
        ),
    ],
)
async def test_add_prefix(input_lines, expected):
    gen = DummyAsyncIterator(input_lines)
    result = [line async for line in add_prefix(gen)]
    assert result == expected


@pytest.mark.parametrize(
    "pipe_method",
    [
        "pipes_returns_expected_model",
        "pipes_no_extra_models",
    ],
)
def test_pipe_methods(pipe_method):
    pipe = Pipe()
    if pipe_method == "pipes_returns_expected_model":
        result = pipe.pipes()
        assert isinstance(result, list)
        assert len(result) == 1
        model = result[0]
        assert isinstance(model, dict)
        assert model.get("id") == "scholarai"
        assert model.get("name") == "ScholarAI"
    elif pipe_method == "pipes_no_extra_models":
        result = pipe.pipes()
        assert len(result) == 1
        assert all("id" in m and "name" in m for m in result)

        @pytest.mark.asyncio
        async def test_pipe_streaming(monkeypatch):
            """
            Test the pipe method with streaming enabled.
            """
            pipe = Pipe()
            pipe.valves.API_KEY = "testkey"
            pipe.valves.BASE_URL = "https://mocked"
            pipe.valves.STREAM = True
            pipe.valves.DEBUG = False

            # Mock response.aiter_lines to yield some lines
            class MockResponse:
                def __init__(self):
                    self._lines = iter(["line1", "line2"])
                    self.status_code = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                async def aiter_lines(self):
                    for line in ["line1", "line2"]:
                        yield line

                def raise_for_status(self):
                    pass

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                def stream(self, *args, **kwargs):
                    return MockResponse()

            with patch("httpx.AsyncClient", return_value=MockClient()):
                body = {"model": "scholarai.testmodel", "stream": True}
                __user__ = {}
                result = [line async for line in pipe.pipe(body, __user__)]
                # Should prefix lines and add [DONE]
                assert result == ["data: line1", "data: line2", "data: [DONE]"]

        @pytest.mark.asyncio
        async def test_pipe_non_streaming(monkeypatch):
            """
            Test the pipe method with streaming disabled.
            """
            pipe = Pipe()
            pipe.valves.API_KEY = "testkey"
            pipe.valves.BASE_URL = "https://mocked"
            pipe.valves.STREAM = False
            pipe.valves.DEBUG = False

            class MockResponse:
                def __init__(self):
                    self.status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"result": "ok"}

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                async def post(self, *args, **kwargs):
                    return MockResponse()

            with patch("httpx.AsyncClient", return_value=MockClient()):
                body = {"model": "scholarai.testmodel", "stream": False}
                __user__ = {}
                result = [line async for line in pipe.pipe(body, __user__)]
                assert result == [{"result": "ok"}]

        @pytest.mark.asyncio
        async def test_pipe_streaming_error(monkeypatch):
            """
            Test the pipe method handles httpx.RequestError in streaming mode.
            """
            pipe = Pipe()
            pipe.valves.API_KEY = "testkey"
            pipe.valves.BASE_URL = "https://mocked"
            pipe.valves.STREAM = True
            pipe.valves.DEBUG = False

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                def stream(self, *args, **kwargs):
                    raise httpx.RequestError("stream error")

            with patch("httpx.AsyncClient", return_value=MockClient()):
                body = {"model": "scholarai.testmodel", "stream": True}
                __user__ = {}
                result = [line async for line in pipe.pipe(body, __user__)]
                assert result and result[0].startswith("Error: ")

        @pytest.mark.asyncio
        async def test_pipe_non_streaming_error(monkeypatch):
            """
            Test the pipe method handles httpx.RequestError in non-streaming mode.
            """
            pipe = Pipe()
            pipe.valves.API_KEY = "testkey"
            pipe.valves.BASE_URL = "https://mocked"
            pipe.valves.STREAM = False
            pipe.valves.DEBUG = False

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

                async def post(self, *args, **kwargs):
                    raise httpx.RequestError("post error")

            with patch("httpx.AsyncClient", return_value=MockClient()):
                body = {"model": "scholarai.testmodel", "stream": False}
                __user__ = {}
                result = [line async for line in pipe.pipe(body, __user__)]
                assert result and result[0].startswith("Error: ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, mock_response, mock_lines, expected",
    [
        # Streaming mode
        (
            True,
            None,
            ["line1", "line2"],
            ["data: line1", "data: line2", "data: [DONE]"],
        ),
        # Non-streaming mode
        (False, {"result": "ok"}, None, [{"result": "ok"}]),
    ],
)
async def test_pipe_stream_and_nonstream(
    monkeypatch, stream, mock_response, mock_lines, expected
):
    pipe = Pipe()
    pipe.valves.API_KEY = "testkey"
    pipe.valves.BASE_URL = "https://mocked"
    pipe.valves.STREAM = stream
    pipe.valves.DEBUG = False

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, *args, **kwargs):
            return MockAsyncResponse(json_data=mock_response)

        def stream(self, *args, **kwargs):
            return MockAsyncResponse(lines=mock_lines)

    with patch("httpx.AsyncClient", return_value=MockClient()):
        body = {"model": "scholarai.testmodel", "stream": stream}
        __user__ = {}
        result = [line async for line in pipe.pipe(body, __user__)]
        assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, error_type",
    [
        (True, "stream"),
        (False, "post"),
    ],
)
async def test_pipe_stream_and_nonstream_error(monkeypatch, stream, error_type):
    pipe = Pipe()
    pipe.valves.API_KEY = "testkey"
    pipe.valves.BASE_URL = "https://mocked"
    pipe.valves.STREAM = stream
    pipe.valves.DEBUG = False

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.RequestError("post error")

        def stream(self, *args, **kwargs):
            raise httpx.RequestError("stream error")

    with patch("httpx.AsyncClient", return_value=MockClient()):
        body = {"model": "scholarai.testmodel", "stream": stream}
        __user__ = {}
        result = [line async for line in pipe.pipe(body, __user__)]
        assert result and result[0].startswith("Error: ")
