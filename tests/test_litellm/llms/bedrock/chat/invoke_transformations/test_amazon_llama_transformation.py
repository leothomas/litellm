import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.llms.bedrock.chat.invoke_transformations.amazon_llama_transformation import (
    AmazonLlamaConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.types.utils import ModelResponse


def _make_raw_response(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode(),
        headers={
            "x-amzn-bedrock-input-token-count": "10",
            "x-amzn-bedrock-output-token-count": "5",
        },
    )


class TestAmazonLlamaConfig:
    def test_logprobs_in_supported_params(self):
        config = AmazonLlamaConfig()
        params = config.get_supported_openai_params(model="meta.llama3-3-70b-instruct-v1:0")
        assert "logprobs" in params

    def test_logprobs_mapped_to_return_logprobs(self):
        config = AmazonLlamaConfig()
        result = config.map_openai_params(
            non_default_params={"logprobs": True},
            optional_params={},
            model="meta.llama3-3-70b-instruct-v1:0",
            drop_params=False,
        )
        assert result.get("return_logprobs") is True

    def test_logprobs_false_not_mapped(self):
        config = AmazonLlamaConfig()
        result = config.map_openai_params(
            non_default_params={"logprobs": False},
            optional_params={},
            model="meta.llama3-3-70b-instruct-v1:0",
            drop_params=False,
        )
        assert "return_logprobs" not in result

    def test_transform_response_with_logprobs(self):
        config = AmazonLlamaConfig()
        raw_body = {
            "generation": "Hello, world!",
            "logprobs": [
                {"1234": -0.5},
                {"5678": -1.2},
                {"91011": -0.3},
            ],
        }
        raw_response = _make_raw_response(raw_body)
        model_response = ModelResponse()
        result = config.transform_response(
            model="invoke/meta.llama3-3-70b-instruct-v1:0",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Hello, world!"
        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert len(logprobs.content) == 3

        assert logprobs.content[0].token == "1234"
        assert logprobs.content[0].logprob == pytest.approx(-0.5)
        assert logprobs.content[0].top_logprobs == []

        assert logprobs.content[1].token == "5678"
        assert logprobs.content[1].logprob == pytest.approx(-1.2)

        assert logprobs.content[2].token == "91011"
        assert logprobs.content[2].logprob == pytest.approx(-0.3)

    def test_transform_response_without_logprobs(self):
        config = AmazonLlamaConfig()
        raw_body = {"generation": "Hello!"}
        raw_response = _make_raw_response(raw_body)
        model_response = ModelResponse()
        result = config.transform_response(
            model="invoke/meta.llama3-3-70b-instruct-v1:0",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=None,
            request_data={},
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Hello!"
        assert getattr(result.choices[0], "logprobs", None) is None
