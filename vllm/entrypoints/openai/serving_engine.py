import json
import pathlib
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field
from typing_extensions import Annotated

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              EmbeddingRequest, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              ModelCard, ModelList,
                                              ModelPermission, TokenizeRequest,
                                              UnloadLoraAdapterRequest)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class PromptAdapterPath:
    name: str
    local_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: Optional[str] = None


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
    ):
        super().__init__()

        self.engine = engine
        self.model_config = model_config
        self.max_model_len = model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            tokenizer_revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
            truncation_side="left")

        self.base_model_paths = base_model_paths

        self.lora_requests = []
        if lora_modules is not None:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_path=lora.path,
                    base_model_id=lora.base_model_name if lora.base_model_name
                    and self._is_model_supported(lora.base_model_name) else
                    self.base_model_paths[0].name)
                for i, lora in enumerate(lora_modules, start=1)
            ]

        self.prompt_adapter_requests = []
        if prompt_adapters is not None:
            for i, prompt_adapter in enumerate(prompt_adapters, start=1):
                with pathlib.Path(prompt_adapter.local_path,
                                  "adapter_config.json").open() as f:
                    adapter_config = json.load(f)
                    num_virtual_tokens = adapter_config["num_virtual_tokens"]
                self.prompt_adapter_requests.append(
                    PromptAdapterRequest(
                        prompt_adapter_name=prompt_adapter.name,
                        prompt_adapter_id=i,
                        prompt_adapter_local_path=prompt_adapter.local_path,
                        prompt_adapter_num_virtual_tokens=num_virtual_tokens))

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=base_model.name,
                      max_model_len=self.max_model_len,
                      root=base_model.model_path,
                      permission=[ModelPermission()])
            for base_model in self.base_model_paths
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=lora.lora_local_path,
                      parent=lora.base_model_id,
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        prompt_adapter_cards = [
            ModelCard(id=prompt_adapter.prompt_adapter_name,
                      root=self.base_model_paths[0].name,
                      permission=[ModelPermission()])
            for prompt_adapter in self.prompt_adapter_requests
        ]
        model_cards.extend(lora_cards)
        model_cards.extend(prompt_adapter_cards)
        return ModelList(data=model_cards)

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str

    async def _check_model(
        self, request: Union[ChatCompletionRequest, CompletionRequest,
                             DetokenizeRequest, EmbeddingRequest,
                             TokenizeRequest]
    ) -> Optional[ErrorResponse]:
        if self._is_model_supported(request.model):
            return None
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return None
        if request.model in [
                prompt_adapter.prompt_adapter_name
                for prompt_adapter in self.prompt_adapter_requests
        ]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_adapter(
        self, request: Union[CompletionRequest, ChatCompletionRequest,
                             EmbeddingRequest]
    ) -> Tuple[Optional[str], Optional[Union[LoRARequest,
                                             PromptAdapterRequest]]]:
        if self._is_model_supported(request.model):
            return None, None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return 'LoRA', lora
        for prompt_adapter in self.prompt_adapter_requests:
            if request.model == prompt_adapter.prompt_adapter_name:
                return 'PromptAdapter', prompt_adapter
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _validate_prompt_and_tokenize(
            self,
            request: Union[ChatCompletionRequest, CompletionRequest,
                           DetokenizeRequest, EmbeddingRequest,
                           TokenizeRequest],
            prompt: Optional[str] = None,
            prompt_ids: Optional[List[int]] = None,
            truncate_prompt_tokens: Optional[Annotated[int,
                                                       Field(ge=1)]] = None,
            add_special_tokens: Optional[bool] = True
    ) -> Tuple[List[int], str]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if (prompt and prompt_ids):
            raise ValueError(
                "Only one of prompt or prompt_ids should be provided.")

        if prompt_ids is None:
            # When using OpenAIServingChat for chat completions, for
            # most models the special tokens (e.g., BOS) have already
            # been added by the chat template. Therefore, we do not
            # need to add them again.
            # Set add_special_tokens to False (by default) to avoid
            # adding the BOS tokens again.
            tokenizer_kwargs: Dict[str, Any] = {
                "add_special_tokens": add_special_tokens
            }
            if truncate_prompt_tokens is not None:
                tokenizer_kwargs.update({
                    "truncation": True,
                    "max_length": truncate_prompt_tokens,
                })
            input_ids = self.tokenizer(prompt, **tokenizer_kwargs).input_ids
        elif truncate_prompt_tokens is not None:
            input_ids = prompt_ids[-truncate_prompt_tokens:]
        else:
            input_ids = prompt_ids

        input_text = prompt if prompt is not None else self.tokenizer.decode(
            prompt_ids)
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, EmbeddingRequest):
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.", )
            return input_ids, input_text

        # Note: TokenizeRequest and DetokenizeRequest doesn't have max_tokens
        # and does not require model context length validation
        if isinstance(request, (TokenizeRequest, DetokenizeRequest)):
            return input_ids, input_text

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.", )
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.", )
        else:
            return input_ids, input_text

    def _get_decoded_token(self, logprob: Logprob, token_id: int) -> str:
        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return self.tokenizer.decode(token_id)

    async def _check_load_lora_adapter_request(
            self, request: LoadLoraAdapterRequest) -> Optional[ErrorResponse]:
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return self.create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name already exists
        if any(lora_request.lora_name == request.lora_name
               for lora_request in self.lora_requests):
            return self.create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' has already been"
                "loaded.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        return None

    async def _check_unload_lora_adapter_request(
            self,
            request: UnloadLoraAdapterRequest) -> Optional[ErrorResponse]:
        # Check if either 'lora_name' or 'lora_path' is provided
        if not request.lora_name and not request.lora_int_id:
            return self.create_error_response(
                message=
                "either 'lora_name' and 'lora_int_id' needs to be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        # Check if the lora adapter with the given name exists
        if not any(lora_request.lora_name == request.lora_name
                   for lora_request in self.lora_requests):
            return self.create_error_response(
                message=
                f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST)

        return None

    async def load_lora_adapter(
            self,
            request: LoadLoraAdapterRequest) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_load_lora_adapter_request(request)
        if error_check_ret is not None:
            return error_check_ret

        lora_name, lora_path = request.lora_name, request.lora_path
        unique_id = len(self.lora_requests) + 1
        self.lora_requests.append(
            LoRARequest(lora_name=lora_name,
                        lora_int_id=unique_id,
                        lora_path=lora_path))
        return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(
            self,
            request: UnloadLoraAdapterRequest) -> Union[ErrorResponse, str]:
        error_check_ret = await self._check_load_lora_adapter_request(request)
        if error_check_ret is not None:
            return error_check_ret

        lora_name = request.lora_name
        self.lora_requests = [
            lora_request for lora_request in self.lora_requests
            if lora_request.lora_name != lora_name
        ]
        return f"Success: LoRA adapter '{lora_name}' removed successfully."

    def _is_model_supported(self, model_name):
        return any(model.name == model_name for model in self.base_model_paths)
