from typing import List, Optional

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.chat_utils import (ConversationMessage,
                                                load_chat_template,
                                                parse_chat_message_content)
from vllm.entrypoints.openai.protocol import (DetokenizeRequest,
                                              DetokenizeResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
from vllm.entrypoints.openai.serving_engine import BaseModelPath, OpenAIServing


class OpenAIServingTokenization(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 base_model_paths: List[BaseModelPath],
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None)

        load_chat_template(self, chat_template)

    async def create_tokenize(self,
                              request: TokenizeRequest) -> TokenizeResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if not (request.prompt or request.messages):
            return self.create_error_response(
                "Either `prompt` or `messages` should be provided.")

        if (request.prompt and request.messages):
            return self.create_error_response(
                "Only one of `prompt` or `messages` should be provided.")

        if request.messages:
            conversation: List[ConversationMessage] = []

            for message in request.messages:
                conversation.extend(
                    parse_chat_message_content(self, message).messages)

            request.prompt = self.tokenizer.apply_chat_template(
                add_generation_prompt=request.add_generation_prompt,
                conversation=conversation,
                tokenize=False)

        (input_ids, input_text) = self._validate_prompt_and_tokenize(
            request,
            prompt=request.prompt,
            add_special_tokens=request.add_special_tokens)

        return TokenizeResponse(tokens=input_ids,
                                count=len(input_ids),
                                max_model_len=self.max_model_len)

    async def create_detokenize(
            self, request: DetokenizeRequest) -> DetokenizeResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        (input_ids, input_text) = self._validate_prompt_and_tokenize(
            request, prompt_ids=request.tokens)

        return DetokenizeResponse(prompt=input_text)
