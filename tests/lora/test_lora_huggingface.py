from typing import List

import pytest

from vllm.lora.models import LoRAModel
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.lora.utils import get_adapter_absolute_path

# Provide absolute path and huggingface lora ids
lora_lst = [sql_lora_files, sql_lora_huggingface_id]


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints_from_huggingface(
    lora_name,
):
    supported_lora_modules = LlamaForCausalLM.supported_lora_modules
    packed_modules_mapping = LlamaForCausalLM.packed_modules_mapping
    embedding_modules = LlamaForCausalLM.embedding_modules
    embed_padding_modules = LlamaForCausalLM.embedding_padding_modules
    expected_lora_modules: List[str] = []
    for module in supported_lora_modules:
        if module in packed_modules_mapping:
            expected_lora_modules.extend(packed_modules_mapping[module])
        else:
            expected_lora_modules.append(module)

    lora_path = get_adapter_absolute_path(sql_lora_huggingface_id)

    # lora loading should work for either absolute path and hugggingface id.
    lora_model = LoRAModel.from_local_checkpoint(
        lora_path,
        expected_lora_modules,
        lora_model_id=1,
        device="cpu",
        embedding_modules=embedding_modules,
        embedding_padding_modules=embed_padding_modules)

    # Assertions to ensure the model is loaded correctly
    assert lora_model is not None, "LoRAModel is not loaded correctly"