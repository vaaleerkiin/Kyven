"""Validate standalone compact Qwen3.5 loading, embeddings and generation on CPU."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/kyven/resources/generative_fill"))

import torch
from accelerate import init_empty_weights
from comfy_kitchen.tensor import AsymW4A8Int8Layout, TensorWiseINT8Layout
from moss_quantized import load_checkpoint
from moss_worker import Runtime
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.tokenization_utils_base import BatchEncoding

torch.set_num_threads(4)
torch.manual_seed(12)
config = Qwen3_5TextConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=16,
    layer_types=["linear_attention", "full_attention"], full_attention_interval=2,
    max_position_embeddings=128, partial_rotary_factor=0.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
        "mrope_section": [1, 1, 2], "mrope_interleaved": True, "partial_rotary_factor": 0.5},
    pad_token_id=0, bos_token_id=1, eos_token_id=2)
config._attn_implementation = "sdpa"
dense = Qwen3_5ForCausalLM(config).bfloat16().eval()
state = {}
for name, value in dense.state_dict().items():
    key = "model.language_model."+name[len("model."):] if name.startswith("model.") else name
    prefix = key.removesuffix("weight")
    if value.ndim == 2 and name.endswith("weight"):
        int8 = name in {"model.embed_tokens.weight", "lm_head.weight"}
        if int8:
            data, params = TensorWiseINT8Layout.quantize(value, per_channel=True,
                convrot=True, convrot_groupsize=16)
            state[prefix+"weight_scale"] = params.scale
        else:
            data, params = AsymW4A8Int8Layout.quantize(value, group_size=16, convrot_groupsize=16)
            state[prefix+"weight_s_rel"] = params.scale
            state[prefix+"weight_s_channel"] = params.s_channel
            if params.codebook is not None:
                state[prefix+"weight_codebook"] = params.codebook
        state[key] = data
        options = {"format": "int8_tensorwise" if int8 else "asym_w4a8_int8",
                   "convrot": True, "convrot_groupsize": 16, "group_size": 16}
        state[prefix+"comfy_quant"] = torch.tensor(list(json.dumps(options).encode()), dtype=torch.uint8)
    else:
        state[key] = value.clone()
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/"tiny-qwen.safetensors"
    save_file(state, path)
    with init_empty_weights():
        model = Qwen3_5ForCausalLM(config)
    model, _ = load_checkpoint(model, path, language=True)
    ids = torch.tensor([[1, 10, 32, 17]])
    with torch.no_grad():
        reference = dense(ids, use_cache=False).logits.float()
        result = model(ids, use_cache=False).logits.float()
        embeddings = model.model(ids, use_cache=False).last_hidden_state
        generated = model.generate(ids, max_new_tokens=4, do_sample=False, use_cache=True)
    assert torch.isfinite(result).all() and embeddings.shape == (1, 4, 64)
    assert generated.shape[1] > ids.shape[1]
    error = float((result-reference).abs().max())
    assert error < 0.5, error


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 10+len(text) % 20, 2]])})


runtime = Runtime(ROOT / "models/moss-sft-int8-edit-lanpaint", "cpu")
loads = []


def load_tiny_language():
    loads.append(True)
    runtime.lm, runtime.tokenizer = model, TinyTokenizer()


runtime.language_model = load_tiny_language
first = runtime.encode_prompt("Remove the person")
assert runtime.encode_prompt("Remove the person") is first
assert len(loads) == 1 and runtime.condition_cache_hit
assert runtime.lm is None and runtime.tokenizer is None
assert first.positive.embeds.shape == (1, 3, 64)
for i in range(5):
    runtime.encode_prompt(f"Replace object {i}")
assert len(runtime.condition_cache) == 4
print(json.dumps({"ok": True, "compact_embeddings_and_generation": True,
                  "max_logit_error_tiny_model": error, "device": "cpu", "bounded_prompt_cache": True}))
