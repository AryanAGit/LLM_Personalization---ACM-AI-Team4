from functools import lru_cache
from pathlib import Path
from typing import Sequence


DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_CACHE_DIR = "data/hf_cache"


def generate_with_peft(
    messages: Sequence[dict],
    base_model: str,
    adapter_path: str = "",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    if not base_model:
        raise ValueError("A Hugging Face base model path or id is required for PEFT/LoRA generation.")

    tokenizer, model = load_peft_pipeline(base_model, adapter_path)
    prompt = format_messages(tokenizer, messages)
    context_limit = max_context_tokens(model, max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_limit)
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=min(max_new_tokens, 140),
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def resolve_adapter_path(adapter_path: str = "", adapter_root: str = "", user_id=0) -> str:
    if adapter_path:
        return str(Path(adapter_path).expanduser())
    if not adapter_root or not user_id:
        return ""
    candidate = Path(adapter_root).expanduser() / str(user_id)
    return str(candidate) if candidate.exists() else ""


def max_context_tokens(model, max_new_tokens: int) -> int:
    config = getattr(model, "config", None)
    limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    if not limit:
        return 4096
    return max(128, int(limit) - max_new_tokens)


@lru_cache(maxsize=2)
def load_peft_pipeline(base_model: str, adapter_path: str = ""):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing LoRA inference dependencies. Run: python3 -m pip install transformers peft accelerate sentencepiece safetensors"
        ) from exc

    model_path = str(Path(base_model).expanduser()) if Path(base_model).expanduser().exists() else base_model
    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=DEFAULT_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, cache_dir=DEFAULT_CACHE_DIR)
    adapter_path = str(Path(adapter_path).expanduser()) if adapter_path else ""
    if adapter_path:
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return tokenizer, model


def format_messages(tokenizer, messages: Sequence[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"
