import argparse
import inspect
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with transformers + peft.")
    parser.add_argument("--base-model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--train", default="data/lora/train.jsonl")
    parser.add_argument("--val", default="data/lora/val.jsonl")
    parser.add_argument("--out", default="data/lora_adapter")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help="Comma-separated LoRA target modules, 'all-linear', or empty for inferred modules.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--no-eval", action="store_true", help="Skip validation during training.")
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Missing LoRA training dependencies. Run: python3 -m pip install transformers peft datasets accelerate sentencepiece"
        ) from exc

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    target_modules = parse_target_modules(args.target_modules)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules or infer_lora_targets(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    try:
        model = get_peft_model(model, lora_config)
    except Exception:
        if target_modules == "all-linear":
            lora_config.target_modules = infer_lora_targets(model)
            model = get_peft_model(model, lora_config)
        else:
            raise

    dataset = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
        cache_dir=str(cache_dir / "datasets"),
    )

    def tokenize(row):
        prompt_text = format_messages(
            tokenizer, row["messages"][:-1], add_generation_prompt=True
        )
        full_text = format_messages(tokenizer, row["messages"], add_generation_prompt=False)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        overflow = max(0, len(full_ids) - args.max_length)
        input_ids = full_ids[overflow:]
        visible_prompt_length = max(0, len(prompt_ids) - overflow)
        labels = input_ids.copy()
        labels[: min(visible_prompt_length, len(labels))] = [-100] * min(
            visible_prompt_length, len(labels)
        )
        attention_mask = [1] * len(input_ids)
        pad_length = args.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            labels = labels + [-100] * pad_length
        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    training_args = TrainingArguments(**build_training_args_kwargs(args, torch, TrainingArguments))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=None if args.no_eval else tokenized["validation"],
    )
    result = trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = {"train_loss": float(result.training_loss), "adapter_dir": args.out}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "train_result.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_target_modules(value: str) -> list:
    if value.strip().lower() == "all-linear":
        return "all-linear"
    return [item.strip() for item in value.split(",") if item.strip()]


def build_training_args_kwargs(args, torch, training_args_cls) -> dict:
    kwargs = {
        "output_dir": args.out,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.lr,
        "logging_steps": 1,
        "save_steps": max(args.max_steps, 1),
        "eval_steps": max(args.max_steps // 2, 1),
        "report_to": [],
        "remove_unused_columns": False,
    }
    signature = inspect.signature(training_args_cls)
    eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_key] = "no" if args.no_eval else "steps"
    if "use_mps_device" in signature.parameters:
        kwargs["use_mps_device"] = torch.backends.mps.is_available()
    return kwargs


def infer_lora_targets(model) -> list:
    module_names = {name.split(".")[-1] for name, _module in model.named_modules()}
    llama_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if any(name in module_names for name in llama_targets):
        return [name for name in llama_targets if name in module_names]
    if "c_attn" in module_names:
        return ["c_attn"]
    if "query" in module_names and "value" in module_names:
        return ["query", "value"]
    raise ValueError("Could not infer LoRA target modules for this base model.")


def format_messages(tokenizer, messages: list, add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


if __name__ == "__main__":
    main()
