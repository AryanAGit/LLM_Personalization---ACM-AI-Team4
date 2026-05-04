import argparse
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

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        cache_dir=str(cache_dir),
    )
    target_modules = infer_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    dataset = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
        cache_dir=str(cache_dir / "datasets"),
    )

    def tokenize(row):
        text = format_messages(tokenizer, row["messages"])
        tokens = tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    training_args = TrainingArguments(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        eval_strategy="steps",
        eval_steps=max(args.max_steps // 2, 1),
        report_to=[],
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    result = trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = {"train_loss": float(result.training_loss), "adapter_dir": args.out}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "train_result.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


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


def format_messages(tokenizer, messages: list) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


if __name__ == "__main__":
    main()
