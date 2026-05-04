import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny local LoRA smoke train.")
    parser.add_argument("--train", default="data/lora/train.jsonl")
    parser.add_argument("--out", default="data/lora_smoke_adapter")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception as exc:
        raise RuntimeError(
            "LoRA dependencies are missing. Run: python3 -m pip install transformers peft datasets accelerate sentencepiece"
        ) from exc

    rows = read_rows(Path(args.train))
    if not rows:
        raise ValueError(f"No training rows found in {args.train}")

    text = "\n".join(flatten_messages(row["messages"]) for row in rows[:8])
    vocab = build_vocab(text)
    ids = torch.tensor([vocab.get(ch, vocab["?"]) for ch in text[:2048]], dtype=torch.long)
    if ids.numel() < 32:
        raise ValueError("Training text is too short for LoRA smoke training.")

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=vocab["<"],
        eos_token_id=vocab[">"],
    )
    model = GPT2LMHeadModel(config)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(args.steps):
        start = (step * 64) % max(ids.numel() - 129, 1)
        batch = ids[start : start + 128].unsqueeze(0)
        labels = batch.clone()
        output = model(input_ids=batch, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(round(float(loss.detach()), 4))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "smoke_result.json").write_text(
        json.dumps({"steps": args.steps, "losses": losses, "adapter_dir": str(out_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "losses": losses, "adapter_dir": str(out_dir)}, indent=2))


def read_rows(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_messages(messages: list) -> str:
    return "\n".join(f"<{message['role']}>\n{message['content']}" for message in messages)


def build_vocab(text: str) -> dict:
    chars = sorted(set(text) | set("<?>"))
    return {ch: index for index, ch in enumerate(chars)}


if __name__ == "__main__":
    main()
