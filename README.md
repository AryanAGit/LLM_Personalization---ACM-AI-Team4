# Enron Style Dataset Pipeline

This project builds three JSON files for user-style email generation from the Enron maildir:

- `profile_user.json`
- `user_email_history.json`
- `user_email_history_label.json`

It chooses high-volume users whose maildir name starts with `t` by default, then uses sent mail as profile examples and held-out sent messages as generation golds.

## Download

Configure Kaggle credentials, then either download from the Kaggle UI or run:

```bash
python3 project.py download
```

Kaggle's `wcukierski/enron-email-dataset` download commonly contains `emails.csv`. The processor accepts either that CSV path/directory or a restored maildir root with per-user folders such as `taylor/sent/`.

## Build JSON

```bash
python3 project.py build --maildir /path/to/enron-or-emails.csv --out data/processed --prefix t
```

For local Llama/Ollama descriptions:

```bash
python3 project.py build --maildir /path/to/enron-or-emails.csv --out data/processed --prefix t --use-ollama
```

## Generate A Response

`generate` uses local Ollama/Llama by default. Make sure Ollama is running and `llama3.1:8b` is installed:

```bash
ollama pull llama3.1:8b
```

```bash
python3 project.py generate \
  --history data/processed/user_email_history.json \
  --prompt "Write a reply to this email in the user's style: Can we reschedule Friday's call?"
```

Use `--no-ollama` for the deterministic offline fallback.

## Evaluate

```bash
python3 project.py evaluate --history data/processed/user_email_history.json --limit 5
```

The generated validation split currently gives the model only subject lines for many gold emails, so low content-overlap scores are expected. Use free-form prompts with enough incoming-message context to evaluate realistic replies.

## Web Interface

```bash
python3 run_app.py
```

Open `http://127.0.0.1:8787`. The page includes a prompt composer and a Test Bench that compares a held-out incoming prompt, the real user response, the Llama response, and similarity/style scores.

## LoRA And RAG Direction

Use the generated JSON as supervised fine-tuning data by converting each query into an instruction/output pair where the instruction includes the style profile plus retrieved examples, and the output is the held-out gold email. A practical first pass is RAG-only generation with the user's profile emails; add LoRA once you have enough examples and a stable evaluation split.

Export LoRA SFT records:

```bash
python3 project.py export-lora --history data/processed/user_email_history.json --out data/lora
```

Verify the local PEFT/LoRA stack with a tiny no-download-from-Ollama smoke train:

```bash
python3 lora_smoke_train.py --train data/lora/train.jsonl --out data/lora_smoke_adapter --steps 3
```

Train a real LoRA adapter from a Hugging Face-compatible causal LM:

```bash
python3 train_lora.py \
  --base-model /path/to/huggingface/llama-compatible-model \
  --train data/lora/train.jsonl \
  --val data/lora/val.jsonl \
  --out data/lora_adapter
```

Ollama model blobs are not directly trainable by PEFT. For real Llama LoRA, use a Hugging Face-format model path or model id.
