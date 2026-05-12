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

Use `--no-ollama` or `--backend fallback` for the deterministic offline fallback.
Use `--backend peft` for Hugging Face/PEFT generation with a base model and optional LoRA adapter:

```bash
python3 project.py generate \
  --history data/processed/user_email_history.json \
  --backend peft \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path /path/to/qwen-lora-adapter \
  --prompt "Write a reply to this email in the user's style: Can we reschedule Friday's call?"
```

For per-user adapters, omit `--adapter-path` and point `--adapter-root` at a directory containing one adapter folder per `user_id`:

```bash
python3 project.py generate \
  --history data/processed/user_email_history.json \
  --user-id 20000001 \
  --backend peft \
  --base-model /path/to/huggingface/llama-compatible-model \
  --adapter-root data/lora_adapters \
  --prompt "Write a reply to Wendy: Can you send the files by Wednesday?"
```

## Evaluate

```bash
python3 project.py evaluate --history data/processed/user_email_history.json --limit 5
```

The generated validation split currently gives the model only subject lines for many gold emails, so low content-overlap scores are expected. Use free-form prompts with enough incoming-message context to evaluate realistic replies.

## Web Interface

```bash
python3 run_app.py
```

Open `http://127.0.0.1:8787`. The page includes a prompt composer, backend selector, Qwen/LoRA fields, and a Test Bench that compares a held-out incoming prompt, the real user response, the generated response, and similarity/style scores. The default backend is the local fallback so the site works even when Ollama or a Hugging Face model is not loaded.

## LoRA And RAG Direction

Use the generated JSON as supervised fine-tuning data by converting each query into an instruction/output pair where the instruction includes the style profile plus retrieved examples, and the output is the held-out gold email. The intended split of responsibility is:

- Base Llama-family model: general English and instruction following.
- RAG: prompt-time examples from the selected user's past emails.
- LoRA: the selected user's writing style, trained as a small adapter.

## Historical Or Personal Voice Corpora

The pipeline can also convert plain text documents into the same profile/query schema used by the Enron data. This is the recommended path for demos with public-domain authors such as Lincoln or Shakespeare, or for a user's own writing samples:

```bash
python3 project.py build-author \
  --input data/raw/lincoln.txt \
  --out data/authors/lincoln/user_email_history.json \
  --author-id lincoln \
  --display-name "Abraham Lincoln" \
  --profile-size 80 \
  --query-count 12
```

Then export one adapter dataset per personality:

```bash
python3 project.py export-lora \
  --history data/authors/lincoln/user_email_history.json \
  --out data/lora_lincoln \
  --per-user
```

Each personality should get its own LoRA adapter directory under `data/lora_adapters/<author-id>`. Reuse the same base model, but train separate LoRA weights for each target voice.

The benchmark plan in `docs/BENCHMARKING.md` tracks style distance, copy risk, and human evaluation separately so the project rewards recognizable style without rewarding memorized source text.

Export mixed and per-user LoRA SFT records:

```bash
python3 project.py export-lora \
  --history data/processed/user_email_history.json \
  --out data/lora \
  --per-user
```

Verify the local PEFT/LoRA stack with a tiny no-download-from-Ollama smoke train:

```bash
python3 lora_smoke_train.py --train data/lora/train.jsonl --out data/lora_smoke_adapter --steps 3
```

Train a real LoRA adapter from a Hugging Face-compatible causal LM:

```bash
python3 project.py train-user-lora \
  --base-model /path/to/huggingface/llama-compatible-model \
  --lora-dir data/lora \
  --user-id 20000001 \
  --adapter-root data/lora_adapters \
  --max-steps 100
```

This writes the adapter to `data/lora_adapters/20000001`, which the web app and `generate --backend peft --adapter-root data/lora_adapters` can load automatically for that user.

Ollama model blobs are not directly trainable by PEFT. For real Llama LoRA, use a Hugging Face-format Llama model path or model id. LoRA adapters only work with the base model family they were trained against, so a Qwen adapter cannot be used with Llama/Ollama directly.
