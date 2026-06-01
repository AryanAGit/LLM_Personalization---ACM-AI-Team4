# LoRA Voice Lab

Frontend and lightweight local API for generating text with the team's Hugging Face LoRA voice adapters.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_app.py
```

Then open http://127.0.0.1:8787.

The first LoRA request downloads the base model and selected adapter into `data/hf_cache`. That folder is ignored by git and is only a cache; a fresh clone does not need any local model files checked in.

## Voices

All presets use `Qwen/Qwen2.5-1.5B-Instruct` as the base model and `alchin2/lora-project` as the adapter repo.

| UI voice | Adapter subfolder |
| --- | --- |
| Barack Obama | `Obama_v2` |
| Donald Trump | `trump` |
| Mark Twain | `Twain_v1` |
| Enron | `Enron` |
| Thomas Jefferson | `Jefferson_Model` |

## Validate

```bash
python3 validate_app.py
```

This checks that the five Hugging Face presets are wired to the expected adapter folders and that the Enron held-out examples load for the Test Bench.

## Project Shape

- `web/` contains the browser UI.
- `web_app.py` serves the frontend and exposes `/api/users`, `/api/model-presets`, `/api/generate`, `/api/compare`, and `/api/test`.
- `lora_backend.py` loads the Hugging Face base model and PEFT adapters.
- `enron_style.py` contains the prompt, fallback, and scoring helpers still used by the local API.
- `evaluators.py` contains the validation scoring metrics.
- `per_user.json` is the checked-in Enron validation/demo source.

The template fallback is kept only so the UI can still be opened and inspected without downloading model weights. The intended project path is the LoRA backend with the five Hugging Face adapter presets above.
