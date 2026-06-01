from pathlib import Path

from web_app import DEFAULT_HISTORY, HF_MODEL_PRESETS, load_web_history, StyleLabHandler


EXPECTED_PRESETS = {
    "hf_obama": "Obama_v2",
    "hf_trump": "trump",
    "hf_twain": "Twain_v1",
    "hf_enron": "Enron",
    "hf_jefferson": "Jefferson_Model",
}


def main() -> None:
    assert DEFAULT_HISTORY.name == "per_user.json", "Default history must be checked in, not local generated data"
    histories = load_web_history(Path("per_user.json"))
    StyleLabHandler.histories = histories

    presets = {preset["id"]: preset for preset in HF_MODEL_PRESETS}
    assert set(presets) == set(EXPECTED_PRESETS), f"Unexpected presets: {sorted(presets)}"

    for preset_id, subfolder in EXPECTED_PRESETS.items():
        preset = presets[preset_id]
        assert preset["adapter_path"] == "alchin2/lora-project"
        assert preset["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert preset["adapter_subfolder"] == subfolder

    handler = object.__new__(StyleLabHandler)
    users = handler.serialize_users()
    assert len(users) == 5, f"Expected 5 LoRA voices, found {len(users)}"

    enron = next(user for user in users if user["user_id"] == "hf_enron")
    assert enron["query_count"] >= 1, "Enron validation examples were not loaded"
    assert enron["profile_count"] >= 1, "Enron validation profile was not loaded"

    print("Validation passed: 5 HF LoRA presets are wired and Enron checks are available.")


if __name__ == "__main__":
    main()
