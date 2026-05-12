import sys
try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_model = "alchin2/lora-project"
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model)

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_model, subfolder="trump")
    print("Success")
except Exception as e:
    print("ERROR:", e)

