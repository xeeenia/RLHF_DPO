import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat(model_dir, prompt):
    tok = AutoTokenizer.from_pretrained(model_dir); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    inp = tok(f"User: {prompt}\nAssistant:", return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=80, do_sample=True, temperature=0.7)
    return tok.decode(out[0], skip_special_tokens=True).split("Assistant:",1)[-1].strip()

prompt = "Define dropout in deep learning."
print("\nSFT  =>", chat("out_sft/model", prompt))
print("PPO  =>", chat("out_ppo/model", prompt))
print("DPO  =>", chat("out_dpo/model", prompt))
