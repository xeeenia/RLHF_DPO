import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

BASE = "out_sft/model"
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token
ds = load_dataset("json", data_files={"train":"data/dpo.jsonl"})["train"]

args = TrainingArguments(
    output_dir="out_dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_steps=5, save_strategy="epoch", optim="adamw_torch"
)

model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16 if torch.cuda.is_available() else None
)
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                  target_modules=["c_attn","c_proj"])
model = get_peft_model(model, lora)
model.to("cuda" if torch.cuda.is_available() else "cpu")

trainer = DPOTrainer(
    model=model, ref_model=None, beta=0.1,
    args=args, train_dataset=ds, tokenizer=tok, max_length=256
)
trainer.train()
trainer.model.save_pretrained("out_dpo/model"); tok.save_pretrained("out_dpo/model")
print("DPO done.")
