import torch, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 准备模型与分词器
MODEL_ID = "distilgpt2"                 # 4GB 更稳；换 "gpt2" 也可
MAX_LEN = 256  # 文本的最大长度为 256 个 token。这个长度用于之后截断（truncation）或填充（padding）文本序列，确保输入输出长度一致。
tok = AutoTokenizer.from_pretrained(MODEL_ID)  # 使用 transformers 库的 AutoTokenizer 自动加载对应模型的分词器
tok.pad_token = tok.eos_token # 让分词器在 padding 的时候，用 [EOS] 来填充句子，而不是 undefined 的 [PAD]。

# 读数据并整理成对话格式
ds = load_dataset("json", data_files={"train":"data/sft.jsonl"})["train"]
def to_text(ex): return {"text": f"User: {ex['prompt']}\nAssistant: {ex['response']}"}
ds = ds.map(to_text, remove_columns=ds.column_names)

# 分词并构造标签
def tokenize(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = out["input_ids"].copy()
    return out
ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if torch.cuda.is_available() else None
)
# LoRA 适配
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                  target_modules=["c_attn","c_proj"])
model = get_peft_model(model, lora) # 返回的 model 还是个能正常 forward 的因果语言模型，但参数量大幅减少（只剩 LoRA 参数可更新）。
model.enable_input_require_grads()  # 让输入参与反传（gradient checkpointing 需要）.丢掉前向传播中的中间激活，只在需要反传时重新计算。
model.config.use_cache = False      # 与 gradient checkpointing 兼容
model.train()  # model.train() = 训练模式，启用 dropout、更新 BN 统计量。 model.eval() = 推理模式，固定权重，不再随机。
model.to("cuda" if torch.cuda.is_available() else "cpu") # 把模型移动到指定设备

# 训练配置
args = TrainingArguments(
    output_dir="out_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_torch",
    gradient_checkpointing=True
)

# 开训 + 保存
Trainer(model=model, args=args, train_dataset=ds).train()
model.save_pretrained("out_sft/model"); tok.save_pretrained("out_sft/model")
print("SFT done.")
