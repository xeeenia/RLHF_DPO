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
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)  # 如果文本太长（超过 max_length），就截断
    out["labels"] = out["input_ids"].copy()  # 将输入的 input_ids 拷贝一份作为标签 labels，用于语言模型的训练。
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
    output_dir="out_sft",  # 指定训练过程中的输出目录
    per_device_train_batch_size=1,  # 每个设备（GPU/CPU）的 batch size，设置为 1 → 一次只喂 1 个样本
    gradient_accumulation_steps=8,  # 梯度累积的步数，累积 8 次 batch_size=1 的梯度，再做一次反向更新
    learning_rate=2e-5,
    num_train_epochs=3,  # 训练总轮数 = 3，一个 epoch = 数据集完整跑一遍
    fp16=torch.cuda.is_available(),  # 是否启用半精度浮点
    logging_steps=5,  # 每隔多少 step 打印一次日志（loss 等信息）
    save_strategy="epoch",  # 模型保存策略，每个 epoch 结束时保存一次 checkpoint
    optim="adamw_torch",  # 优化器选择，adamw_torch → PyTorch 原生的 AdamW（权重衰减版 Adam）
    gradient_checkpointing=True  # 开启梯度检查点，在反向传播时，不保存所有中间激活值，而是需要时重新计算。减少显存消耗（可换显存换算力），缺点：训练速度会稍微变慢。
)

# 开训 + 保存
Trainer(model=model, args=args, train_dataset=ds).train()
model.save_pretrained("out_sft/model"); tok.save_pretrained("out_sft/model")
print("SFT done.")
