import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 读入数据
rows=[]
for l in open("data/rm_pairs.jsonl", "r", encoding="utf-8"):
    r=json.loads(l)
    rows += [
        {"text": f"User: {r['prompt']}\nAssistant: {r['chosen']}", "label":1},
        {"text": f"User: {r['prompt']}\nAssistant: {r['rejected']}", "label":0},
    ]
ds = Dataset.from_list(rows) # 转换为 HuggingFace Dataset，方便后续 .map() 处理。(对整个数据集做“批量处理 + 转换”)

# 分词
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokf(b):
    out = tok(b["text"], truncation=True, padding="max_length", max_length=384)
    out["labels"] = b["label"]
    return out
ds = ds.map(tokf, batched=True, remove_columns=["text","label"]) # 对整个数据集做“批量处理 + 转换”

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
args = TrainingArguments(
    output_dir="out_rm",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)
Trainer(model=model, args=args, train_dataset=ds).train()
model.save_pretrained("out_rm/model"); tok.save_pretrained("out_rm/model")
print("RM done.")
