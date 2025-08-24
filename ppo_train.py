import torch, random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "out_sft/model"   # SFT 结果
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token

# policy 带 value head（会在 SFT 权重上加一个 V head）
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE, torch_dtype=torch.float16 if torch.cuda.is_available() else None
).to(DEVICE)

# 参考模型让 TRL 自动克隆冻结
cfg = PPOConfig(
    batch_size=2, mini_batch_size=2, forward_batch_size=2,
    learning_rate=1e-6, ppo_epochs=4, target_kl=0.1
)
trainer = PPOTrainer(cfg, policy, ref_model=None, tokenizer=tok)

# 奖励模型（放 CPU 节省显存也可以）
rm_tok = AutoTokenizer.from_pretrained("out_rm/model")
rm = AutoModelForSequenceClassification.from_pretrained("out_rm/model").to("cpu").eval()

prompts = [
    "Define dropout in deep learning.",
    "Describe label smoothing in classification.",
    "Explain what is regularization briefly."
]

def get_reward_batch(prompts, responses):
    with torch.no_grad():
        texts = [f"User: {p}\nAssistant: {r}" for p,r in zip(prompts,responses)]
        batch = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        logits = rm(**batch).logits[:,1]    # 正类分数
        return logits.to(DEVICE)

for _ in range(60):  # 迭代若干步，跑通就行；数据多时可加大
    batch_prompts = random.sample(prompts, k=min(len(prompts), cfg.batch_size))
    # 编码 query
    queries = tok([f"User: {p}\nAssistant:" for p in batch_prompts],
                  return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        output = policy.generate(**queries, max_new_tokens=64, do_sample=True, temperature=0.7)
    # 只取生成部分作为 response_tensors
    response_tensors = []
    responses = []
    for out, q in zip(output, queries["input_ids"]):
        resp = out[q.shape[-1]:]
        response_tensors.append(resp.unsqueeze(0))
        responses.append(tok.decode(resp, skip_special_tokens=True))
    rewards = get_reward_batch(batch_prompts, responses)
    trainer.step(queries["input_ids"], response_tensors, rewards)

policy.save_pretrained("out_ppo/model"); tok.save_pretrained("out_ppo/model")
print("PPO done.")
