import torch, random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 用 GPU 优先，否则用 CPU
BASE = "out_sft/model"   # 使用 SFT 阶段保存的模型作为 PPO 的初始化
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token  # 加载分词器，并设置 pad_token（必要，否则 generate 会报错）

# 加载 SFT 微调后的模型，并加上 PPO 所需的 Value Head（自动添加）。policy 带 value head（会在 SFT 权重上加一个 V head）
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    BASE, torch_dtype=torch.float16 if torch.cuda.is_available() else None
).to(DEVICE)

# PPO 配置（可以理解为 RL 超参数）
cfg = PPOConfig(
    batch_size=2,  # 每轮 PPO 使用的样本数量
    mini_batch_size=2,  # 每轮内部分成多少 mini-batch 做优化
    forward_batch_size=2,  # 前向计算时的 batch（防止 OOM）
    learning_rate=1e-6, 
    ppo_epochs=4,  # 每轮训练做几次 PPO 优化
    target_kl=0.1  # 控制 policy 不要偏离 ref_model 太远
)
# 初始化 PPOTrainer；ref_model=None 表示自动克隆一个冻结的旧 policy
trainer = PPOTrainer(cfg, policy, ref_model=None, tokenizer=tok)

# 加载奖励模型（训练好的分类器，判别 chosen vs rejected）
rm_tok = AutoTokenizer.from_pretrained("out_rm/model")  # 分词器
rm = AutoModelForSequenceClassification.from_pretrained("out_rm/model").to("cpu").eval()  # 模型放 CPU 节省显存

# 准备用于训练的 prompt（真实项目里会更多样）
prompts = [
    "Define dropout in deep learning.",
    "Describe label smoothing in classification.",
    "Explain what is regularization briefly."
]

# 奖励函数：将生成的回答输入 reward model，输出正类得分作为 reward
def get_reward_batch(prompts, responses):
    with torch.no_grad():
        texts = [f"User: {p}\nAssistant: {r}" for p,r in zip(prompts,responses)]
        batch = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
        logits = rm(**batch).logits[:,1]    # 取 label=1 的得分作为 reward
        return logits.to(DEVICE)

# PPO 训练主循环
for _ in range(60):  # 训练 60 步（demo 用较小，真实项目可设大）
    # 从 prompt 中随机取一个 batch（这里 batch_size=2）
    batch_prompts = random.sample(prompts, k=min(len(prompts), cfg.batch_size))
    # 构造输入 query（不含回答部分）
    queries = tok([f"User: {p}\nAssistant:" for p in batch_prompts],
                  return_tensors="pt", padding=True).to(DEVICE)
    # 使用当前 policy 生成回答（do_sample=True 启用采样）
    with torch.no_grad():
        output = policy.generate(**queries, max_new_tokens=64, do_sample=True, temperature=0.7)
    # 解析生成结果：把回答部分单独截取出来
    response_tensors = []  # 存储 token tensor
    responses = []  # 存储文本回答
    for out, q in zip(output, queries["input_ids"]):
        resp = out[q.shape[-1]:]  # 截取生成部分
        response_tensors.append(resp.unsqueeze(0))
        responses.append(tok.decode(resp, skip_special_tokens=True))
    # 用 reward model 给回答打分
    rewards = get_reward_batch(batch_prompts, responses)
    # PPO 核心一步：更新 policy（让它更偏好高 reward 回答）
    trainer.step(queries["input_ids"], response_tensors, rewards)
    
# 保存 PPO 微调后的模型
policy.save_pretrained("out_ppo/model"); tok.save_pretrained("out_ppo/model")
print("PPO done.")
