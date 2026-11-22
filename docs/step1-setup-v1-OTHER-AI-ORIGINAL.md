# 第一步：准备环境（约 1–2 小时）

对应设计 v3（核心版，硬上限 $500）。

## 1. 只申请 1 个 key：OpenRouter

- openrouter.ai 注册，先充 **$50**（不够再充，总共不超过 $500）。
- 在设置里给这个 key 设花费上限（Credit limit），比如 $50。
- 这一个 key 就能调用 GPT、Claude、Gemini、Qwen、DeepSeek，不用分别找各家申请。
- 用个人邮箱和个人信用卡，不要用公司账号或公司电脑。

## 2. 检查 key（5 分钟）

在 `check_keys.py` 同目录建一个 `.env` 文件，写一行 `OPENROUTER_API_KEY=你的key`，然后：

```bash
pip install litellm python-dotenv
python check_keys.py
```

它会让 5 个模型各答一道算术题并报信心，打印 OK/FAIL 和花费（加起来不到 1 分钱）。如果某个模型名报错，去 openrouter.ai/models 查最新名字改一下。

## 3. 跑通 τ²-bench 5 题（约 1 小时，花费 < $1）

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
uv sync                 # 需要先装 uv
cp .env.example .env    # 填入 OPENROUTER_API_KEY
tau2 run --domain airline \
  --agent-llm openrouter/openai/gpt-5-mini \
  --user-llm openrouter/openai/gpt-5-mini \
  --num-trials 1 --num-tasks 5
```

结果在 `data/simulations/`。（命令来自官方 README，它通过 LiteLLM 调模型，所以 `openrouter/...` 这种写法应该可以用；如果不行告诉我。）

## 4. SWE-bench 部分先不做

SWE 新任务要用 Docker 和 x86 机器，等 2 月租云机器时再做。

## 做完后告诉我

把 `check_keys.py` 的输出和 τ²-bench 的结果文件发给我。我接着写"让 agent 报信心"的代码。
