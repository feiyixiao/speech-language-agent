# Speech Language Learning Agent — 设计与时间线

> **2026-09-30 更新**：Phase 1–3 的大部分已完成，但和原计划有几处不同。**可观测性**改用本地 JSONL trace 加 `scripts/trace_report.py`，不依赖 W&B/Weave 账号，以后可以接 Langfuse 或 Cloud Logging。**评估**是 eval/ 下的自建种子集加 v1/v2/v3 对比，W&I+LOCNESS 还没接入。**部署**目标从 HF Spaces 改成 FastAPI + Docker + Cloud Run，对口 Retorio 这类岗位；CI 已写好，但还没真正部署。架构、结果和踩坑分别见 README.md、eval/RESULTS.md、INCIDENTS.md。

面向语言学习者的语音反馈智能体。用户上传一段口语录音（或直接输入文本），系统转录后同时给出 **语法纠错 / 词汇建议 / 逐词发音评分** 三类反馈，并跨轮记忆用户的历史错误。目标语言：英/德/日/普通话。

这份文档记录整体设计、4 阶段时间线，以及 **MLOps（W&B / Docker / HF Spaces）如何自然融入**——注意本项目主要是「推理型 LLM 应用」，不训练大模型，所以 MLOps 的用法和训练类项目不同。

---

## 1. 现状（Phase 0，已完成）

原型链路已跑通：

```
音频 ──Whisper──▶ transcript ──▶ [LangGraph router]
                                     ├─ grammar_node      (Groq / Llama-3.1-8b)  语法纠错 + 错误历史记忆
                                     ├─ vocabulary_node   (Groq / Llama-3.1-8b)  词汇改进建议
                                     ├─ pronunciation_node(Azure Speech)         逐词发音评分
                                     └─ rag_node          (Chroma + Groq)        语法知识问答
                                              │
                                          Streamlit UI (ui/app.py)
```

- **ASR**: `agent/asr.py` — OpenAI Whisper（base），转录 + 语种检测
- **路由与节点**: `agent/nodes.py` + `agent/graph.py` — LangGraph StateGraph 条件路由
- **发音评估**: `agent/pronunciation.py` — Azure Pronunciation Assessment，词级 accuracy/fluency/completeness
- **RAG**: `agent/rag.py` — 英/德语法 md 文档 → Chroma 持久化向量库
- **前端**: `ui/app.py` — Streamlit，音频上传 + 文本输入两种模式，含 session 历史

**当前缺口**：① 没有任何评估（不知道 router 路对没、语法纠错准不准、发音分靠不靠谱）；② 没有可观测性（每轮 agent 内部发生了什么无法回看）；③ 没打包，别人跑不起来；④ 没在线 demo。以下 4 个阶段依次补齐。

---

## 2. MLOps 如何自然融入（关键说明）

本项目不训练模型，所以 **W&B 不是用来记 training loss 的**，而是两个更贴合 LLM 应用的用法：

### ① W&B Weave —— agent 可观测性（对应「实验追踪」信号）
Weave 是 W&B 面向 LLM 应用的追踪工具。给关键函数加一个装饰器，它就自动记录**每一轮**：router 选了哪条路、输入 transcript、各节点输出的反馈、token 数、每步延迟。这就是 agent 版的「实验追踪」——让你（和看简历的人）能回看系统每一次决策。

```python
import weave
weave.init("speech-language-agent")

@weave.op()
def router_node(state): ...      # 自动记录输入/输出/延迟

@weave.op()
def grammar_node(state): ...     # 每次调用都留痕，可在 W&B 面板逐条回放
```

### ② W&B 记录评估指标（对应「有真实数字」信号）
Phase 2 建评估集后，把每次评估 run 的指标 log 到 W&B，形成可对比的 run 历史（换 prompt、换模型后指标怎么变）：

```python
import wandb
wandb.init(project="speech-language-agent", job_type="eval")
wandb.log({"router_acc": 0.92, "grammar_F1": 0.81, "pron_corr_with_human": 0.74})
```

### ③ Docker（对应「可复现」信号）
本项目依赖重（whisper / langchain / chromadb / azure-sdk / streamlit）且需要 API key 环境变量，正是 Docker 的典型场景。一个 `Dockerfile` 让别人 `docker build` + `docker run` 一步复现。

### ④ HuggingFace Spaces（对应「能点的 demo」信号）
已经是 Streamlit，天然可部署到 HF Spaces。API key 用 Space secrets 注入；免费 CPU tier 用 whisper-base 即可。产出一个别人打开浏览器就能试的链接。

---

## 3. 四阶段时间线

| 阶段 | 目标 | 主要工作 | 产出的求职信号 | 预估 |
|---|---|---|---|---|
| **Phase 1** 可观测性与稳固 | 把原型变「可回看、不崩」 | 集成 **W&B Weave** 追踪各节点；补配置管理（.env 校验、模型/语言可配）；错误处理（Azure/Groq 调用失败降级）；清理 app.py 里 `os.unlink` 时机等小 bug | 「会给 agent 加可观测性」 | 3–5 天 |
| **Phase 2** 评估 | 让项目有**真实数字** | 建小评估集：(a) 20–30 条学习者句子标注 gold 路由标签 → 测 router 准确率；(b) 用 W&I+LOCNESS 子集测语法纠错 P/R；(c) 一批带人工发音评分的录音 → 测 Azure 分与人工分相关性。指标全部 log 到 **W&B** | 「有评估、有数字」——最缺的一环 | 1–1.5 周 |
| **Phase 3** 打包与部署 | 让别人**跑得起来、点得到** | 写 `Dockerfile`；部署到 **HuggingFace Spaces**（API key 作 secret）；README 加架构图 + 在线 demo 链接 + W&B 报告链接 | 「工程化 + 能点的 demo」 | 3–5 天 |
| **Phase 4**（可选/加分） | 差异化扩展 | 二选一：**(A) 口音反馈**——加 wav2vec2 口音分类头（复用 LID 项目经验），给「你的发音更接近 X 口音」反馈；**(B) 学习曲线**——持久化用户历史，画个人错误随时间下降曲线 | 「不只是调 API，能加自研模型」 | 1–2 周 |

**建议执行顺序**：Phase 1 → 2 → 3 一定按序（评估依赖可观测性，部署依赖前两步收尾）。Phase 4 视精力选一个即可，A（口音）和你整体语音画像更契合。

---

## 4. 和其它项目的联动

- **口音扩展（Phase 4A）** 直接复用 `~/LID-project` 的 wav2vec2 微调经验。
- **HF Space 部署经验**可迁移给 LID 项目（「上传语音→判断语种」demo）和语音共情项目。
- **W&B 用法**与 DementiaBank 探针项目（记录逐层 probe 指标）互通，两个项目一起挂 W&B，整体「实验追踪」信号更实。

---

_最后更新：2026-07-02_
