# GPT-SoVITS V5 微调：系统性负结果与工程审计

**Paper:** [paper.md](paper.md) · [paper.pdf](paper.pdf)  
**Repo:** https://github.com/wangkaxds/gpt-sovits-v5-finetune-paper

这是一篇**负结果 + 一个真实增益 + 一次工程审计**的实验论文，范围从微调实验开始，不包含游戏解包、数据清洗等前置工作。

## 结论（v2.0）

- 标准 q/v/out LoRA 在 1,200 句单说话人、6 GB、V5dev 上**没有超过零样本**；`vo16_s300` 在 3 个 seed 上均值 SV **−0.0004**（t=−0.75）。
- **turbo8** 相对 turbo4：SV **+0.0052**（p=.013）；ASR −0.0045（p=.135）。
- **cut5** 分句：30 句 test、5 个 seed，SV **0.9041** vs cut0 **0.8811**，ASR 0.9236 vs 0.9203；成本不变。
- **best-of-5 已撤回**：SV 增益是 max-of-k 选择偏差（P≈0.89），第二个 ASR 只 +0.0023。
- **工程审计**：上游 `TTS.py` 把 LoRA 目标硬编码为 q/k/v/out，导致 `ada8` 完全不加载、`all8` 的 FF 被忽略、`last5` 全参加载未验证。修复后重测：`ada8_s300` test SV +0.0002 / ASR −0.008；`all8_s1200` test SV +0.0010（CI +0.0003…+0.0017）/ ASR +0.003（均 seed1）。all8 刚过配对 MDE≈0.0010，但效应量只有 0.001，仍不能称可用增益。`last5` 权重已删，无法复核。
- **ref_enc 解冻**后 Loss 下降但 SV/ASR 同降，是最强的“目标不匹配”证据。

## 文件

```
paper.md / paper.html / paper.pdf   论文 v2.0
figures/                            8 张图（流程、结构、微调、turbo、cut5、选择偏差、Loss、参数）
data/                               可复现的汇总数据 CSV
code/                               图表、统计、日志解析、LoRA 模块审计脚本
```

## 复现

```bash
python code/make_figures.py
python code/make_more_figures.py
python code/parse_train_log.py
python code/summarize_train_runs.py
python code/stats_finetune_retest.py
python code/stats_cut5.py
python code/inspect_lora_modules.py
```

统计脚本依赖 numpy；图表依赖 matplotlib；LoRA 审计需要 GPT-SoVITS V5 本地包与 PyTorch。

## 数据与版权

本仓库**不包含音频、角色语音数据、模型权重**。论文中的指标来自本地实验；角色名与作品名仅用于说明实验对象。

## 引用

如果这篇记录对你有用，请引用本仓库 URL，并注明 GPT-SoVITS V5 的原始项目。

