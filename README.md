# TTS Voice C: Fine-Tuning vs Zero-Shot on GPT-SoVITS V5

**Paper:** [paper.md](paper.md) · [paper.pdf](paper.pdf)  
**Repo:** https://github.com/wangkaxds/gpt-sovits-v5-finetune-paper

这是一篇**负结果 + 一个真实增益**的实验论文，范围从微调实验开始，不包含游戏解包、数据清洗等前置工作。

## 结论

- 1,200 句单说话人数据、6 GB 显存、V5dev 上，**没有任何微调显著超过零样本 v5dev**。
- 最好的 `vo16_s300` 在 seed1–3 重测中 SV 均值 **−0.0004**（t=−0.75），仍为零。
- turbo8 相对 turbo4：SV **+0.0052**（p=.013）；ASR −0.0045（p=.135）。
- **cut5** 分句：30 句 test、5 个 seed，SV **0.9041** vs cut0 **0.8811**，ASR 0.9236 vs 0.9203；成本不变。
- **best-of-5 已撤回**：SV 增益是 max-of-k 选择偏差（P≈0.89），第二个 ASR 只 +0.0023。

最终候选：`turbo8 + ref1 + seed1 + cut5`。仍需人耳验收。

## 文件

```
paper.md / paper.html / paper.pdf   论文
figures/                            5 张图
data/*.csv                          可复现的汇总数据
code/                               图表、统计脚本
```

## 复现

```bash
python code/make_figures.py
python code/stats_finetune_retest.py
python code/stats_cut5.py
```

统计脚本只依赖 numpy；图表依赖 matplotlib。

## 数据与版权

本仓库**不包含音频、角色语音数据、模型权重**。论文中的指标来自本地实验；角色名与作品名仅用于说明实验对象。

## 引用

如果这篇记录对你有用，请引用本仓库 URL，并注明 GPT-SoVITS V5 的原始项目。
