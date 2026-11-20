# GPT-SoVITS V5 微调：系统性负结果与工程审计

**Paper:** [paper.md](paper.md) · [paper.pdf](paper.pdf)
**Repo:** https://github.com/wangkaxds/gpt-sovits-v5-finetune-paper

这是一篇**负结果 + 两个推理侧效应 + 一次工程审计**的实验论文，范围从微调实验开始，不包含游戏解包、数据清洗等前置工作。

## 结论（v2.2）

- 标准 q/v/out LoRA 在 1,200 句单说话人、6 GB、V5dev 上**没有达到稳定/可用的提升**；`vo16_s300` 在 3 个 seed 上均值 SV **−0.0004**（t=−0.75）。
- **turbo8** 相对 turbo4：SV **+0.0052**（配对 t p=.013；符号检验 5/5，p=.0625）；ASR −0.0045（p=.135）。
- **cut5** 分句：3 个配对 seed 的 SV **+0.0244**（配对 t p=.011；Wilcoxon/符号检验 p=.25，n=3）；cut5 自身 5 seed 均值 0.9041，cut0 3 seed 均值 0.8811；ASR 3-seed 0.9209 vs 0.9203。cut5 的 seed1 平均音频时长 4.324 s vs cut0 3.963 s（+9.1%），采样步数相同但总延迟/GPU 未测，不能断言总成本相同。
- **best-of-5 已撤回**：SV 增益是 max-of-k 选择偏差（P≈0.89），第二个 ASR 只 +0.0023。
- **工程审计**：上游 `TTS.py` 把 LoRA 目标硬编码为 q/k/v/out，导致 `ada8` 完全不加载、`all8` 的 FF 被忽略、`last5` 混合 Checkpoint 加载未验证。修复后逐模块验证 5 个 Checkpoint 的 330 个目标模块，全部符合 `(α/r)BA`（最大绝对误差 1.87e-7，未修改层变化 0，无 NaN/Inf）；`last5` 无权重，未验证。修复后重测：`ada8_s300` test SV +0.0002 / ASR −0.008；`all8_s1200` test SV +0.0010（CI +0.0003…+0.0017）/ ASR +0.003（均 seed1）。all8 在本样本内 p=.0084、CI +0.0003…+0.0017，但仅 seed1、效应量 0.001、未过多重比较校正，仍不能称可用或稳定增益。`last5` 权重已删，无法复核。
- **ref_enc 解冻**后 Loss 下降但 SV/ASR 同降，与“训练目标与感知目标不一致”的假设一致；机制仍未被单独验证。

## 证据与索引

- 19 条路线的证据矩阵：`data/evidence_matrix.csv`（训练日志、配置、Checkpoint、运行时加载、val/test、seed、统计、人耳听感）。
- 19 条路线的失败分类与完整记录：`data/route_ledger.csv`（A–F 分类、训练过程、未排除解释）。
- 关键结果 → 数据 → 脚本 → 论文章节：`data/result_index.csv`。
- 逐模块 LoRA 合并审计：`data/lora_merge_verification.csv`、`data/lora_merge_module_audit.csv`。

## 文件

```
paper.md / paper.html / paper.pdf   论文 v2.2
figures/                            8 张图（流程、结构、微调、turbo、cut5、选择偏差、Loss、参数）
data/                               汇总数据、证据矩阵、结果索引、逐模块审计 CSV
code/                               图表、统计、日志解析、LoRA 模块审计脚本
```

## 复现

只复现统计/图表（不需要 GPT-SoVITS）：

```bash
pip install -r requirements.txt
python code/stats_cut5.py
python code/stats_finetune_retest.py
python code/stats_sensitivity.py
python code/make_figures.py
python code/make_more_figures.py
python code/validate_paper.py
```

需要本地 V5 包和 Checkpoint 的审计：

```bash
set GPT_SOVITS_ROOT=<GPT-SoVITS-v5-package>
set GSV_TRAIN_LOG=<path-to-train.log>
set V5_EVAL_ROOT=<path-to-v5_ft_eval>
python code/audit_lora_checkpoints.py
python code/test_lora_target_inference.py
python code/test_lora_loader_guards.py
```

`validate_paper.py` 会重算 cut5、vo16、turbo8、ada8/all8 的均值/p 值/MDE，检查 19 条路线的证据矩阵与分类、结果索引、逐模块 LoRA 合并审计、图引用，以及仓库是否误含音频或权重。

## 数据与版权

本仓库**不包含音频、角色语音数据、模型权重**。论文中的指标来自本地实验；角色名与作品名仅用于说明实验对象。

**可复现边界**：上面的 CSV/JSON 足以复现统计表格和图表；但不能独立复现语音生成或训练，因为 1,200 句语料、参考音频、底模和 Checkpoint 未公开。`last5/last5b` 权重已删除，无法复现。

## 引用

如果这篇记录对你有用，请引用本仓库 URL，并注明 GPT-SoVITS V5 的原始项目。
