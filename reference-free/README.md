# GPT-SoVITS V5 的严格无参考单说话人推理：条件通路审计与系统性负结果

**Paper:** [paper.md](paper.md) · [paper.html](paper.html) · [paper.pdf](paper.pdf)

版本：v1.0（开放域复核版），2026-11-21。

这是一篇**负结果 + 条件通路审计**的实验论文。范围从“严格无参考推理合同”开始，不包含游戏解包、文本抽取等前置工作。

## 结论

- 严格无参考推理路径通过：只输入日文文本；不读参考音频/文本/参考特征；`STRICT_NO_REF=1` 下 `torchaudio.load`/`set_ref_audio` 抛异常；缓存污染测试清空；参考目录改名后新进程仍能生成 48 kHz 音频。
- 50 条 held-out 台词 × 3 seeds：SV **0.8258**，同文本参考控制 **0.8938**；CER 中位数 **0.2692**，均值 0.4078，2/150 长重复。
- 20 句完全新编开放域文本：CER **0.0754**，目标 SV **0.7332**（范围 0.666–0.860），pairwise 均值 **0.823**、最低 **0.634**。
- 用户试听后判定“**没有一个同一个人**”；按验收标准，单说话人稳定性失败。
- E10–E18 的九组条件注入/解冻（E15 两个生成臂）全部失败；E17/E18 解冻内容编码器后 SV 0.541/0.545、CER 1.44/1.43。
- 条件通路审计：`ge` 由 `enc_p.mrte`（E2b 冻结）与 `wns1.enc.cond_layer`（E2b 训练但弱）消费；`spk_emb` 范数 1.4622 vs `ref_enc` 25.34，`wns1` 说话人分量只有该层 bias 的 0.41 倍。CFM 的 `spk_film` 零初始化，E2b 实际未把 `ge` 接进 CFM。
- **C1M 是静态母带 wrapper**（低切 <600 Hz、软压缩、2 kHz +0.75 dB、RMS .1228），不是模型级说话人学习。

## 文件

```
paper.md / paper.html / paper.pdf   论文
figures/                             5 张图（流程、实验阶梯、频带、开放域、证伪链）
data/                                CSV 汇总（实验、开放域、pairwise、频带、划分）
code/                                export_data.py、make_figures.py、make_html.py、validate_paper.py
```

## 复现

只复现统计/图（不需要 GPT-SoVITS）：

```bash
python paper/code/validate_paper.py
python paper/code/make_figures.py
python paper/code/make_html.py
```

`export_data.py` 需要本地音频与 ERes2NetV2；它从 `logs/unseen20_c1m/` 重建逐句 CSV 与 pairwise 矩阵。

## 数据与版权

- 论文不包含音频、角色语音数据或模型权重。
- 角色名与作品名仅用于说明实验对象。
- 评测用的目标 SV 质心来自训练参考，仅用于评测，不进入推理；推理路径不读参考。
- 人耳结论来自用户非盲试听，不是 ABX；论文把它单独列为验收项。

## 已知限制

- 开放域只有 20 句，不足以估计总体分布。
- E15/E16 的逐样本 JSON 未保留，只有均值。
- 未做多说话人、跨语言、跨风格。
- 未上传音频、角色语音或权重；本地实验产物仍在 `D:\杂活\甜蜜女友3\gsv-v5-reference-free`。
