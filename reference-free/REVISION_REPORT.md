# 修订报告：Reference-Free 论文 v1.1

日期：2026-11-21  
公开路径：`reference-free/`  
提交：`6f9df86`（上一版 `a1a85b1`）

## 1. 这轮改了什么

### 1.1 结论层

- 把“reference-free”拆成 L1–L4，并明确当前状态：**L1 有代码/运行时证据；L2 部分；L3 未成立；L4 失败**。不再写“严格无参考已成功”。
- 把 E17/E18 从“内容通道不能承载身份”降为“当前优化设置下失败”；列出 lr/优化器/数据/训练-推理不一致等替代解释。
- 把 §7.2 的“双标定点”降为描述性假设；`ge` 进入 `enc_p` 与 `wns1` 是代码事实，但“两个消费者要求冲突尺度”未证。

### 1.2 数字与可比性

- **真实同说话人基线**：test50 目标 SV **0.8398**；pairwise **0.7684**（前 20 条同 n **0.7530**）。此前没有这条基线。
- **pairwise 反向纠正**：生成 open20 pairwise **0.8233**，高于真实同说话人；不能再用 pairwise 证明“不是同一个人”。主判据改为到目标质心。
- **同划分 C1M-val20**：补跑 `logs/e2b_pilot20_c1m/`，SV **0.7909**、CER **0.3523**；图 2 不再用 test50 的 **0.8128** 作 val20 基线。
- **C1M CER 跨划分**：val20 从 raw **0.3268** 到 C1M **0.3523**（4/20 句变化），test50 从 **0.2900** 到 **0.2874**；方向相反，已写明不可直接相减。
- **Shelf-B**：50 条 SV **0.8256**（唯一高于 raw 的 wrapper），但 2–8 kHz **0.206** 过冲；同一 test50 选择，不能当独立泛化。
- **权重审计修正**：`spk_emb` 向量每通道 RMS **0.0646**；`Wg` 输出每通道 RMS **0.0332**（0.41× bias）；×17.33 后 **0.576**（7.17×）。此前把 0.0646 当成 `Wg`，已修正。
- **初始化缺陷**：`s2_train_v3_lora.py` 在加载预训练 `ref_enc` 之前调用 `init_spk_emb`，E2b 的 `spk_emb` 从随机 `ref_enc(neutral)` 出发；已写入 §3.5 与限制。
- **E1/E2b CER**：`experiments.csv` 的 E1 **0.8733**、E2b **0.3268** 已改为日志值。

### 1.3 合同与证据边界

- `decode_encp` 在 `ge is None` 时仍有 `ref_enc` 回退；无参考路径因 `ge=spk_emb` 不触发，但论文不再说“没有 ref_enc 分支”。
- `STRICT_NO_REF=1` 的公开版本拦截六个 API（`torchaudio.load`、`sf.read`、可选 `librosa.load`、`TTS.set_ref_audio`、`_set_ref_spec`、`_set_prompt_semantic`）；仍不拦截任意文件读取，且历史运行时日志来自较早的两 API 版本，已写入限制。
- 改名测试只覆盖 `data/kan/wavs`，没有覆盖 `logs/kan/4-cnhubert`、`5-wav32k`，且状态不可复现；不再作为完整 L2 证明。
- E2b 零化探针没有保留工件，正文不再引用 **0.8039**；L3 只写“未成立”。

### 1.4 工件与代码

新增：

- `code/infer_noref.py`、`code/c1_master.py`、`code/postprocess_noref.py`、`infer.py`；
- `data/unseen20.txt`、`data/split_ids.csv`、`data/provenance.csv`；
- `data/open20_score.json`、`data/c1m_val20_score.json`；
- `data/real_test50.csv`、`data/real_pairwise.csv`、`data/real_pairwise20.csv`、`data/real_baseline.json`；
- `data/heldout_summary.csv`、`data/worst3.csv`、`data/worst3_*.txt`、`data/weight_audit.csv`、`data/diagnostics.csv`；
- `requirements.txt`、`requirements-infer.txt`。

修正：

- `export_data.py` 不再读音频/模型，只从 `open20_score.json` + `unseen20.txt` 重建 CSV；
- `export_real_baseline.py` 参数化为 `--v5-root/--corpus-root`；
- `validate_paper.py` 从“常量自证”改为逐句 CSV/JSON 重算、文件存在性、绝对路径、真实 id 与 test id 对齐、无音频/权重检查；
- README 路径改成公开仓库布局，执行顺序改为 export → figures → html → validate。

## 2. 验证

本会话实际执行：

- `python code/validate_paper.py` → **103 passed, 0 failed**；
- `python code/export_data.py` → 20 行重建；
- `python code/make_figures.py` → 5 图；
- `python code/make_html.py` → HTML；
- Edge headless → PDF **16 页、5 图**，1,245,881 B；
- 公开仓 raw 检查：`paper.pdf` 200 / 1,245,881 B，`paper.md` 200，`paper.html` 200，`infer_noref.py` 200。

未执行：

- 未重训 E2b；未对六 API 版本重跑私有语料；
- 未从音频/权重开始完整复现；
- 未做多人盲听/ABX；用户听感仍是非盲单听众。

## 3. 研究结论

- 合同层：无参考文本路径成立（L1）；缓存清空有部分证据（L2）。
- 身份层：失败。held-out 目标 SV **0.8241（raw）/0.8128（C1M）**，真实 **0.8398**；开放域 **0.7332**。
- 感知层：失败。用户“没有一个同一个人”；pairwise 高不能反驳，因为真实同说话人 pairwise 更低。
- 机制层：`ge` 进入 `wns1`/`enc_p` 有代码与权重证据；E10–E18 是 summary-only，E17/E18 只能说明当前配方崩溃。
- C1M：频响 wrapper，不是模型；不改善 CPP，test50 SV 低于 raw。

## 4. 仍未解决

1. E10–E18 权重/日志/逐样本 JSON 已删除，不能逐样本复算；表中只能 summary-only。
2. E16 的 `spk_t` 注入代码不在当前树，无法外部复核。
3. 没有训练 seed 重复；3 个采样 seed 不等于训练 seed。
4. 权重/音频/语料不公开，不能从推理开始完整复现。
5. 开放域只有 20 句，且是 C1M 后音频；真实基线是 raw，差异含约 0.011 wrapper 效应。
6. 参考控制 SV 0.8938 高于真实 0.8398，说明 SV 会奖励原型化输出；不能把 SV 当人耳替代。
7. 初始化顺序缺陷未重训；若要重做，应把 `init_spk_emb` 移到预训练加载之后，并重新跑 E2b。
8. 若要继续，优先补逐样本工件与同划分对照，再考虑改架构；继续在冻结 ckpt 上换 lr/注入点没有证据支持。
