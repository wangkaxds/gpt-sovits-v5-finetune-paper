# GPT-SoVITS V5 的严格无参考单说话人推理：实现边界、条件通路审计与负结果

**Paper:** [paper.md](paper.md) · [paper.html](paper.html) · [paper.pdf](paper.pdf)

版本：v1.1（审计修订版），2026-11-21。

这是一篇**负结果 + 条件通路审计**的实验论文。范围从“严格无参考推理”开始，不包含游戏解包、文本抽取等前置工作。

## 结论

- **L1 有证据**：`ref_free=True` 不传参考音频、拒绝参考路径、清空缓存，`ge = spk_emb`。
- **L2 只有部分证据**：缓存污染测试通过；严格模式拦截 `torchaudio.load`、`soundfile.read`、可选 `librosa.load` 与三个 TTS 参考方法，但仍不拦截任意文件读取；改名测试不完整且不可复现。
- **L3/L4 失败**：真实 test50 目标 SV **0.8398**；无参考 raw **0.8241**；C1M **0.8128**；开放域 20 句 **0.7332**。
- **pairwise 不能证明“不同人”**：生成 open20 pairwise **0.8233**，真实同说话人 **0.7684**；生成更同质。真正有判别力的是到目标质心。
- **人耳**：用户非盲判定“没有一个同一个人”；这是否定验收，不是 ABX。
- **E10–E18**：九组（E15 两臂）点估计均低于同划分 C1M-val20 **0.7909**；但权重/日志/逐样本 JSON 已删除，只能 summary-only。
- **E17/E18**：当前设置下 SV 0.541/0.545、CER 1.44/1.43；不能推出“内容通道架构上不能承载身份”。
- **C1M**：静态母带 wrapper；50 条 SV **0.8128**（低于 raw），同划分 val20 **0.7909**；不是模型。
- **Shelf-B**：50 条 SV 0.8256（唯一高于 raw 的 wrapper），但 2–8 kHz 0.206 过冲，且同一 test50 选择。

## 文件

```
paper.md / paper.html / paper.pdf   论文
figures/                             5 张图
data/                                CSV/JSON 汇总与数据字典
code/                                推理、母带、数据导出、图、HTML、校验
```

## 复现边界

**只复现统计/图（不需要模型/音频）**：

```bash
cd reference-free
python code/export_data.py
python code/make_figures.py
python code/make_html.py
python code/validate_paper.py
```

**需要本地 V5 包 + 权重 + 语料**：

```bash
export V5_ROOT=/path/to/GPT-SoVITS-v5
export CORPUS_ROOT=/path/to/private/corpus
python code/infer_noref.py --v5-root "$V5_ROOT" --checkpoint weights/e2_v5dev.pth \
  --text "おはようございます。" --out samples/hello.wav
python code/export_real_baseline.py --v5-root "$V5_ROOT" --corpus-root "$CORPUS_ROOT"
```

`infer.py` 是 `code/infer_noref.py` 的入口。权重、音频、语料均不发布，因此**不能从模型推理开始完整复现**；论文只做数据一致性验证、已有产物重新评测和公开代码静态/语法测试。

## 数据与版权

- 论文不包含音频、角色语音数据或模型权重。
- 角色名与作品名仅用于说明实验对象。
- 目标 SV 质心来自训练参考，仅用于评测，不进入推理。
- 人耳结论来自用户非盲试听，不是 ABX。
- `data/c1_fir.npy` 是静态 FIR 系数，不是音频或权重。

## 已知限制

- E10–E18 只有均值，工件已删除。
- E15/E16 无逐样本 JSON，不能确认机制。
- 没有训练 seed 重复；3 个采样 seed 不是训练 seed。
- Whisper/ERes2NetV2 版本未固定 hash。
- 开放域只有 20 句。
