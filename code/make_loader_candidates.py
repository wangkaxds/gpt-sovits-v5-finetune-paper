import json, os
ROOT = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006"
base = r"D:\DevCache\temp\dsh-ellEc6\gsv\v5\GPT-SoVITS-v5-20261006\GPT_SoVITS\pretrained_models\gsv-v5-pretrained\s2Gv5dev.pth"
out = {"v5dev": {"path": base, "steps": 32, "cfg": 1.3}}
for tag, d, steps in [
    ("ada8_s100", "ada8", [100]), ("ada8_s300", "ada8", [300]),
    ("ada8_s600", "ada8", [600]), ("ada8_s1200", "ada8", [1200]),
    ("all8_s600", "all8", [600]), ("all8_s1200", "all8", [1200]),
    ("r16_s300", "r16", [300]),
]:
    name = f"kan_v5dev_{d}_e1_s{steps[0]}_l{'8' if d in ('ada8','all8') else '16'}.pth"
    out[tag] = {"path": os.path.join(ROOT, f"SoVITS_weights_v5dev_{d}", name), "steps": 32, "cfg": 1.3}
json.dump(out, open(r"D:\杂活\甜蜜女友3\v5_ft_eval\candidates_loader_fix_val.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(out)
