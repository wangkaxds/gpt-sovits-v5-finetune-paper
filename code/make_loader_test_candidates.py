import json
src = r"D:\杂活\甜蜜女友3\v5_ft_eval\candidates_loader_fix_val.json"
d = json.load(open(src, encoding="utf-8"))
keep = ["v5dev", "ada8_s300", "all8_s1200"]
out = {k: d[k] for k in keep}
json.dump(out, open(r"D:\杂活\甜蜜女友3\v5_ft_eval\candidates_loader_fix_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(out)
