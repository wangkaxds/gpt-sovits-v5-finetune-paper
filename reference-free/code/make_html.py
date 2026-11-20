import os
from pathlib import Path

import markdown

BASE = Path(__file__).resolve().parent.parent
src = (BASE / "paper.md").read_text(encoding="utf-8")
body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc", "sane_lists"])
css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Microsoft YaHei", "Noto Sans SC", "Segoe UI", sans-serif;
       font-size: 10.5pt; line-height: 1.65; color: #222; max-width: 900px; margin: 0 auto; }
h1 { font-size: 21pt; line-height: 1.3; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 14pt; margin-top: 22px; border-left: 4px solid #2e75b6; padding-left: 8px; }
h3 { font-size: 12pt; margin-top: 16px; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 9.0pt;
        table-layout: fixed; }
thead { display: table-header-group; }
tbody tr { break-inside: avoid; }
p { orphans: 3; widows: 3; }
th, td { border: 1px solid #bbb; padding: 5px 6px; text-align: left;
         vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }
th { background: #eef2f7; }
code { background: #f2f2f2; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f6f6; padding: 9px; overflow-x: auto; font-size: 8.6pt; }
img { max-width: 100%; margin: 8px 0; }
blockquote { border-left: 3px solid #999; margin: 8px 0; padding-left: 10px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
@media print {
  h2, h3 { break-after: avoid; }
  tr, img, pre, li { break-inside: avoid; }
  table { font-size: 8.4pt; }
}
"""
html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>GPT-SoVITS V5 的严格无参考单说话人推理：条件通路审计与系统性负结果</title>
<style>{css}</style></head><body>
{body}
</body></html>"""
(BASE / "paper.html").write_text(html, encoding="utf-8")
print("html written", BASE / "paper.html")
