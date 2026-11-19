import os
import markdown

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(BASE, "paper.md"), encoding="utf-8").read()
body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc", "sane_lists"])
css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Microsoft YaHei", "Noto Sans SC", "Segoe UI", sans-serif;
       font-size: 10.5pt; line-height: 1.65; color: #222; max-width: 900px; margin: 0 auto; }
h1 { font-size: 21pt; line-height: 1.3; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 14pt; margin-top: 22px; border-left: 4px solid #2e75b6; padding-left: 8px; }
h3 { font-size: 12pt; margin-top: 16px; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 9.2pt; }
th, td { border: 1px solid #bbb; padding: 5px 6px; text-align: left; }
th { background: #eef2f7; }
code { background: #f2f2f2; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f6f6; padding: 9px; overflow-x: auto; font-size: 8.6pt; }
img { max-width: 100%; margin: 8px 0; }
blockquote { border-left: 3px solid #999; margin: 8px 0; padding-left: 10px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
"""
html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>GPT-SoVITS V5 微调的系统性负结果</title>
<style>{css}</style></head><body>
{body}
</body></html>"""
open(os.path.join(BASE, "paper.html"), "w", encoding="utf-8").write(html)
print("html written")
