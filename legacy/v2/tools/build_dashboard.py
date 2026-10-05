"""Assemble site/index.html from site/phrases.json + the template.

The dashboard must open from file:// with zero dependencies, so the JSON is
inlined into a <script type="application/json"> block. Re-run after
mine_phrases.py (or any hand edit of phrases.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def main():
    data = json.loads((SITE / "phrases.json").read_text(encoding="utf-8"))
    tpl = (SITE / "index_template.html").read_text(encoding="utf-8")
    if "__DATA__" not in tpl:
        sys.exit("template missing __DATA__ placeholder")
    html = tpl.replace("__DATA__",
                       json.dumps(data).replace("</", "<\\/"))
    (SITE / "index.html").write_text(html, encoding="utf-8")
    size_kb = (SITE / "index.html").stat().st_size / 1024
    print(f"wrote site/index.html ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()