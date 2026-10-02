"""Extract text (paragraphs + tables) from a .docx file.

Usage: python tools/read_docx.py <input.docx> [output.txt]
Prints the document text to stdout; optionally saves it to output.txt.
"""
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


def para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(q("t")))


def walk(el: ET.Element, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for child in el:
        if child.tag == q("p"):
            txt = para_text(child)
            if txt.strip():
                lines.append("  " * depth + txt)
        elif child.tag == q("tbl"):
            for tr in child.findall(q("tr")):
                cells = []
                for tc in tr.findall(q("tc")):
                    cell_txt = " ".join(
                        para_text(p) for p in tc.iter(q("p"))
                    ).strip()
                    cells.append(cell_txt)
                lines.append("  " * depth + " | ".join(cells))
        elif child.tag == q("sdt"):
            lines.extend(walk(child, depth))
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: read_docx.py <input.docx> [output.txt]")
        return 1
    src = Path(sys.argv[1])
    z = zipfile.ZipFile(src)
    root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(q("body"))
    lines = walk(body) if body is not None else []
    text = "\n".join(lines)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")
        print(f"\n[saved -> {sys.argv[2]}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
