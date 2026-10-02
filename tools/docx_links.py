"""Extract paragraph text + hyperlink URLs from a .docx.

Usage: python tools/docx_links.py <file.docx>
Prints each non-empty paragraph; hyperlinks are inlined as [LINK url].
"""
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def main() -> None:
    z = zipfile.ZipFile(sys.argv[1])
    rels: dict[str, str] = {}
    if "word/_rels/document.xml.rels" in z.namelist():
        root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        for rel in root:
            rels[rel.get("Id", "")] = rel.get("Target", "")
    doc = ET.fromstring(z.read("word/document.xml"))
    for p in doc.iter(f"{{{W}}}p"):
        parts: list[str] = []
        for el in p.iter():
            if el.tag == f"{{{W}}}t":
                parts.append(el.text or "")
            elif el.tag == f"{{{W}}}hyperlink":
                rid = el.get(f"{{{R}}}id", "")
                if rid in rels:
                    parts.append(f" [LINK {rels[rid]}]")
        line = "".join(parts).strip()
        if line:
            print(line)


if __name__ == "__main__":
    main()
