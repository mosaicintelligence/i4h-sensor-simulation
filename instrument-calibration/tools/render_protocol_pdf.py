# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render docs/ivus_calibration_protocol.md to a PDF.

Pipeline: Markdown -> HTML (python-markdown w/ tables, fenced_code, toc, codehilite,
pymdownx.tilde, pymdownx.tasklist, pymdownx.magiclink) -> PDF (xhtml2pdf).

Run:
    PYTHONPATH=/tmp/calpkgs python3 tools/render_protocol_pdf.py
"""
from __future__ import annotations

import os
from pathlib import Path

import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "ivus_calibration_protocol.md"
OUT = ROOT / "docs" / "ivus_calibration_protocol.pdf"

UNICODE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
UNICODE_MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _register_fonts() -> tuple[str, str]:
    """Register a Unicode-capable serif and monospace font with reportlab.

    Returns (sans_face_name, mono_face_name).
    """
    sans_path = next((p for p in UNICODE_FONT_CANDIDATES if Path(p).exists()),
                     None)
    if sans_path is None:
        return "Helvetica", "Courier"
    sans_name = "Body"
    pdfmetrics.registerFont(TTFont(sans_name, sans_path))

    mono_path = next((p for p in UNICODE_MONO_CANDIDATES if Path(p).exists()),
                     None)
    if mono_path is None or mono_path.endswith(".ttc"):
        mono_name = sans_name
    else:
        mono_name = "Mono"
        pdfmetrics.registerFont(TTFont(mono_name, mono_path))
    return sans_name, mono_name

CSS_TEMPLATE = """
@page {
    size: letter;
    margin: 1.6cm 1.6cm 2.0cm 1.6cm;
    @frame footer {
        -pdf-frame-content: footer_content;
        bottom: 0.7cm;
        left: 1.6cm;
        right: 1.6cm;
        height: 0.7cm;
    }
}
body {
    font-family: "{sans}";
    font-size: 9.5pt;
    line-height: 1.42;
    color: #1a1a1a;
}
h1 {
    font-size: 19pt;
    color: #0b3d6b;
    border-bottom: 2px solid #0b3d6b;
    padding-bottom: 4pt;
    margin-top: 14pt;
    margin-bottom: 10pt;
    -pdf-keep-with-next: true;
}
h2 {
    font-size: 14pt;
    color: #0b3d6b;
    border-bottom: 1px solid #b0c4d8;
    padding-bottom: 2pt;
    margin-top: 16pt;
    margin-bottom: 7pt;
    -pdf-keep-with-next: true;
    page-break-before: always;
}
h2:first-of-type, h2.no-break {
    page-break-before: auto;
}
h3 {
    font-size: 11.5pt;
    color: #244c75;
    margin-top: 12pt;
    margin-bottom: 5pt;
    -pdf-keep-with-next: true;
}
h4 {
    font-size: 10.5pt;
    color: #244c75;
    margin-top: 9pt;
    margin-bottom: 4pt;
    -pdf-keep-with-next: true;
}
p { margin: 0 0 6pt 0; text-align: left; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
blockquote {
    margin: 6pt 0 6pt 0;
    padding: 6pt 9pt;
    background: #f4f7fb;
    border-left: 3px solid #244c75;
    color: #1a1a1a;
    font-size: 9pt;
}
code {
    font-family: "{mono}";
    font-size: 8.5pt;
    background: #f0f1f3;
    padding: 0 2pt;
    color: #4a235a;
}
pre {
    font-family: "{mono}";
    font-size: 8.0pt;
    background: #f0f1f3;
    border: 0.5pt solid #d0d4da;
    padding: 6pt;
    margin: 4pt 0 8pt 0;
    -pdf-keep-in-frame-mode: shrink;
}
pre code { background: transparent; padding: 0; color: #1a1a1a; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 4pt 0 8pt 0;
    font-size: 8.5pt;
    -pdf-keep-in-frame-mode: shrink;
}
th, td {
    border: 0.5pt solid #b0b6bf;
    padding: 3pt 5pt;
    vertical-align: top;
    text-align: left;
}
th {
    background: #e6edf5;
    color: #0b3d6b;
    font-weight: bold;
}
tr:nth-child(even) td { background: #fafbfc; }
img {
    max-width: 100%;
    -pdf-keep-with-previous: true;
}
.figure {
    margin: 6pt 0 10pt 0;
    page-break-inside: avoid;
}
hr { border: 0; border-top: 0.5pt solid #b0b6bf; margin: 8pt 0; }
a { color: #0b5394; text-decoration: none; }
.title-block {
    text-align: center;
    margin-bottom: 14pt;
}
.subtitle {
    color: #555;
    font-size: 11pt;
    margin-top: -4pt;
}
.footer {
    font-size: 8pt;
    color: #888;
    text-align: center;
}
"""

FOOTER_HTML = (
    '<div id="footer_content" class="footer">'
    'IVUS Calibration & Characterization Protocol — Volcano s5i &nbsp;·&nbsp; '
    'page <pdf:pagenumber> of <pdf:pagecount>'
    '</div>'
)


def _link_callback(uri: str, rel: str) -> str:
    """Resolve `<img src="...">` paths to absolute filesystem paths for xhtml2pdf."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    if uri.startswith("file://"):
        return uri[7:]
    base = (ROOT / "docs").resolve()
    candidate = (base / uri).resolve()
    if candidate.exists():
        return str(candidate)
    return uri


PRE_MD_SUBSTITUTIONS = {
    "⌀": "Ø",
}

POST_MD_SUBSTITUTIONS = {
    "₀": "<sub>0</sub>",
    "₁": "<sub>1</sub>",
    "₂": "<sub>2</sub>",
    "₃": "<sub>3</sub>",
    "₄": "<sub>4</sub>",
    "₅": "<sub>5</sub>",
    "₆": "<sub>6</sub>",
    "₇": "<sub>7</sub>",
    "₈": "<sub>8</sub>",
    "₉": "<sub>9</sub>",
    "ₙ": "<sub>n</sub>",
    "²": "<sup>2</sup>",
    "³": "<sup>3</sup>",
    "⁻¹": "<sup>-1</sup>",
    "⁻²": "<sup>-2</sup>",
    "⁰": "<sup>0</sup>",
    "ⁿ": "<sup>n</sup>",
}


def _substitute(text: str, table: dict[str, str]) -> str:
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


def main() -> None:
    sans, mono = _register_fonts()
    print(f"  using fonts: sans={sans!r}, mono={mono!r}")
    css = CSS_TEMPLATE.replace("{sans}", sans).replace("{mono}", mono)
    md_text = SRC.read_text(encoding="utf-8")
    md_text = _substitute(md_text, PRE_MD_SUBSTITUTIONS)
    md = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "attr_list",
            "admonition",
            "md_in_html",
            "pymdownx.tilde",
            "pymdownx.tasklist",
        ],
        extension_configs={
            "toc": {"permalink": False, "toc_depth": "2-3"},
        },
    )
    body_html = md.convert(md_text)
    body_html = _substitute(body_html, POST_MD_SUBSTITUTIONS)

    full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>
{FOOTER_HTML}
<div class="title-block">
  <h1 class="no-break">IVUS Calibration &amp; Characterization Protocol</h1>
  <div class="subtitle">Volcano s5i / Eagle Eye Gold &nbsp;·&nbsp; bench protocol for E1–E9</div>
</div>
{body_html}
</body></html>"""

    out_path = OUT
    with open(out_path, "wb") as fp:
        result = pisa.CreatePDF(full, dest=fp, encoding="utf-8",
                                link_callback=_link_callback)
    if result.err:
        raise SystemExit(f"xhtml2pdf reported {result.err} error(s)")
    print(f"  wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
