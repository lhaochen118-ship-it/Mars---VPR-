"""Render Markdown files to PDF with pandoc (HTML) + the preinstalled Chromium (Playwright).

python3 docs/handoff/md2pdf.py docs/handoff/*.md
"""
import os, subprocess, sys, tempfile

CSS = """body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.45;margin:0 8mm;color:#111}
h1{font-size:17pt;border-bottom:2px solid #333;padding-bottom:4px}h2{font-size:13.5pt;margin-top:18px;border-bottom:1px solid #bbb}
h3{font-size:11.5pt}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9pt}th,td{border:1px solid #999;padding:3px 5px;vertical-align:top}
th{background:#eee}code{font-family:'DejaVu Sans Mono',monospace;font-size:8.5pt;background:#f4f4f4;padding:0 2px}pre{background:#f4f4f4;padding:6px;white-space:pre-wrap}"""


def main():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        for md in sys.argv[1:]:
            html = subprocess.run(["pandoc", md, "-f", "gfm", "-t", "html5", "-s", "--metadata", f"title={os.path.basename(md)}"],
                                  capture_output=True, text=True, check=True).stdout
            html = html.replace("</head>", f"<style>{CSS}</style></head>")
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
                f.write(html)
            pg = b.new_page()
            pg.goto(f"file://{f.name}")
            out = os.path.splitext(md)[0] + ".pdf"
            pg.pdf(path=out, format="A4", margin={"top": "12mm", "bottom": "12mm", "left": "8mm", "right": "8mm"}, print_background=True)
            pg.close()
            print("wrote", out)
        b.close()


if __name__ == "__main__":
    main()
