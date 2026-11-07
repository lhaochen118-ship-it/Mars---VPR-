"""Render Markdown files to PDF with pandoc (HTML) + the preinstalled headless Chromium (print-to-pdf).

python3 docs/handoff/md2pdf.py docs/handoff/*.md
"""
import os, subprocess, sys, tempfile

CSS = """body{max-width:none !important;font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.45;margin:0 8mm;color:#111}
h1{font-size:17pt;border-bottom:2px solid #333;padding-bottom:4px}h2{font-size:13.5pt;margin-top:18px;border-bottom:1px solid #bbb}
h3{font-size:11.5pt}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9pt}th,td{border:1px solid #999;padding:3px 5px;vertical-align:top}
th{background:#eee}code{font-family:'DejaVu Sans Mono',monospace;font-size:8.5pt;background:#f4f4f4;padding:0 2px}pre{background:#f4f4f4;padding:6px;white-space:pre-wrap;font-size:8pt}"""


CHROME = os.environ.get("CHROME", "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome")


def main():
    for md in sys.argv[1:]:
        html = subprocess.run(["pandoc", md, "-f", "gfm", "-t", "html5", "-s", "--metadata", f"pagetitle={os.path.basename(md)}"],
                              capture_output=True, text=True, check=True).stdout
        html = html.replace("</head>", f"<style>{CSS}</style></head>")
        # relative image links keep working: write the HTML next to the Markdown file
        tmp = os.path.join(os.path.dirname(os.path.abspath(md)), ".md2pdf_tmp.html")
        open(tmp, "w").write(html)
        out = os.path.splitext(md)[0] + ".pdf"
        try:
            subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={os.path.abspath(out)}", f"file://{tmp}"], check=True, capture_output=True, timeout=180)
        finally:
            os.remove(tmp)
        print("wrote", out)


if __name__ == "__main__":
    main()
