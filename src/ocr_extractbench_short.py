"""OCR the ExtractBench short-split PDFs that pypdf could not read.

69 of 252 short PDFs are scanned images with no text layer, so
load_extractbench skips them (0 chars extracted). This script renders
those PDFs with pdftoppm (200 DPI) and OCRs them with tesseract,
writing the text into the same .txt cache load_extractbench reads
(data/extractbench/docs/short/<name>.txt), so re-running the dry-run
picks them up with no code change.

Usage: .venv/bin/python src/ocr_extractbench_short.py [--thin]
  Default: OCR the .txt files that are empty (0 chars).
  --thin: also OCR files with <300 chars per PDF page — scanned-content
      PDFs whose text layer holds only headers/stamps (42 docs in the
      short split; 37 of the 38 docs flash returned empty for are thin).
      Safe to run while model runs are in flight: those processes already
      loaded their text into memory and never re-read the .txt cache.
Writes progress to stdout (redirect to a log file; the run takes
several minutes for ~69 docs).
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import pypdf

SHORT_DIR = os.path.join("data", "extractbench", "docs", "short")


def select(thin):
    targets = []
    for txt in sorted(glob.glob(os.path.join(SHORT_DIR, "*.txt"))):
        nchars = os.path.getsize(txt)
        if nchars == 0:
            targets.append(txt)
            continue
        if not thin:
            continue
        pdf = txt[:-4] + ".pdf"
        try:
            npages = len(pypdf.PdfReader(pdf).pages)
        except Exception:
            npages = 1
        if nchars < 300 * max(npages, 1):
            targets.append(txt)
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", action="store_true")
    args = ap.parse_args()
    targets = select(args.thin)
    print(f".txt files to OCR: {len(targets)} (thin mode: {args.thin})", flush=True)
    done = 0
    failed = []
    for txt in targets:
        pdf = txt[:-4] + ".pdf"
        name = os.path.basename(pdf)
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                ["pdftoppm", "-r", "200", "-png", pdf, os.path.join(td, "p")],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{name}: pdftoppm FAILED: {r.stderr.strip()[:200]}", flush=True)
                failed.append(name)
                continue
            pages = sorted(glob.glob(os.path.join(td, "p-*.png")))
            chunks = []
            for pg in pages:
                t = subprocess.run(
                    ["tesseract", pg, "stdout", "--psm", "6"],
                    capture_output=True, text=True)
                if t.returncode == 0:
                    chunks.append(t.stdout)
            text = "\n".join(chunks)
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(text)
        done += 1
        print(f"[{done}/{len(targets)}] {name}: {len(text)} chars from {len(pages)} pages", flush=True)
    print(f"OCR done: {done} docs, {len(failed)} failed: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
