#!/usr/bin/env python3
"""Fetch SPRING's two weight artifacts and PIN THE DIGEST OF WHAT ARRIVED.

⚑⚑⚑ THIS CANNOT VERIFY PROVENANCE AND DOES NOT CLAIM TO. The README publishes the checkpoint over
  PLAIN HTTP with no digest of any kind, so nothing here distinguishes the authors' bytes from a
  substitution in flight. What this DOES buy: the sha256 of what arrived is recorded in-tree, so a
  LATER substitution — a re-fetch that differs, a file swapped on disk — is detectable. First-fetch
  trust made auditable is strictly better than first-fetch trust left unrecorded, and it is NOT the
  same thing as verified.
⚑⚑ THE BASE MODEL IS THE ASYMMETRIC CASE: huggingface serves facebook/bart-large over HTTPS and the
  hub records per-blob hashes, so that half IS verifiable against an upstream authority. Only the
  checkpoint is first-fetch trust. Say which is which rather than reporting one confidence.
⚑ WRITES THE PIN FILE ONLY ON A COMPLETE FETCH. A partial download whose digest got recorded would
  pin the truncation, and every later comparison would agree with it.
"""
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

SPRING = Path(__file__).resolve().parent
PINS = SPRING / "weights-pins.json"
CKPT_URL = "http://nlp.uniroma1.it/AMR/AMR3.parsing-1.0.tar.bz2"
CKPT_DEST = SPRING / "weights" / "AMR3.parsing-1.0.tar.bz2"

# ⚑ A FLOOR ON THE SIZE, because an HTTP error page is a successful 200 with a body. A few-KB
#   "not found" would otherwise be hashed and pinned as the checkpoint.
CKPT_MIN_BYTES = 500 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def free_bytes() -> int:
    return shutil.disk_usage(SPRING).free


def fetch_checkpoint() -> dict:
    CKPT_DEST.parent.mkdir(parents=True, exist_ok=True)
    if CKPT_DEST.is_file() and CKPT_DEST.stat().st_size >= CKPT_MIN_BYTES:
        print(f"checkpoint already present: {CKPT_DEST} ({CKPT_DEST.stat().st_size} bytes)")
    else:
        need = 2 * 1024 * 1024 * 1024
        if free_bytes() < need:
            sys.exit(f"REFUSING: {free_bytes() // (1024**3)} GiB free, want >= 2 GiB headroom")
        print(f"fetching {CKPT_URL}")
        tmp = CKPT_DEST.with_suffix(".partial")
        with urllib.request.urlopen(CKPT_URL, timeout=120) as r, tmp.open("wb") as out:
            shutil.copyfileobj(r, out, length=1024 * 1024)
        size = tmp.stat().st_size
        if size < CKPT_MIN_BYTES:
            tmp.unlink()
            sys.exit(f"REFUSING: got {size} bytes, below the {CKPT_MIN_BYTES} floor — "
                     f"that is an error page or a truncation, not a checkpoint")
        tmp.rename(CKPT_DEST)
    return {
        "url": CKPT_URL,
        "path": str(CKPT_DEST.relative_to(SPRING)),
        "bytes": CKPT_DEST.stat().st_size,
        "sha256": sha256(CKPT_DEST),
        "provenance": "FIRST-FETCH TRUST — plain HTTP, no upstream digest published. "
                      "This pin detects a later change; it does not verify the origin.",
    }


def fetch_base() -> dict:
    from huggingface_hub import snapshot_download
    print("fetching facebook/bart-large weights (HTTPS, hub-hashed)")
    local = snapshot_download(
        "facebook/bart-large",
        allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "*.safetensors"],
    )
    root = Path(local)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    # ⚑⚑⚑ A FILTERED DOWNLOAD THAT MATCHES NOTHING RETURNS SUCCESSFULLY WITH AN EMPTY SNAPSHOT, and
    #   the first run of this script did exactly that: it returned, the pin file would have recorded
    #   the tokenizer files as the model, and nothing raised. MEASURED: the hub serves
    #   `pytorch_model.bin`, so the pattern was never the fault -- a shadowed stdlib import in the
    #   CWD broke huggingface_hub silently enough that the download became a no-op. The pattern being
    #   correct is exactly why this needs an assertion rather than a better pattern: THE PROBE MUST
    #   REQUIRE ITS SUBJECT TO HAVE ARRIVED, not trust that a call which returned did its job.
    weights = {n: m for n, m in files.items() if m["bytes"] > 100 * 1024 * 1024}
    if not weights:
        sys.exit(f"REFUSING: snapshot_download returned but no file over 100 MiB landed in {root} "
                 f"({len(files)} file(s) present) -- an empty snapshot is not a fetched model")
    return {
        "repo": "facebook/bart-large",
        "local": str(root),
        "files": files,
        "provenance": "HTTPS + hub per-blob hashes — verifiable against an upstream authority.",
    }


if __name__ == "__main__":
    base = fetch_base()
    # ⚑⚑⚑ THE CHECKPOINT'S UPSTREAM IS DOWN AND THAT IS RECORDED AS A MEASUREMENT, NOT A CRASH.
    #   Measured 2026-09-19 from three instruments: DNS resolves (151.100.59.110), port 80 REFUSES
    #   in 261 ms, port 443 RESETS -- two different errors on one host, so the box is up and
    #   filtering rather than absent. Controls separate it from my own egress: huggingface.co -> 200
    #   and www.uniroma1.it -> 301 in the same minute, so the network is fine and the NLP group's
    #   host specifically is not. ⚑⚑ THE PROVENANCE RISK ARRIVED AS UNAVAILABILITY RATHER THAN
    #   SUBSTITUTION: five years after publication the paper's only published download host is gone,
    #   which is the same single-point-of-trust defect showing its other face.
    # ⚑ SO THE BASE-MODEL PIN IS STILL WRITTEN. A run that refused to record the half that DID
    #   succeed would make the reachable half look unmeasured too.
    try:
        ckpt = fetch_checkpoint()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        ckpt = {"url": CKPT_URL, "status": "UNREACHABLE",
                "error": f"{type(e).__name__}: {e}",
                "provenance": "NOT FETCHED — upstream host does not answer. No substitute may be "
                              "pinned here without saying where it came from."}
        print(f"\ncheckpoint UNREACHABLE: {type(e).__name__}: {e}", file=sys.stderr)
    pins = {"base_model": base, "checkpoint": ckpt}
    PINS.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {PINS}")
    print(f"  checkpoint        : {pins['checkpoint'].get('sha256', pins['checkpoint'].get('status'))}")
    print(f"  base model files  : {len(pins['base_model']['files'])}")
