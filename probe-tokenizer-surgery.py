#!/usr/bin/env python3
"""Does SPRING's in-place vocabulary surgery still work on transformers 5.17's BartTokenizer?

⚑⚑⚑ THE RISK IS SILENT, WHICH IS WHY THIS IS A PROBE AND NOT AN ASSUMPTION. `init_amr_vocabulary`
  DELETES keys from `self.encoder`, reindexes it, rebuilds `self.decoder`, and reassigns
  bos/pad/eos/unk to `Ġ`-prefixed variants. Those are the SLOW tokenizer's instance attributes.
  transformers 5.17 resolves `BartTokenizer` to `RobertaTokenizer` -- if that class no longer keeps
  `encoder`/`decoder`, or keeps them as a property, the surgery either raises or writes into
  something nothing reads, and THE MODEL THEN GETS THE WRONG EMBEDDING TABLE while still producing
  output.
⚑⚑ NO NETWORK ASSUMPTION: this needs the real bart-large vocab to load. If the model is not cached
  locally it will try to fetch; the probe says so rather than hanging silently, and a failure here is
  UNEVALUABLE, not a pass.
⚑ Checks the properties that decide correctness, in order: class identity, attribute presence,
  attribute MUTABILITY, and finally that ids stay consistent after the surgery.
"""
import sys
from pathlib import Path

SPRING = Path("/home/mikemol/github/aeternum/spring-amr")
sys.path.insert(0, str(SPRING))

from transformers import BartTokenizer

print("BartTokenizer resolves to:", BartTokenizer.__name__)
print("module                   :", BartTokenizer.__module__)

# ⚑ Class-level presence is not the question; these are built in __init__. But the CLASS tells us
#   whether they are properties (unsettable) versus plain attributes.
for attr in ("encoder", "decoder"):
    desc = getattr(BartTokenizer, attr, None)
    print("  class-level %-8s -> %s" % (attr, type(desc).__name__ if desc is not None else "absent (instance attr)"))

print()
print("loading facebook/bart-large tokenizer (local cache if present)...")
try:
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
except Exception as e:  # noqa: BLE001
    sys.exit("UNEVALUABLE: could not load the tokenizer (%s: %s). Not a pass."
             % (type(e).__name__, e))

print("  loaded. vocab size:", len(tok))
have_enc = hasattr(tok, "encoder")
have_dec = hasattr(tok, "decoder")
print("  instance encoder:", have_enc, " decoder:", have_dec)

if not (have_enc and have_dec):
    # ⚑⚑⚑ THIS IS A FACT ABOUT THE BASE CLASS, NOT A VERDICT ON THE PORT, AND MY FIRST VERSION
    #   CONFLATED THEM. It exited here, so it never reached AMRBartTokenizer -- the very subject the
    #   port changes. The base class LACKING `encoder` is exactly WHY the subclass now supplies it as
    #   a property over `_vocab`, so this condition is expected-and-handled, not incompatible.
    # ⚑ A GATE ON THE WRONG SUBJECT REPORTS A TRUE FACT AS A FAILURE OF SOMETHING ELSE.
    print("  NOTE: the BASE class lacks these (expected on 5.x) -- the subclass must supply them;")
    print("        continuing to the real subject, AMRBartTokenizer.")
else:
    enc = tok.encoder
    print("  type(encoder):", type(enc).__name__, " entries:", len(enc))
    probe_key = "Ġ__spring_probe__"
    try:
        enc[probe_key] = len(enc)
        del enc[probe_key]
        print("  base encoder is MUTABLE (set + del round-tripped)")
    except Exception as e:  # noqa: BLE001
        print("  base encoder not mutable (%s) -- the subclass property must own this"
              % type(e).__name__)

print()
print("now the real surgery, via SPRING's own subclass:")
try:
    from spring_amr.tokenization_bart import AMRBartTokenizer

    amr = AMRBartTokenizer.from_pretrained("facebook/bart-large")
except Exception as e:  # noqa: BLE001
    sys.exit("SURGERY FAILED: %s: %s" % (type(e).__name__, e))

print("  AMRBartTokenizer loaded.")
print("  old_enc_size      :", getattr(amr, "old_enc_size", "MISSING"))
print("  len(encoder)      :", len(amr.encoder))
print("  bos/pad/eos/unk   :", repr(amr.bos_token), repr(amr.pad_token),
      repr(amr.eos_token), repr(amr.unk_token))

# ⚑ THE INVARIANT THE MODEL DEPENDS ON: encoder and decoder must be exact inverses, and the id space
#   must be contiguous from 0. A reindex that leaves a gap gives resize_token_embeddings the wrong
#   size and silently mis-maps every token above the gap.
inv_ok = all(amr.decoder[i] == t for t, i in amr.encoder.items())
ids = sorted(amr.encoder.values())
contiguous = ids == list(range(len(ids)))
print("  encoder/decoder are exact inverses :", inv_ok)
print("  ids contiguous from 0              :", contiguous)

amr_toks = [t for t in amr.encoder if t.startswith(":") or t.startswith("Ġ:")]
print("  AMR relation tokens present        :", len(amr_toks))
print("  ':snt' family present              :", any("snt" in t for t in amr_toks))

if not (inv_ok and contiguous):
    sys.exit("SURGERY PRODUCED AN INCONSISTENT VOCABULARY -- would mis-map embeddings")
print()
print("TOKENIZER OK: surgery applied, vocabulary consistent, AMR tokens installed")
