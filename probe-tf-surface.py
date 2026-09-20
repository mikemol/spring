#!/usr/bin/env python3
"""What does the installed transformers actually offer for SPRING's 16 needed symbols?

⚑ WRITTEN AS A FILE, NOT AN INLINE -c: the hook is right that an inline script dies with the turn,
  and this is the measurement the whole port is designed against -- it must be re-runnable.
⚑⚑ THE POPULATION IS THE ENUMERATION FROM THE CHECKOUT, not my memory of what transformers has.
  Each symbol is probed at every plausible location and the result printed, so a reader can see
  WHICH path answered rather than trusting that one did.
"""
import importlib
import sys

print("python     ", sys.version.split()[0])
for m in ("torch", "transformers", "penman", "networkx"):
    try:
        print("%-11s %s" % (m, importlib.import_module(m).__version__))
    except Exception as e:  # noqa: BLE001
        print("%-11s IMPORT FAILED: %s" % (m, e))

print()
print("== the 12 symbols spring_amr/modeling_bart.py takes from the 2020 `modeling_bart` ==")
CANDIDATE_MODULES = [
    "transformers.models.bart.modeling_bart",
    "transformers.models.bart.configuration_bart",
    "transformers.modeling_utils",
    "transformers",
]
BART_SYMS = [
    "BartConfig", "DecoderLayer", "EncoderLayer", "LayerNorm",
    "LearnedPositionalEmbedding", "PretrainedBartModel",
    "SinusoidalPositionalEmbedding", "invert_mask",
    "_filter_out_falsey_values", "_make_linear_from_emb",
    "_prepare_bart_decoder_inputs", "_reorder_buffer",
]
GEN_SYMS = [
    "BeamHypotheses", "calc_banned_ngram_tokens",
    "calc_banned_bad_words_ids", "top_k_top_p_filtering",
]

def where(sym):
    """Return the first module that carries `sym`, or None. Prints the module, not just a boolean."""
    for modname in CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(mod, sym):
            return modname
    return None

missing = []
for sym in BART_SYMS + GEN_SYMS:
    loc = where(sym)
    if loc:
        print("  FOUND    %-32s %s" % (sym, loc))
    else:
        print("  ABSENT   %-32s -- must be VENDORED" % sym)
        missing.append(sym)

print()
print("== the tokenizer question: slow class, and does it still carry encoder/decoder ==")
try:
    from transformers import BartTokenizer
    print("  BartTokenizer imported:", BartTokenizer.__name__)
    print("  is-fast attr        :", getattr(BartTokenizer, "is_fast", "n/a"))
    for attr in ("encoder", "decoder"):
        # ⚑ CLASS-LEVEL hasattr is not the answer -- these are INSTANCE attributes built in
        #   __init__ from the vocab files. Say so rather than printing a misleading False.
        print("  %-20s instance-only (built in __init__ from vocab.json); needs a real load"
              % attr)
except Exception as e:  # noqa: BLE001
    print("  BartTokenizer IMPORT FAILED:", e)

print()
print("== the config fields spring_amr/utils.py sets ==")
try:
    from transformers import BartConfig
    c = BartConfig()
    for f in ("output_past", "prefix", "no_repeat_ngram_size", "output_attentions",
              "dropout", "attention_dropout"):
        print("  %-22s present=%s" % (f, hasattr(c, f)))
    print("  ⚑ an ABSENT field can still be SET without raising -- that is the quiet direction")
except Exception as e:  # noqa: BLE001
    print("  BartConfig FAILED:", e)

print()
print("VENDOR LIST (%d symbol(s)): %s" % (len(missing), ", ".join(missing) or "none"))
