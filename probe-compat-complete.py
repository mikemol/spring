#!/usr/bin/env python3
"""Does bart_compat supply EVERY symbol spring_amr/modeling_bart.py takes from `bart`?

⚑⚑⚑ THE POPULATION IS DERIVED FROM THE SOURCE, NOT FROM MY LIST. I wrote the compat layer against a
  hand-copied set of 14 names; re-deriving the requirement from the file that consumes it is what
  catches a name I mistyped or omitted. A COMPAT LAYER VERIFIED AGAINST THE LIST I USED TO WRITE IT
  IS A SELF-AGREEING CHECK.
⚑⚑ IT ALSO CHECKS THE modeling_utils IMPORT LINE SEPARATELY, because those four arrive from a
  different 2020 module and my earlier import-grep treated them as one population with the rest.
⚑ REFUSES (rc=2) if it cannot import the compat layer at all -- a probe that cannot reach its
  subject must not report zero missing.
"""
import re
import sys
from pathlib import Path

SPRING = Path("/home/mikemol/github/aeternum/spring-amr")
SRC = SPRING / "spring_amr/modeling_bart.py"

if not SRC.is_file():
    sys.exit("probe-compat: %s absent -- UNEVALUABLE" % SRC)

text = SRC.read_text(encoding="utf-8")

# every `bart.<Name>` the consumer actually uses
used_bart = sorted(set(re.findall(r"\bbart\.([A-Za-z_][A-Za-z0-9_]*)", text)))

# The generation helpers, WHEREVER the consumer imports them from.
# ⚑⚑⚑ THIS WAS ANCHORED ON `transformers.modeling_utils` AND THE PORT MOVED THE LINE. Once the import
#   was rewritten to point at bart_compat, the probe reported 0 from modeling_utils and reclassified
#   those four helpers as exported-but-unrequired -- TRUE of the string it matched, FALSE about what
#   the consumer needs. A PROBE ANCHORED ON THE PRE-FIX SHAPE STOPS MEASURING THE MOMENT THE FIX
#   LANDS, which is exactly when its answer starts being quoted.
# ⚑ So match either home and PRINT WHICH ONE ANSWERED, rather than reporting a count whose subject
#   the reader cannot see.
mu = re.search(
    r"from\s+(transformers\.modeling_utils|spring_amr\.bart_compat)\s+import\s+"
    r"([^\n]*(?:\\\n[^\n]*)*)",
    text,
)
used_mu = []
mu_home = "NONE FOUND"
if mu:
    mu_home = mu.group(1)
    used_mu = sorted({s.strip() for s in mu.group(2).replace("\\", " ").split(",") if s.strip()})

print("derived from %s:" % SRC.name)
print("  bart.*            %d symbol(s)" % len(used_bart))
print("  gen helpers       %d symbol(s)  (imported from: %s)" % (len(used_mu), mu_home))
print()

sys.path.insert(0, str(SPRING))
try:
    from spring_amr import bart_compat as compat
except Exception as e:  # noqa: BLE001
    sys.exit("probe-compat: cannot import bart_compat (%s: %s) -- REFUSING to report a count"
             % (type(e).__name__, e))

missing = []
for name in used_bart + used_mu:
    ok = hasattr(compat, name)
    print("  %-34s %s" % (name, "ok" if ok else "MISSING"))
    if not ok:
        missing.append(name)

print()
extra = sorted(set(compat.__all__) - set(used_bart) - set(used_mu))
if extra:
    print("exported but NOT required by the consumer (fine, but say so): %s" % ", ".join(extra))
print()
if missing:
    print("INCOMPLETE: %d symbol(s) missing -- %s" % (len(missing), ", ".join(missing)))
    sys.exit(1)
print("COMPLETE: bart_compat supplies every symbol the consumer imports (%d)"
      % (len(used_bart) + len(used_mu)))
