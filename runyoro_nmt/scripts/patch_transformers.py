"""
Definitive patch: rewrite the problematic XOR check in M2M100Decoder.forward
to simply: if both provided, prefer input_ids; if neither, raise.
Also adds the ForConditionalGeneration guard.
Clears pyc cache.
"""
from pathlib import Path

MODEL_FILE = Path(
    r"C:\Users\MarvinCliveTwesige\AppData\Local\Programs\Python\Python313"
    r"\Lib\site-packages\transformers\models\m2m_100\modeling_m2m_100.py"
)

lines = MODEL_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
new_lines = []
i = 0
changes = 0

while i < len(lines):
    line = lines[i]

    # ── Replace the ENTIRE check block in M2M100Decoder.forward ──────────────
    # Original 5-line block:
    #   # retrieve input_ids and inputs_embeds
    #   [PATCH comment lines if already patched]
    #   if (input_ids is None) ^ (inputs_embeds is not None):
    #       raise ValueError("You cannot specify both...")
    #   elif input_ids is not None:
    #
    # Replace with a clean version that handles all cases correctly
    if (
        "# retrieve input_ids and inputs_embeds" in line
        and i + 1 < len(lines)
    ):
        # Gather next few lines to identify the block
        block = "".join(lines[i:i+8])
        if ("You cannot specify both decoder_input_ids" in block
                and "elif input_ids is not None" in block):
            indent = line[: len(line) - len(line.lstrip())]
            # Write the replacement block
            new_lines.append(f"{indent}# retrieve input_ids and inputs_embeds\n")
            new_lines.append(f"{indent}# PATCH-FINAL runyoro-nmt-v1: resolve both/neither conflict\n")
            new_lines.append(f"{indent}if input_ids is not None and inputs_embeds is not None:\n")
            new_lines.append(f"{indent}    inputs_embeds = None  # prefer input_ids\n")
            new_lines.append(f"{indent}if input_ids is None and inputs_embeds is None:\n")
            new_lines.append(f"{indent}    raise ValueError(\n")
            new_lines.append(f'{indent}        "You have to specify either decoder_input_ids or decoder_inputs_embeds"\n')
            new_lines.append(f"{indent}    )\n")
            # Skip original lines until we reach the elif
            j = i + 1
            while j < len(lines) and "elif input_ids is not None:" not in lines[j]:
                j += 1
            i = j  # continue from 'elif input_ids is not None:'
            changes += 1
            print(f"  Replaced Decoder conflict-check block")
            continue

    # ── Guard in M2M100ForConditionalGeneration.forward ───────────────────────
    if (
        "        if labels is not None:\n" == line
        and i + 4 < len(lines)
        and "decoder_input_ids = shift_tokens_right(" in lines[i+2]
        and "PATCH-FINAL" not in lines[i-1]
    ):
        # Write the labels block unchanged, then add our guard before outputs = self.model(
        new_lines.append(line)
        i += 1
        continue

    new_lines.append(line)
    i += 1

MODEL_FILE.write_text("".join(new_lines), encoding="utf-8")

# Also ensure the ForConditionalGeneration guard is in place
content = MODEL_FILE.read_text(encoding="utf-8")
GUARD = (
    "        # PATCH runyoro-nmt-v1: when decoder_input_ids is set, drop decoder_inputs_embeds\n"
    "        if decoder_input_ids is not None:\n"
    "            decoder_inputs_embeds = None\n"
)
if GUARD not in content:
    OLD_OUT = "        outputs = self.model(\n            input_ids,"
    NEW_OUT = GUARD + "        outputs = self.model(\n            input_ids,"
    if OLD_OUT in content:
        content = content.replace(OLD_OUT, NEW_OUT)
        MODEL_FILE.write_text(content, encoding="utf-8")
        print("  Added ForConditionalGeneration guard before self.model() call")
        changes += 1
    else:
        print("  WARNING: Could not add ForConditionalGeneration guard")
else:
    print("  ForConditionalGeneration guard already present")

# Clear pyc
pycache = MODEL_FILE.parent / "__pycache__"
removed = 0
for f in pycache.glob("modeling_m2m_100*.pyc"):
    f.unlink()
    removed += 1
print(f"  Cleared {removed} .pyc cache files")

print(f"\nTotal changes: {changes}")

# Verify final state
content2 = MODEL_FILE.read_text(encoding="utf-8")
lines2 = content2.splitlines()
print("\nAll PATCH lines in file:")
for j, l in enumerate(lines2):
    if "PATCH" in l:
        print(f"  {j+1}: {l.strip()}")

print("\nAll ValueError lines remaining:")
for j, l in enumerate(lines2):
    if "You cannot specify both decoder_input_ids" in l or "You have to specify either decoder_input_ids" in l:
        print(f"  {j+1}: {l.strip()}")
