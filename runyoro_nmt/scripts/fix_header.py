"""Remove the 3 accidentally-inserted lines at the top of the transformers file."""
from pathlib import Path

MODEL_FILE = Path(
    r"C:\Users\MarvinCliveTwesige\AppData\Local\Programs\Python\Python313"
    r"\Lib\site-packages\transformers\models\m2m_100\modeling_m2m_100.py"
)

lines = MODEL_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

# Check if first 3 lines are the accidental patch
if (
    "PATCH runyoro" in lines[0]
    and "if input_ids is not None and inputs_embeds is not None:" in lines[1]
    and "inputs_embeds = None" in lines[2]
):
    lines = lines[3:]
    MODEL_FILE.write_text("".join(lines), encoding="utf-8")
    print("Fixed: removed 3 accidental lines from file header")
else:
    print("Header looks OK — nothing changed")
    print("Line 1:", repr(lines[0]))

# Clear pyc
pycache = MODEL_FILE.parent / "__pycache__"
for f in pycache.glob("modeling_m2m_100*.pyc"):
    f.unlink()
    print(f"Deleted cache: {f.name}")

# Confirm final state
lines2 = MODEL_FILE.read_text(encoding="utf-8").splitlines()
print("\nVerification — first 4 lines:")
for i, l in enumerate(lines2[:4]):
    print(f"  {i+1}: {l}")
print("\nAll PATCH locations:")
for i, l in enumerate(lines2):
    if "PATCH runyoro" in l:
        print(f"  Line {i+1}: {l.strip()}")
print("\nAll remaining ValueError lines:")
for i, l in enumerate(lines2):
    if "You cannot specify both" in l:
        print(f"  Line {i+1}: {l.strip()}")
