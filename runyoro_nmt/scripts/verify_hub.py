from huggingface_hub import HfApi

api = HfApi(token="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")

print("\n=== Hub Verification ===\n")

# Model
m = api.repo_info("kathay/runyoro-nmt-v1", repo_type="model")
model_files = list(api.list_repo_files("kathay/runyoro-nmt-v1", repo_type="model"))
print(f"MODEL   : https://huggingface.co/{m.id}")
print(f"  Files : {model_files}")

# Dataset
d = api.repo_info("kathay/runyoro-rutooro-en-parallel", repo_type="dataset")
ds_files = list(api.list_repo_files("kathay/runyoro-rutooro-en-parallel", repo_type="dataset"))
print(f"\nDATASET : https://huggingface.co/datasets/{d.id}")
print(f"  Files ({len(ds_files)}):")
for f in ds_files:
    print(f"    - {f}")

# Space
s = api.repo_info("kathay/runyoro-translator", repo_type="space")
space_files = list(api.list_repo_files("kathay/runyoro-translator", repo_type="space"))
print(f"\nSPACE   : https://huggingface.co/spaces/{s.id}")
print(f"  Files : {space_files}")

print("\n=== All resources live on kathay Hub ===\n")
