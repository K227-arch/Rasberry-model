"""Update the HF Space with correct model path and requirements."""
from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

HF_TOKEN = "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"
api = HfApi(token=HF_TOKEN)

# Fix app.py — use HF Hub model ID for Space deployment
app = Path("ui/gradio_app.py").read_text(encoding="utf-8")
app = app.replace(
    'MODEL_ID = os.environ.get("MODEL_ID", "./models/checkpoints/runyoro-nmt-v1")',
    'MODEL_ID = os.environ.get("MODEL_ID", "kathay/runyoro-nmt-v1")',
)
space_app = Path("ui/gradio_app_space.py")
space_app.write_text(app, encoding="utf-8")

# Requirements compatible with gradio 5.x
reqs = (
    "gradio>=5.0.0\n"
    "transformers>=4.40.0\n"
    "torch>=2.0.0\n"
    "sentencepiece\n"
    "sacremoses\n"
    "huggingface-hub>=0.23\n"
)
req_path = Path("ui/requirements_space.txt")
req_path.write_text(reqs, encoding="utf-8")

ops = [
    CommitOperationAdd(path_in_repo="app.py",           path_or_fileobj=str(space_app)),
    CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=str(req_path)),
]

api.create_commit(
    repo_id="kathay/runyoro-translator",
    repo_type="space",
    operations=ops,
    commit_message="Fix requirements for gradio 5.x + transformers compatibility",
    token=HF_TOKEN,
)
print("Space updated: https://huggingface.co/spaces/kathay/runyoro-translator")
