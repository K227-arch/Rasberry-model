from huggingface_hub import HfApi
import io

api = HfApi()

dockerfile_content = (
    "FROM python:3.11-slim\n"
    "WORKDIR /app\n"
    "COPY requirements.txt .\n"
    "RUN pip install --no-cache-dir -r requirements.txt\n"
    "COPY app.py .\n"
    "EXPOSE 7860\n"
    'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]\n'
)

api.upload_file(
    path_or_fileobj=io.BytesIO(dockerfile_content.encode()),
    path_in_repo="Dockerfile",
    repo_id="keithtwesigye/runyoro-translator-api",
    repo_type="space",
    commit_message="fix: run app:app not main:app — use our clean FastAPI endpoint",
)
print("Dockerfile updated on Space — rebuild triggered.")
