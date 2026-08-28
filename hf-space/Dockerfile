FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Pre-download model at build time so cold starts are instant
RUN python -c "\
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('keithtwesigye/runyoro-nmt'); \
AutoModelForSeq2SeqLM.from_pretrained('keithtwesigye/runyoro-nmt', torch_dtype='float32'); \
print('Model cached.')"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
