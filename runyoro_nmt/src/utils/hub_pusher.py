"""
HubPusher
=========
Handles all Hugging Face Hub operations:
  - Push fine-tuned model to kathay/runyoro-nmt-v1
  - Push dataset to kathay/runyoro-rutooro-en-parallel
  - Create/update Hugging Face Space for demo
  - Version artifacts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HubPusher:
    """Manages Hugging Face Hub uploads for runyoro-nmt-v1."""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi  # type: ignore
                self._api = HfApi(token=self.hf_token)
            except ImportError:
                raise RuntimeError("huggingface_hub required: pip install huggingface_hub")
        return self._api

    # ------------------------------------------------------------------
    # Dataset push
    # ------------------------------------------------------------------
    def push_dataset(
        self,
        pairs: List[Tuple[str, str]],
        dataset_id: str = "kathay/runyoro-rutooro-en-parallel",
        commit_message: str = "runyoro-nmt-v1 training data",
    ) -> str:
        try:
            from datasets import Dataset  # type: ignore

            logger.info("Pushing dataset to Hub: %s", dataset_id)
            data = {
                "runyoro_rutooro": [r for r, e in pairs],
                "english": [e for r, e in pairs],
            }
            ds = Dataset.from_dict(data)
            ds.push_to_hub(dataset_id, token=self.hf_token, commit_message=commit_message)
            url = f"https://huggingface.co/datasets/{dataset_id}"
            logger.info("Dataset pushed: %s", url)
            return url
        except Exception as e:
            logger.error("Dataset push failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Model push
    # ------------------------------------------------------------------
    def push_model(
        self,
        model_path: str,
        model_id: str = "kathay/runyoro-nmt-v1",
        model_card_path: Optional[str] = None,
        commit_message: str = "runyoro-nmt-v1: production fine-tune",
    ) -> str:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

            logger.info("Pushing model to Hub: %s", model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            tokenizer.push_to_hub(model_id, token=self.hf_token)
            model.push_to_hub(model_id, token=self.hf_token, commit_message=commit_message)

            if model_card_path:
                api = self._get_api()
                api.upload_file(
                    path_or_fileobj=model_card_path,
                    path_in_repo="README.md",
                    repo_id=model_id,
                    token=self.hf_token,
                )

            url = f"https://huggingface.co/{model_id}"
            logger.info("Model pushed: %s", url)
            return url
        except Exception as e:
            logger.error("Model push failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Space creation
    # ------------------------------------------------------------------
    def create_space(
        self,
        space_id: str = "kathay/runyoro-translator",
        space_app_path: Optional[str] = None,
    ) -> str:
        try:
            api = self._get_api()
            logger.info("Creating/updating Space: %s", space_id)

            # Create space if it doesn't exist
            try:
                api.create_repo(
                    repo_id=space_id,
                    repo_type="space",
                    space_sdk="gradio",
                    token=self.hf_token,
                    exist_ok=True,
                )
            except Exception:
                pass  # already exists

            if space_app_path:
                api.upload_file(
                    path_or_fileobj=space_app_path,
                    path_in_repo="app.py",
                    repo_id=space_id,
                    repo_type="space",
                    token=self.hf_token,
                )

            url = f"https://huggingface.co/spaces/{space_id}"
            logger.info("Space deployed: %s", url)
            return url
        except Exception as e:
            logger.error("Space creation failed: %s", e)
            raise
