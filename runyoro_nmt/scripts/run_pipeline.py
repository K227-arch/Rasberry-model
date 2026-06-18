#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master orchestrator for the runyoro-nmt-v1 pipeline.

Executes all stages in order:
  1. Data extraction from raw files
  2. Validation & QA
  3. Alignment check
  4. Cleaning & normalisation
  5. Deduplication
  6. Translation memory + glossary + TBX build
  7. Augmentation
  8. Train/val/test split
  9. Model training (NLLB-200 fine-tune, runyoro-nmt-v1)
 10. Evaluation (BLEU, chrF++, COMET, TER, BERTScore)
 11. Error analysis
 12. Model quantization & export
 13. Push to Hugging Face Hub (kathay)
 14. Deploy Hugging Face Space demo
 15. Generate final report

Run:
    python scripts/run_pipeline.py --config configs/config.yaml
    python scripts/run_pipeline.py --config configs/config.yaml --skip-training
    python scripts/run_pipeline.py --config configs/config.yaml --data-only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml  # type: ignore

# Make sure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline import (
    DataExtractor,
    DataValidator,
    DataCleaner,
    SentenceAligner,
    DataAugmentor,
    TranslationMemoryBuilder,
)
from training import NMTTrainer
from evaluation import NMTEvaluator, ErrorAnalyzer
from inference import ModelQuantizer
from utils import ReportGenerator, split_dataset, HubPusher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("run_pipeline")


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pairs_tsv(pairs, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rny, eng in pairs:
            f.write(f"{rny}\t{eng}\n")


def load_pairs_tsv(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def run_pipeline(config: dict, args: argparse.Namespace) -> None:
    report = ReportGenerator(project_name=config["project"]["name"])

    proc_dir = Path(config["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = Path(config["data"]["augmented_dir"])
    aug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(config["data"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    tm_dir = Path(config["data"]["tm_dir"])
    tm_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # STAGE 1: Extract
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Extraction")
    logger.info("=" * 60)

    raw_data_dir = Path(config["data"]["raw_files"][0]).parent
    # Use the directory containing the raw files
    extractor = DataExtractor(raw_data_dir)
    raw_pairs = extractor.extract_flat()
    logger.info("Extracted %d raw pairs", len(raw_pairs))

    report.add_section(
        "extraction",
        f"## Data Extraction\n\n**Total raw pairs extracted:** {len(raw_pairs)}\n"
        f"**Source directory:** `{raw_data_dir}`",
    )

    if not raw_pairs:
        logger.error("No data extracted — aborting")
        return

    # =========================================================
    # STAGE 2: Validation
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Validation & QA")
    logger.info("=" * 60)

    validator = DataValidator(
        min_tokens=config["data"]["min_tokens"],
        max_tokens=config["data"]["max_tokens"],
        min_char_ratio=config["data"]["min_char_ratio"],
        max_char_ratio_multiplier=config["data"]["max_char_ratio_multiplier"],
    )
    val_result = validator.validate(raw_pairs)
    logger.info("Validation: %s", val_result.summary())
    report.add_data_stats(val_result.stats)

    valid_pairs = val_result.valid_pairs

    # Save validation report
    val_report_path = reports_dir / "validation_report.md"
    with open(val_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Validation Report\n\n{json.dumps(val_result.stats, indent=2)}\n\n")
        f.write("## Issues\n\n")
        for iss in val_result.issues[:50]:
            f.write(f"- [{iss.severity}] {iss.issue_type}: {iss.message}\n")
    logger.info("Validation report saved: %s", val_report_path)

    # =========================================================
    # STAGE 3: Alignment
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 3: Alignment Check")
    logger.info("=" * 60)

    aligner = SentenceAligner(length_ratio_threshold=0.25)
    align_result = aligner.check_alignment(valid_pairs)
    logger.info("Alignment: %s", align_result.summary())
    valid_pairs = align_result.aligned_pairs

    # =========================================================
    # STAGE 4: Cleaning
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 4: Cleaning & Normalisation")
    logger.info("=" * 60)

    cleaner = DataCleaner()
    cleaned_pairs, clean_records = cleaner.clean(valid_pairs)
    clean_report = cleaner.generate_report(
        clean_records, str(reports_dir / "cleaning_report.md")
    )

    report.add_section("cleaning", f"## Cleaning\n\n**Pairs after cleaning:** {len(cleaned_pairs)}")

    # =========================================================
    # STAGE 5: Translation Memory + Glossary
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 5: Building Linguistic Resources")
    logger.info("=" * 60)

    tm_builder = TranslationMemoryBuilder(tm_dir)
    resource_paths = tm_builder.build_all(cleaned_pairs)
    logger.info("Resources: %s", resource_paths)

    report.add_section(
        "resources",
        f"## Linguistic Resources\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in resource_paths.items()),
    )

    # =========================================================
    # STAGE 6: Augmentation
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 6: Data Augmentation")
    logger.info("=" * 60)

    augmentor = DataAugmentor(
        seed=config["data"]["seed"],
        deletion_prob=config["data"]["augmentation"]["token_deletion_prob"],
        swap_prob=config["data"]["augmentation"]["token_swap_prob"],
        augment_multiplier=config["data"]["augmentation"]["augment_multiplier"],
    )
    aug_result = augmentor.augment(cleaned_pairs)
    augmentor.generate_report(aug_result, str(reports_dir / "augmentation_report.md"))

    # Combine original + augmented
    all_pairs = cleaned_pairs + aug_result.augmented_pairs
    logger.info("Total pairs after augmentation: %d", len(all_pairs))

    # Save processed data
    save_pairs_tsv(all_pairs, str(aug_dir / "all_pairs.tsv"))
    save_pairs_tsv(cleaned_pairs, str(proc_dir / "cleaned_pairs.tsv"))

    # =========================================================
    # STAGE 7: Split
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 7: Train/Val/Test Split")
    logger.info("=" * 60)

    train_pairs, val_pairs, test_pairs = split_dataset(
        all_pairs,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=config["data"]["seed"],
    )

    save_pairs_tsv(train_pairs, str(proc_dir / "train.tsv"))
    save_pairs_tsv(val_pairs, str(proc_dir / "val.tsv"))
    save_pairs_tsv(test_pairs, str(proc_dir / "test.tsv"))

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_pairs), len(val_pairs), len(test_pairs),
    )

    report.add_section(
        "splits",
        f"## Data Splits\n\n"
        f"| Split | Count |\n|-------|-------|\n"
        f"| Train | {len(train_pairs)} |\n"
        f"| Val | {len(val_pairs)} |\n"
        f"| Test | {len(test_pairs)} |",
    )

    if args.data_only:
        logger.info("--data-only flag set — stopping after data preparation")
        report.generate_markdown(str(reports_dir / "pipeline_report.md"))
        return

    # =========================================================
    # STAGE 8: Training
    # =========================================================
    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("STAGE 8: Model Training (runyoro-nmt-v1)")
        logger.info("=" * 60)

        trainer = NMTTrainer(config)
        trainer.setup()
        trainer.train(train_pairs, val_pairs)

        model_card_path = str(
            Path(config["training"]["output_dir"]) / "README.md"
        )
        trainer.generate_model_card(model_card_path)
        logger.info("Training complete")

    model_path = config["training"]["output_dir"]

    # =========================================================
    # STAGE 9: Evaluation
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 9: Evaluation")
    logger.info("=" * 60)

    evaluator = NMTEvaluator(
        model_path=model_path,
        src_lang=config["model"]["src_lang_nllb"],
        tgt_lang=config["model"]["tgt_lang_nllb"],
        beam_size=config["training"]["generation_num_beams"],
    )

    eval_results = evaluator.evaluate(test_pairs, output_dir=str(reports_dir))
    report.add_eval_results(eval_results)
    logger.info("Evaluation results: %s", eval_results)

    # =========================================================
    # STAGE 10: Error Analysis
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 10: Error Analysis")
    logger.info("=" * 60)

    sources = [r for r, e in test_pairs[:200]]
    references = [e for r, e in test_pairs[:200]]
    predictions = evaluator.translate_batch(
        sources,
        config["model"]["src_lang_nllb"],
        config["model"]["tgt_lang_nllb"],
    )

    # Load glossary for error analysis
    glossary_json = resource_paths.get("glossary_json", "")
    glossary = {}
    if glossary_json and Path(str(glossary_json)).exists():
        import json as _json
        data = _json.loads(Path(str(glossary_json)).read_text(encoding="utf-8"))
        glossary = {item["runyoro"]: item["english"] for item in data}

    analyzer = ErrorAnalyzer(glossary=glossary)
    error_result = analyzer.analyze(sources, predictions, references)
    analyzer.generate_report(error_result, str(reports_dir / "error_analysis_report.md"))
    logger.info("Error analysis: %s", error_result.summary())

    # =========================================================
    # STAGE 11: Quantization
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 11: Model Quantization")
    logger.info("=" * 60)

    quantizer = ModelQuantizer(
        model_path=model_path,
        output_dir=str(Path(config["training"]["output_dir"]).parent / "exported"),
    )

    try:
        quantized_path = quantizer.quantize_int8_pytorch()
        logger.info("INT8 model: %s", quantized_path)
    except Exception as e:
        logger.warning("Quantization failed: %s", e)

    # =========================================================
    # STAGE 12: Hub Push
    # =========================================================
    if not args.skip_hub:
        logger.info("=" * 60)
        logger.info("STAGE 12: Push to Hugging Face Hub (kathay)")
        logger.info("=" * 60)

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN env var not set — skipping Hub push")
        else:
            pusher = HubPusher(hf_token=hf_token)

            # Push dataset
            pusher.push_dataset(
                cleaned_pairs,
                dataset_id=config["project"]["hf_dataset_id"],
            )

            # Push model
            pusher.push_model(
                model_path=model_path,
                model_id=config["project"]["hf_model_id"],
                model_card_path=str(Path(model_path) / "README.md"),
            )

            # Create Space
            space_app = Path(__file__).parent.parent / "ui" / "gradio_app.py"
            pusher.create_space(
                space_id=config["project"]["hf_space_id"],
                space_app_path=str(space_app) if space_app.exists() else None,
            )

    # =========================================================
    # STAGE 13: Final Report
    # =========================================================
    logger.info("=" * 60)
    logger.info("STAGE 13: Final Report")
    logger.info("=" * 60)

    report.generate_markdown(str(reports_dir / "pipeline_report.md"))
    report.generate_html(str(reports_dir / "pipeline_report.html"))
    logger.info("Pipeline complete. Reports in: %s", reports_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="runyoro-nmt-v1 training pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training (use pre-existing checkpoint)",
    )
    parser.add_argument(
        "--skip-hub", action="store_true",
        help="Skip Hugging Face Hub push",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Run only data preparation stages",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / args.config

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    logger.info("Starting runyoro-nmt-v1 pipeline")
    logger.info("Project: %s", config["project"]["name"])
    logger.info(
        "NOTE: This is a SEPARATE model from any existing NLLB fine-tunes."
    )

    run_pipeline(config, args)


if __name__ == "__main__":
    main()
