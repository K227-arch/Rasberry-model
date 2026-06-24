#!/usr/bin/env python3
"""
process_new2.py
===============
Process the 'new 2' raw data folder through the pipeline:
  1. Extract pairs from new2 raw files (docx + xlsx)
  2. Validate extracted pairs
  3. Align sentence pairs
  4. Clean & normalise
  5. Save cleaned data to 'new2 cleaned' folder
  6. Combine with existing raw data
  7. Run full training pipeline on combined data

Run:
    python scripts/process_new2.py
    python scripts/process_new2.py --clean-only    # Only extract+clean, skip training
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Ensure src is importable
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
        logging.FileHandler("pipeline_new2.log"),
    ],
)
logger = logging.getLogger("process_new2")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_ROOT = PROJECT_ROOT.parent / "raw data"
NEW2_DIR = RAW_DATA_ROOT / "new 2"
NEW2_CLEANED_DIR = RAW_DATA_ROOT / "new2 cleaned"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path) -> dict:
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


def extract_and_clean_new2():
    """Extract and clean the new2 data, saving results to 'new2 cleaned' folder."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Processing 'new 2' folder")
    logger.info("=" * 60)

    # Step 1: Extract
    logger.info("Step 1: Extracting pairs from new2 files...")
    extractor = DataExtractor(NEW2_DIR)
    raw_pairs = extractor.extract_flat()
    logger.info("Extracted %d raw pairs from new2", len(raw_pairs))

    if not raw_pairs:
        logger.error("No data extracted from new2 — aborting")
        return []

    # Step 2: Validate
    logger.info("Step 2: Validating extracted pairs...")
    validator = DataValidator(
        min_tokens=2,
        max_tokens=200,
        min_char_ratio=0.4,
        max_char_ratio_multiplier=4.0,
    )
    val_result = validator.validate(raw_pairs)
    logger.info("Validation: %s", val_result.summary())
    valid_pairs = val_result.valid_pairs

    # Step 3: Alignment
    logger.info("Step 3: Checking alignment...")
    aligner = SentenceAligner(length_ratio_threshold=0.25)
    align_result = aligner.check_alignment(valid_pairs)
    logger.info("Alignment: %s", align_result.summary())
    aligned_pairs = align_result.aligned_pairs

    # Step 4: Clean
    logger.info("Step 4: Cleaning & normalising...")
    cleaner = DataCleaner()
    cleaned_pairs, clean_records = cleaner.clean(aligned_pairs)
    logger.info("Cleaned pairs: %d", len(cleaned_pairs))

    # Step 5: Save to 'new2 cleaned' folder
    logger.info("Step 5: Saving cleaned data to '%s'", NEW2_CLEANED_DIR)
    NEW2_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Save as TSV (the standard format for the pipeline)
    cleaned_tsv_path = NEW2_CLEANED_DIR / "cleaned_pairs.tsv"
    save_pairs_tsv(cleaned_pairs, str(cleaned_tsv_path))
    logger.info("Saved %d cleaned pairs to %s", len(cleaned_pairs), cleaned_tsv_path)

    # Save cleaning report for new2
    report_path = NEW2_CLEANED_DIR / "cleaning_report.md"
    cleaner.generate_report(clean_records, str(report_path))
    logger.info("Cleaning report saved to %s", report_path)

    # Save validation report
    val_report_path = NEW2_CLEANED_DIR / "validation_report.md"
    with open(val_report_path, "w", encoding="utf-8") as f:
        f.write("# New2 Data Validation Report\n\n")
        f.write(f"**Total raw pairs extracted:** {len(raw_pairs)}\n")
        f.write(f"**Valid pairs after validation:** {len(valid_pairs)}\n")
        f.write(f"**Aligned pairs:** {len(aligned_pairs)}\n")
        f.write(f"**Cleaned pairs:** {len(cleaned_pairs)}\n\n")
        f.write(f"## Statistics\n\n```json\n{json.dumps(val_result.stats, indent=2)}\n```\n\n")
        if val_result.issues:
            f.write("## Issues (first 50)\n\n")
            for iss in val_result.issues[:50]:
                f.write(f"- [{iss.severity}] {iss.issue_type}: {iss.message}\n")
    logger.info("Validation report saved to %s", val_report_path)

    # Save a summary
    summary_path = NEW2_CLEANED_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("New2 Data Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Source folder: {NEW2_DIR}\n")
        f.write(f"Files processed: {len(list(NEW2_DIR.iterdir()))}\n")
        f.write(f"Raw pairs extracted: {len(raw_pairs)}\n")
        f.write(f"Valid pairs: {len(valid_pairs)}\n")
        f.write(f"Aligned pairs: {len(aligned_pairs)}\n")
        f.write(f"Final cleaned pairs: {len(cleaned_pairs)}\n")
        f.write(f"\nOutput: {cleaned_tsv_path}\n")

    logger.info("=" * 60)
    logger.info("NEW2 PROCESSING COMPLETE")
    logger.info("  Raw: %d -> Cleaned: %d pairs", len(raw_pairs), len(cleaned_pairs))
    logger.info("  Output: %s", NEW2_CLEANED_DIR)
    logger.info("=" * 60)

    return cleaned_pairs


def run_combined_pipeline(new2_cleaned_pairs, config: dict, skip_training: bool = False):
    """Run the full pipeline combining original + new2 data."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Full Pipeline with Combined Data")
    logger.info("=" * 60)

    report = ReportGenerator(project_name=config["project"]["name"])

    proc_dir = Path(config["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = Path(config["data"]["augmented_dir"])
    aug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(config["data"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    tm_dir = Path(config["data"]["tm_dir"])
    tm_dir.mkdir(parents=True, exist_ok=True)

    # Extract from original raw data directory
    logger.info("Extracting from original raw data...")
    extractor = DataExtractor(RAW_DATA_ROOT)
    original_pairs = extractor.extract_flat()
    logger.info("Original raw pairs: %d", len(original_pairs))

    # Combine original + new2 cleaned data
    combined_pairs = original_pairs + new2_cleaned_pairs
    logger.info("Combined raw pairs (original + new2): %d", len(combined_pairs))

    report.add_section(
        "extraction",
        f"## Data Extraction\n\n"
        f"**Original pairs:** {len(original_pairs)}\n"
        f"**New2 cleaned pairs:** {len(new2_cleaned_pairs)}\n"
        f"**Combined total:** {len(combined_pairs)}\n",
    )

    if not combined_pairs:
        logger.error("No data available — aborting")
        return

    # Validation on combined data
    logger.info("Validating combined data...")
    validator = DataValidator(
        min_tokens=config["data"]["min_tokens"],
        max_tokens=config["data"]["max_tokens"],
        min_char_ratio=config["data"]["min_char_ratio"],
        max_char_ratio_multiplier=config["data"]["max_char_ratio_multiplier"],
    )
    val_result = validator.validate(combined_pairs)
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

    # Alignment
    logger.info("Checking alignment...")
    aligner = SentenceAligner(length_ratio_threshold=0.25)
    align_result = aligner.check_alignment(valid_pairs)
    logger.info("Alignment: %s", align_result.summary())
    valid_pairs = align_result.aligned_pairs

    # Cleaning
    logger.info("Cleaning combined data...")
    cleaner = DataCleaner()
    cleaned_pairs, clean_records = cleaner.clean(valid_pairs)
    cleaner.generate_report(clean_records, str(reports_dir / "cleaning_report.md"))
    logger.info("Cleaned pairs: %d", len(cleaned_pairs))

    report.add_section("cleaning", f"## Cleaning\n\n**Pairs after cleaning:** {len(cleaned_pairs)}")

    # Translation Memory + Glossary
    logger.info("Building linguistic resources...")
    tm_builder = TranslationMemoryBuilder(tm_dir)
    resource_paths = tm_builder.build_all(cleaned_pairs)
    logger.info("Resources: %s", resource_paths)

    report.add_section(
        "resources",
        f"## Linguistic Resources\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in resource_paths.items()),
    )

    # Augmentation
    logger.info("Augmenting data...")
    augmentor = DataAugmentor(
        seed=config["data"]["seed"],
        deletion_prob=config["data"]["augmentation"]["token_deletion_prob"],
        swap_prob=config["data"]["augmentation"]["token_swap_prob"],
        augment_multiplier=config["data"]["augmentation"]["augment_multiplier"],
    )
    aug_result = augmentor.augment(cleaned_pairs)
    augmentor.generate_report(aug_result, str(reports_dir / "augmentation_report.md"))

    all_pairs = cleaned_pairs + aug_result.augmented_pairs
    logger.info("Total pairs after augmentation: %d", len(all_pairs))

    # Save processed data
    save_pairs_tsv(all_pairs, str(aug_dir / "all_pairs.tsv"))
    save_pairs_tsv(cleaned_pairs, str(proc_dir / "cleaned_pairs.tsv"))

    # Split
    logger.info("Splitting data...")
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

    logger.info("Split: train=%d, val=%d, test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    report.add_section(
        "splits",
        f"## Data Splits\n\n"
        f"| Split | Count |\n|-------|-------|\n"
        f"| Train | {len(train_pairs)} |\n"
        f"| Val | {len(val_pairs)} |\n"
        f"| Test | {len(test_pairs)} |",
    )

    if skip_training:
        logger.info("--clean-only flag set — stopping after data preparation")
        report.generate_markdown(str(reports_dir / "pipeline_report.md"))
        return

    # Training
    logger.info("=" * 60)
    logger.info("STAGE: Model Training (runyoro-nmt-v1)")
    logger.info("=" * 60)

    trainer = NMTTrainer(config)
    trainer.setup()
    trainer.train(train_pairs, val_pairs)

    model_card_path = str(Path(config["training"]["output_dir"]) / "README.md")
    trainer.generate_model_card(model_card_path)
    logger.info("Training complete")

    model_path = config["training"]["output_dir"]

    # Evaluation
    logger.info("Evaluating model...")
    evaluator = NMTEvaluator(
        model_path=model_path,
        src_lang=config["model"]["src_lang_nllb"],
        tgt_lang=config["model"]["tgt_lang_nllb"],
        beam_size=config["training"]["generation_num_beams"],
    )

    eval_results = evaluator.evaluate(test_pairs, output_dir=str(reports_dir))
    report.add_eval_results(eval_results)
    logger.info("Evaluation results: %s", eval_results)

    # Error Analysis
    logger.info("Running error analysis...")
    sources = [r for r, e in test_pairs[:200]]
    references = [e for r, e in test_pairs[:200]]
    predictions = evaluator.translate_batch(
        sources,
        config["model"]["src_lang_nllb"],
        config["model"]["tgt_lang_nllb"],
    )

    glossary_json = resource_paths.get("glossary_json", "")
    glossary = {}
    if glossary_json and Path(str(glossary_json)).exists():
        data = json.loads(Path(str(glossary_json)).read_text(encoding="utf-8"))
        glossary = {item["runyoro"]: item["english"] for item in data}

    analyzer = ErrorAnalyzer(glossary=glossary)
    error_result = analyzer.analyze(sources, predictions, references)
    analyzer.generate_report(error_result, str(reports_dir / "error_analysis_report.md"))
    logger.info("Error analysis: %s", error_result.summary())

    # Quantization
    logger.info("Quantizing model...")
    quantizer = ModelQuantizer(
        model_path=model_path,
        output_dir=str(Path(config["training"]["output_dir"]).parent / "exported"),
    )
    try:
        quantized_path = quantizer.quantize_int8_pytorch()
        logger.info("INT8 model: %s", quantized_path)
    except Exception as e:
        logger.warning("Quantization failed: %s", e)

    # Final Report
    report.generate_markdown(str(reports_dir / "pipeline_report.md"))
    try:
        report.generate_html(str(reports_dir / "pipeline_report.html"))
    except Exception:
        pass
    logger.info("Pipeline complete. Reports in: %s", reports_dir)


def main():
    parser = argparse.ArgumentParser(description="Process new2 data and run combined training")
    parser.add_argument(
        "--clean-only", action="store_true",
        help="Only extract and clean new2 data, skip training",
    )
    args = parser.parse_args()

    # Fix MLflow file store deprecation
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # ===== GPU MEMORY MANAGEMENT =====
    # Clear GPU memory and ensure both GPUs are available
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info("CUDA available: %d GPU(s) detected", num_gpus)
        for i in range(num_gpus):
            logger.info("  GPU %d: %s (%.1f GB total, %.1f GB free)", i,
                        torch.cuda.get_device_name(i),
                        torch.cuda.get_device_properties(i).total_memory / 1e9,
                        (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1e9)
        # Clear all GPU caches
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("GPU memory cleared")
    else:
        logger.warning("No CUDA GPUs detected!")

    # Make both GPUs visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Disable NCCL on Windows (use gloo or DataParallel instead)
    os.environ["NCCL_DEBUG"] = "WARN"

    logger.info("=" * 60)
    logger.info("PROCESS NEW2 DATA & COMBINED TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info("New2 source: %s", NEW2_DIR)
    logger.info("New2 cleaned output: %s", NEW2_CLEANED_DIR)

    # Phase 1: Extract and clean new2 data
    new2_cleaned_pairs = extract_and_clean_new2()

    if not new2_cleaned_pairs:
        logger.error("No cleaned pairs from new2 — cannot proceed")
        sys.exit(1)

    if args.clean_only:
        logger.info("--clean-only mode: Done. Cleaned data in: %s", NEW2_CLEANED_DIR)
        return

    # Phase 2: Run full pipeline with combined data
    config = load_config(CONFIG_PATH)

    # Change working directory to project root for relative paths in config
    os.chdir(PROJECT_ROOT)

    run_combined_pipeline(new2_cleaned_pairs, config, skip_training=False)


if __name__ == "__main__":
    main()
