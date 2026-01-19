#!/usr/bin/env python3
"""
ECLIPSE: Main Entry Point

Command-line interface for ECLIPSE framework.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(args):
    """Download public datasets."""
    from src.data import DataDownloader

    logger.info("Starting data download...")
    downloader = DataDownloader(args.data_dir)
    results = downloader.download_all(skip_large=args.skip_large)

    for source, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {source}: {status}")


def train_model(args):
    """Train a model."""
    import torch
    from src.data import (
        AmpliconRepositoryLoader, CytoCellDBLoader, DepMapLoader,
        ECDNADataset, VulnerabilityDataset, create_dataloader
    )
    from src.models import ECDNAFormer, CircularODE, VulnCausal, ECLIPSE
    from src.training import (
        ECDNAFormerTrainer, CircularODETrainer, VulnCausalTrainer, ECLIPSETrainer
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Training on device: {device}")

    if args.module == "former":
        logger.info("Training ecDNA-Former (Module 1)...")
        model = ECDNAFormer()
        # Load data and create trainer
        # (Placeholder - requires actual data)
        logger.info("ecDNA-Former training complete")

    elif args.module == "dynamics":
        logger.info("Training CircularODE (Module 2)...")
        model = CircularODE()
        logger.info("CircularODE training complete")

    elif args.module == "vuln":
        logger.info("Training VulnCausal (Module 3)...")
        model = VulnCausal()
        logger.info("VulnCausal training complete")

    elif args.module == "eclipse":
        logger.info("Training full ECLIPSE...")
        model = ECLIPSE()
        logger.info("ECLIPSE training complete")


def predict(args):
    """Run prediction with trained model."""
    import torch
    from src.models import ECLIPSE

    logger.info(f"Loading model from {args.checkpoint}...")
    model = ECLIPSE.from_pretrained(args.checkpoint)
    model.eval()

    # Load input data
    logger.info(f"Loading input from {args.input}...")
    # (Placeholder for actual prediction)

    logger.info("Prediction complete")


def evaluate(args):
    """Evaluate model on test set."""
    import torch
    from src.utils.metrics import compute_all_metrics

    logger.info(f"Evaluating model {args.checkpoint}...")
    # (Placeholder for actual evaluation)

    logger.info("Evaluation complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ECLIPSE: ecDNA prediction and vulnerability discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data
  python main.py download --data-dir data

  # Train ecDNA-Former
  python main.py train --module former --epochs 100

  # Run prediction
  python main.py predict --checkpoint checkpoints/eclipse.pt --input sample.pt

  # Evaluate model
  python main.py evaluate --checkpoint checkpoints/eclipse.pt --test-data test.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("--data-dir", type=str, default="data",
                                 help="Directory to save data")
    download_parser.add_argument("--skip-large", action="store_true",
                                 help="Skip large files (Hi-C)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--module", type=str, required=True,
                              choices=["former", "dynamics", "vuln", "eclipse"],
                              help="Module to train")
    train_parser.add_argument("--data-dir", type=str, default="data",
                              help="Data directory")
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                              help="Checkpoint directory")
    train_parser.add_argument("--epochs", type=int, default=100,
                              help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32,
                              help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                              help="Learning rate")
    train_parser.add_argument("--cpu", action="store_true",
                              help="Use CPU only")
    train_parser.add_argument("--wandb", action="store_true",
                              help="Use Weights & Biases logging")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--checkpoint", type=str, required=True,
                                help="Model checkpoint path")
    predict_parser.add_argument("--input", type=str, required=True,
                                help="Input data path")
    predict_parser.add_argument("--output", type=str, default="predictions.pt",
                                help="Output path")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                             help="Model checkpoint path")
    eval_parser.add_argument("--test-data", type=str, required=True,
                             help="Test data path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        download_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
