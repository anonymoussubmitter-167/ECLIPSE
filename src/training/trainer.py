"""
Trainers for ECLIPSE modules.

Provides training loops with:
- Automatic mixed precision
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Logging (WandB integration)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import logging
from tqdm import tqdm
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer class with common functionality."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        use_wandb: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Use automatic mixed precision
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for checkpoints
            log_interval: Steps between logging
            use_wandb: Use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_wandb = use_wandb

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # WandB
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling")
                self.use_wandb = False

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch."""
        pass

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_to_device(batch)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    losses = self.compute_loss(batch)
                    loss = losses["total_loss"] / self.gradient_accumulation_steps
            else:
                losses = self.compute_loss(batch)
                loss = losses["total_loss"] / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += losses["total_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": losses["total_loss"].item()})

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics(losses, prefix="train")

        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        all_losses = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._move_to_device(batch)
            losses = self.compute_loss(batch)

            total_loss += losses["total_loss"].item()
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = 0
                all_losses[k] += v.item()
            num_batches += 1

        avg_losses = {f"val_{k}": v / num_batches for k, v in all_losses.items()}
        self._log_metrics(avg_losses)

        return avg_losses

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}
        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate()
            if "val_total_loss" in val_metrics:
                history["val_loss"].append(val_metrics["val_total_loss"])

                # Early stopping
                if val_metrics["val_total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_total_loss"]
                    self.save_checkpoint("best.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}.pt")

        return history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if self.use_wandb:
            self.wandb.log(metrics, step=self.global_step)

        logger.info(f"Step {self.global_step}: {metrics}")


class ECDNAFormerTrainer(BaseTrainer):
    """Trainer for ecDNA-Former (Module 1)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        oncogene_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.oncogene_weight = oncogene_weight

        from .losses import FocalLoss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ecDNA-Former loss."""
        # Forward pass
        outputs = self.model(
            sequence_features=batch.get("sequence_features"),
            topology_features=batch.get("topology_features"),
            fragile_site_features=batch.get("fragile_site_features"),
            copy_number_features=batch.get("copy_number_features"),
            return_embeddings=True,
        )

        losses = {}

        # Formation prediction loss (focal)
        formation_loss = self.focal_loss(
            outputs["formation_probability"],
            batch["label"].unsqueeze(-1),
        )
        losses["formation_loss"] = formation_loss

        # Oncogene prediction loss
        if "oncogene_labels" in batch:
            oncogene_loss = nn.functional.binary_cross_entropy(
                outputs["oncogene_probabilities"],
                batch["oncogene_labels"],
            )
            losses["oncogene_loss"] = self.oncogene_weight * oncogene_loss

        losses["total_loss"] = sum(losses.values())
        return losses


class CircularODETrainer(BaseTrainer):
    """Trainer for CircularODE (Module 2)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        physics_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.physics_weight = physics_weight

        from .losses import PhysicsInformedLoss
        self.physics_loss = PhysicsInformedLoss()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute CircularODE loss."""
        # Forward pass
        outputs = self.model(
            initial_state=batch["initial_state"],
            time_points=batch["time_points"][0],  # Shared time points
            treatment_info=batch.get("treatment"),
        )

        # Physics-informed loss
        losses = self.physics_loss(
            predicted_trajectory=outputs["copy_number_trajectory"],
            observed_trajectory=batch["copy_numbers"],
            mask=batch.get("mask"),
        )

        return losses


class VulnCausalTrainer(BaseTrainer):
    """Trainer for VulnCausal (Module 3)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        irm_weight: float = 1.0,
        dag_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)
        self.irm_weight = irm_weight
        self.dag_weight = dag_weight

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute VulnCausal loss."""
        losses = self.model.get_loss(
            expression=batch["expression"],
            crispr_scores=batch["crispr_scores"],
            ecdna_labels=batch["ecdna_label"],
            environments=batch.get("covariates", torch.zeros(batch["expression"].shape[0])),
        )

        return losses


class ECLIPSETrainer(BaseTrainer):
    """Trainer for full ECLIPSE framework."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        module_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        if module_weights is None:
            module_weights = {
                "former": 1.0,
                "dynamics": 1.0,
                "vuln": 1.0,
                "integration": 0.5,
            }
        self.module_weights = module_weights

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute ECLIPSE loss."""
        # Forward pass
        outputs = self.model(
            sequence_features=batch.get("sequence_features"),
            topology_features=batch.get("topology_features"),
            fragile_site_features=batch.get("fragile_site_features"),
            copy_number_features=batch.get("copy_number_features"),
            initial_state=batch.get("initial_state"),
            time_points=batch.get("time_points"),
            expression=batch.get("expression"),
            crispr_scores=batch.get("crispr_scores"),
            ecdna_labels=batch.get("ecdna_label"),
            run_all_modules=True,
        )

        losses = {}

        # Formation loss
        if "formation_probability" in outputs and "label" in batch:
            formation_loss = nn.functional.binary_cross_entropy(
                outputs["formation_probability"].squeeze(),
                batch["label"].float(),
            )
            losses["formation_loss"] = self.module_weights["former"] * formation_loss

        # Risk classification loss
        if "risk_logits" in outputs and "risk_level" in batch:
            risk_loss = nn.functional.cross_entropy(
                outputs["risk_logits"],
                batch["risk_level"],
            )
            losses["risk_loss"] = self.module_weights["integration"] * risk_loss

        losses["total_loss"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses
