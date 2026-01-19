"""
PyTorch Dataset classes for ECLIPSE.

Provides:
- ECDNADataset: For Module 1 (ecDNA formation prediction)
- DynamicsDataset: For Module 2 (ecDNA evolutionary dynamics)
- VulnerabilityDataset: For Module 3 (therapeutic vulnerability discovery)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData

logger = logging.getLogger(__name__)


class ECDNADataset(Dataset):
    """
    Dataset for ecDNA formation prediction (Module 1: ecDNA-Former).

    Provides:
    - Sequence context features
    - Hi-C topology graphs
    - Fragile site proximity features
    - Copy number context
    - ecDNA formation labels
    """

    def __init__(
        self,
        sample_ids: List[str],
        features: Dict[str, np.ndarray],
        labels: np.ndarray,
        oncogene_labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            sample_ids: List of sample identifiers
            features: Dictionary of feature arrays
            labels: Binary ecDNA formation labels (0/1)
            oncogene_labels: Multi-label oncogene predictions (optional)
            transform: Optional data transformation
        """
        self.sample_ids = sample_ids
        self.features = features
        self.labels = torch.FloatTensor(labels)
        self.oncogene_labels = (
            torch.FloatTensor(oncogene_labels) if oncogene_labels is not None else None
        )
        self.transform = transform

        # Validate
        n_samples = len(sample_ids)
        assert len(labels) == n_samples
        for key, arr in features.items():
            assert len(arr) == n_samples, f"Feature {key} has wrong length"

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        item = {
            "sample_id": self.sample_ids[idx],
            "sequence_features": torch.FloatTensor(
                self.features["sequence_features"][idx]
            ),
            "topology_features": torch.FloatTensor(
                self.features["topology_features"][idx]
            ),
            "fragile_site_features": torch.FloatTensor(
                self.features["fragile_site_features"][idx]
            ),
            "copy_number_features": torch.FloatTensor(
                self.features["copy_number_features"][idx]
            ),
            "label": self.labels[idx],
        }

        if self.oncogene_labels is not None:
            item["oncogene_labels"] = self.oncogene_labels[idx]

        if self.transform:
            item = self.transform(item)

        return item

    @classmethod
    def from_loaders(
        cls,
        sample_ids: List[str],
        feature_extractor,
        genomic_regions: pd.DataFrame,
        labels: np.ndarray,
        **kwargs
    ) -> "ECDNADataset":
        """
        Create dataset from data loaders.

        Args:
            sample_ids: List of sample IDs
            feature_extractor: FeatureExtractor instance
            genomic_regions: DataFrame with genomic coordinates
            labels: ecDNA formation labels
        """
        features = feature_extractor.extract_module1_features(
            sample_ids, genomic_regions
        )
        return cls(sample_ids, features, labels, **kwargs)


class ECDNAGraphDataset(Dataset):
    """
    Graph-based dataset for ecDNA-Former with Hi-C topology.

    Returns PyTorch Geometric Data objects for graph neural network training.
    """

    def __init__(
        self,
        sample_ids: List[str],
        node_features: List[np.ndarray],
        edge_indices: List[np.ndarray],
        edge_attrs: List[np.ndarray],
        global_features: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Initialize graph dataset.

        Args:
            sample_ids: List of sample identifiers
            node_features: List of node feature matrices (per sample)
            edge_indices: List of edge index arrays (per sample)
            edge_attrs: List of edge attribute arrays (per sample)
            global_features: Global (non-graph) features
            labels: ecDNA formation labels
        """
        self.sample_ids = sample_ids
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.edge_attrs = edge_attrs
        self.global_features = torch.FloatTensor(global_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> GraphData:
        """Get a single graph sample."""
        # Create PyG Data object
        data = GraphData(
            x=torch.FloatTensor(self.node_features[idx]),
            edge_index=torch.LongTensor(self.edge_indices[idx]),
            edge_attr=torch.FloatTensor(self.edge_attrs[idx]),
            y=self.labels[idx].unsqueeze(0),
            global_features=self.global_features[idx],
        )
        data.sample_id = self.sample_ids[idx]

        return data


class DynamicsDataset(Dataset):
    """
    Dataset for ecDNA evolutionary dynamics (Module 2: CircularODE).

    Provides time-series data of ecDNA copy numbers under various treatments.
    """

    def __init__(
        self,
        trajectories: List[Dict],
        max_time_points: int = 100,
        normalize: bool = True,
    ):
        """
        Initialize dynamics dataset.

        Args:
            trajectories: List of trajectory dictionaries with keys:
                - initial_state: Initial ecDNA state
                - time_points: Observation times
                - copy_numbers: Observed copy numbers
                - treatment: Treatment information
            max_time_points: Maximum number of time points per trajectory
            normalize: Whether to normalize copy numbers
        """
        self.trajectories = trajectories
        self.max_time_points = max_time_points
        self.normalize = normalize

        if normalize:
            self._compute_normalization_stats()

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single trajectory."""
        traj = self.trajectories[idx]

        # Pad or truncate to fixed length
        time_points = np.array(traj["time_points"])
        copy_numbers = np.array(traj["copy_numbers"])

        n_points = len(time_points)
        if n_points > self.max_time_points:
            # Subsample
            indices = np.linspace(0, n_points - 1, self.max_time_points, dtype=int)
            time_points = time_points[indices]
            copy_numbers = copy_numbers[indices]
        elif n_points < self.max_time_points:
            # Pad
            pad_length = self.max_time_points - n_points
            time_points = np.pad(time_points, (0, pad_length), constant_values=-1)
            copy_numbers = np.pad(copy_numbers, (0, pad_length), constant_values=0)

        # Create mask for valid time points
        mask = time_points >= 0

        if self.normalize:
            copy_numbers = (copy_numbers - self.cn_mean) / self.cn_std

        return {
            "initial_state": torch.FloatTensor(traj["initial_state"]),
            "time_points": torch.FloatTensor(time_points),
            "copy_numbers": torch.FloatTensor(copy_numbers),
            "mask": torch.BoolTensor(mask),
            "treatment": self._encode_treatment(traj.get("treatment")),
        }

    def _compute_normalization_stats(self):
        """Compute normalization statistics."""
        all_cn = []
        for traj in self.trajectories:
            all_cn.extend(traj["copy_numbers"])
        all_cn = np.array(all_cn)
        self.cn_mean = all_cn.mean()
        self.cn_std = all_cn.std() + 1e-8

    def _encode_treatment(self, treatment: Optional[Dict]) -> torch.Tensor:
        """Encode treatment information."""
        if treatment is None:
            return torch.zeros(16)

        # Simple encoding (in production, use learned embeddings)
        encoding = torch.zeros(16)

        treatment_types = {
            "targeted": 0,
            "chemo": 1,
            "immuno": 2,
            "ecdna_specific": 3,
            "none": 4,
        }

        if "type" in treatment:
            type_idx = treatment_types.get(treatment["type"], 4)
            encoding[type_idx] = 1.0

        if "dose" in treatment:
            encoding[8] = treatment["dose"]

        if "duration" in treatment:
            encoding[9] = treatment["duration"]

        return encoding

    @classmethod
    def from_simulator(
        cls,
        n_trajectories: int = 1000,
        time_horizon: float = 100.0,
        dt: float = 1.0,
        **kwargs
    ) -> "DynamicsDataset":
        """
        Create dataset from simulated trajectories.

        Uses ecSimulator-style dynamics for training when real longitudinal
        data is limited.

        Args:
            n_trajectories: Number of trajectories to simulate
            time_horizon: Total simulation time
            dt: Time step
        """
        trajectories = []
        np.random.seed(42)

        for i in range(n_trajectories):
            # Initial copy number (5-50)
            initial_cn = np.random.uniform(5, 50)

            # Simulate trajectory
            time_points = np.arange(0, time_horizon, dt)
            copy_numbers = cls._simulate_trajectory(initial_cn, time_points)

            # Random treatment (for some trajectories)
            treatment = None
            if np.random.random() < 0.5:
                treatment = {
                    "type": np.random.choice(["targeted", "chemo", "none"]),
                    "dose": np.random.uniform(0.1, 1.0),
                    "duration": np.random.uniform(10, 50),
                    "start_time": np.random.uniform(20, 50),
                }
                # Apply treatment effect
                if treatment["type"] != "none":
                    start_idx = int(treatment["start_time"] / dt)
                    end_idx = start_idx + int(treatment["duration"] / dt)
                    copy_numbers[start_idx:end_idx] *= 0.5  # Treatment reduces CN

            trajectories.append({
                "initial_state": np.array([initial_cn, 0.0, 1.0]),  # CN, time, active
                "time_points": time_points.tolist(),
                "copy_numbers": copy_numbers.tolist(),
                "treatment": treatment,
            })

        return cls(trajectories, **kwargs)

    @staticmethod
    def _simulate_trajectory(
        initial_cn: float,
        time_points: np.ndarray,
        growth_rate: float = 0.05,
        noise_scale: float = 0.1
    ) -> np.ndarray:
        """
        Simulate ecDNA copy number trajectory.

        Uses a stochastic model with:
        - Exponential growth (oncogene-driven fitness advantage)
        - Binomial segregation noise (random inheritance)
        """
        n_points = len(time_points)
        copy_numbers = np.zeros(n_points)
        copy_numbers[0] = initial_cn

        for i in range(1, n_points):
            dt = time_points[i] - time_points[i - 1]

            # Growth (deterministic)
            growth = growth_rate * copy_numbers[i - 1] * dt

            # Segregation noise (scales with sqrt(CN))
            noise = noise_scale * np.sqrt(copy_numbers[i - 1]) * np.random.randn()

            # Update
            copy_numbers[i] = max(0, copy_numbers[i - 1] + growth + noise)

        return copy_numbers


class VulnerabilityDataset(Dataset):
    """
    Dataset for therapeutic vulnerability discovery (Module 3: VulnCausal).

    Combines CRISPR screening data with ecDNA status for causal inference.
    """

    def __init__(
        self,
        crispr_scores: pd.DataFrame,
        expression: pd.DataFrame,
        ecdna_labels: pd.Series,
        covariates: Optional[pd.DataFrame] = None,
        gene_subset: Optional[List[str]] = None,
    ):
        """
        Initialize vulnerability dataset.

        Args:
            crispr_scores: CRISPR dependency scores (cell_lines x genes)
            expression: Gene expression (cell_lines x genes)
            ecdna_labels: Binary ecDNA status per cell line
            covariates: Optional covariates (lineage, etc.)
            gene_subset: Subset of genes to include
        """
        # Align indices
        common_ids = (
            set(crispr_scores.index)
            & set(expression.index)
            & set(ecdna_labels.index)
        )
        common_ids = sorted(common_ids)

        self.crispr = crispr_scores.loc[common_ids]
        self.expression = expression.loc[common_ids]
        self.ecdna_labels = torch.FloatTensor(
            ecdna_labels.loc[common_ids].values.astype(float)
        )
        self.sample_ids = common_ids

        if gene_subset:
            valid_genes = [g for g in gene_subset if g in self.crispr.columns]
            self.crispr = self.crispr[valid_genes]

        if covariates is not None:
            self.covariates = covariates.loc[common_ids]
        else:
            self.covariates = None

        logger.info(
            f"VulnerabilityDataset: {len(self.sample_ids)} samples, "
            f"{len(self.crispr.columns)} genes"
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample_id = self.sample_ids[idx]

        item = {
            "sample_id": sample_id,
            "crispr_scores": torch.FloatTensor(self.crispr.iloc[idx].values),
            "expression": torch.FloatTensor(self.expression.iloc[idx].values),
            "ecdna_label": self.ecdna_labels[idx],
        }

        if self.covariates is not None:
            # Encode categorical covariates
            cov_values = self._encode_covariates(self.covariates.iloc[idx])
            item["covariates"] = torch.FloatTensor(cov_values)

        return item

    def _encode_covariates(self, cov_row: pd.Series) -> np.ndarray:
        """Encode covariates (one-hot for categorical)."""
        # Simple encoding - in production, use proper one-hot
        encoded = []
        for col in cov_row.index:
            val = cov_row[col]
            if isinstance(val, str):
                # Hash to fixed dimension
                encoded.append(hash(val) % 100 / 100.0)
            else:
                encoded.append(float(val) if pd.notna(val) else 0.0)
        return np.array(encoded)

    def get_gene_names(self) -> List[str]:
        """Get list of gene names."""
        return self.crispr.columns.tolist()

    def get_ecdna_positive_samples(self) -> List[str]:
        """Get ecDNA-positive sample IDs."""
        positive_mask = self.ecdna_labels > 0.5
        return [self.sample_ids[i] for i in range(len(self.sample_ids)) if positive_mask[i]]

    def get_ecdna_negative_samples(self) -> List[str]:
        """Get ecDNA-negative sample IDs."""
        negative_mask = self.ecdna_labels < 0.5
        return [self.sample_ids[i] for i in range(len(self.sample_ids)) if negative_mask[i]]

    @classmethod
    def from_loaders(
        cls,
        depmap_loader,
        cytocell_loader,
        **kwargs
    ) -> "VulnerabilityDataset":
        """
        Create dataset from data loaders.

        Args:
            depmap_loader: DepMapLoader instance
            cytocell_loader: CytoCellDBLoader instance
        """
        # Get CRISPR and expression
        crispr = depmap_loader.crispr
        expression = depmap_loader.expression

        # Get ecDNA labels
        cytocell_data = cytocell_loader.load()
        ecdna_labels = pd.Series(
            (cytocell_data["ecdna_status"] == "positive").astype(int).values,
            index=cytocell_data["depmap_id"]
        )

        # Get covariates
        cell_lines = depmap_loader.cell_lines
        covariates = cell_lines.set_index("DepMap_ID")[["lineage"]]

        return cls(crispr, expression, ecdna_labels, covariates, **kwargs)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
