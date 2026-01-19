"""
Data loading and processing modules for ECLIPSE.

Handles:
- AmpliconRepository (ecDNA annotations)
- CytoCellDB (cell line ecDNA status)
- DepMap (CRISPR screens, expression, drug sensitivity)
- 4D Nucleome (Hi-C chromatin contact maps)
- Supplementary data (fragile sites, COSMIC genes)
"""

from .download import DataDownloader
from .loaders import (
    AmpliconRepositoryLoader,
    CytoCellDBLoader,
    DepMapLoader,
    HiCLoader,
    FragileSiteLoader,
)
from .processing import (
    DataProcessor,
    FeatureExtractor,
    SplitGenerator,
)
from .datasets import (
    ECDNADataset,
    DynamicsDataset,
    VulnerabilityDataset,
)

__all__ = [
    "DataDownloader",
    "AmpliconRepositoryLoader",
    "CytoCellDBLoader",
    "DepMapLoader",
    "HiCLoader",
    "FragileSiteLoader",
    "DataProcessor",
    "FeatureExtractor",
    "SplitGenerator",
    "ECDNADataset",
    "DynamicsDataset",
    "VulnerabilityDataset",
]
