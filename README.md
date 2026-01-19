# ECLIPSE

**E**xtrachromosomal **C**ircular DNA **L**earning for **I**ntegrated **P**rediction of **S**ynthetic-lethality and **E**xpression

A computational framework for predicting ecDNA formation, modeling evolutionary dynamics, and identifying therapeutic vulnerabilities in cancer.

## Overview

Extrachromosomal DNA (ecDNA) represents a paradigm shift in cancer evolution:
- Present in ~30% of cancers across 39 tumor types
- Drives oncogene amplification and treatment resistance
- Associated with significantly worse patient outcomes (HR ~2.0)

ECLIPSE addresses the critical gap in computational tools for ecDNA research through three integrated modules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ECLIPSE FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐     │
│  │   MODULE 1:       │   │   MODULE 2:       │   │   MODULE 3:       │     │
│  │   ecDNA-Former    │──▶│   CircularODE     │──▶│   VulnCausal      │     │
│  │                   │   │                   │   │                   │     │
│  │ Predict ecDNA     │   │ Model ecDNA       │   │ Identify causal   │     │
│  │ formation from    │   │ evolutionary      │   │ therapeutic       │     │
│  │ genomic context   │   │ dynamics          │   │ vulnerabilities   │     │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Module 1: ecDNA-Former
- **Topological Deep Learning**: Hierarchical graph transformer over Hi-C chromatin contact maps
- **DNA Language Model Integration**: Pre-trained sequence encoders (Nucleotide Transformer)
- **Fragile Site Attention**: First model to explicitly encode chromosomal fragile sites
- **Multi-task Prediction**: ecDNA formation probability + oncogene content

### Module 2: CircularODE
- **Physics-Informed Neural SDE**: Incorporates ecDNA segregation biology
- **Treatment-Conditioned Dynamics**: Models evolution under therapeutic pressure
- **Resistance Prediction**: Probabilistic forecasting of treatment resistance

### Module 3: VulnCausal
- **Causal Representation Learning**: Disentangles ecDNA effects from confounders
- **Invariant Risk Minimization**: Finds context-independent vulnerabilities
- **Do-Calculus Integration**: Formal causal inference for synthetic lethality

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/eclipse.git
cd eclipse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install ECLIPSE
pip install -e .
```

## Data Sources

ECLIPSE uses publicly available data:

| Source | Data Type | Access | URL |
|--------|-----------|--------|-----|
| AmpliconRepository | ecDNA annotations | Open | ampliconrepository.org |
| CytoCellDB | Cell line ecDNA status | Open | NAR Cancer 2024 |
| DepMap | CRISPR screens, expression | Open | depmap.org |
| 4D Nucleome | Hi-C contact maps | Open | data.4dnucleome.org |
| HumCFS | Fragile sites | Open | webs.iiitd.edu.in/raghava/humcfs |

### Download Data

```python
from src.data import DataDownloader

downloader = DataDownloader("data")
downloader.download_all(skip_large=True)  # Skip Hi-C for quick start
```

## Quick Start

### Predict ecDNA Formation

```python
from src.models import ECDNAFormer
from src.data import ECDNADataset

# Load model
model = ECDNAFormer.from_pretrained("checkpoints/ecdna_former.pt")

# Predict
outputs = model(
    sequence_features=seq_features,
    topology_features=topo_features,
)
print(f"ecDNA probability: {outputs['formation_probability'].item():.3f}")
print(f"Predicted oncogenes: {outputs['oncogene_probabilities']}")
```

### Model ecDNA Dynamics

```python
from src.models import CircularODE

model = CircularODE()

# Simulate evolution
trajectory = model(
    initial_state=torch.tensor([[50.0, 0.0, 1.0]]),  # CN, time, active
    time_points=torch.linspace(0, 100, 101),
    treatment_info={"categories": torch.tensor([0])},  # Targeted therapy
)
print(f"Resistance probability: {trajectory['resistance_probability'].item():.3f}")
```

### Discover Vulnerabilities

```python
from src.models import VulnCausal
from src.data import VulnerabilityDataset

model = VulnCausal()

# Find ecDNA-specific vulnerabilities
vulnerabilities = model.discover_vulnerabilities(
    expression=expr_data,
    crispr_scores=crispr_data,
    ecdna_labels=ecdna_status,
    environments=lineages,
    top_k=50,
)

for v in vulnerabilities[:10]:
    print(f"Gene {v['gene_id']}: effect={v['causal_effect']:.3f}, "
          f"specificity={v['specificity']:.3f}")
```

### Full Patient Stratification

```python
from src.models import ECLIPSE

model = ECLIPSE.from_pretrained("checkpoints/eclipse.pt")

stratification = model.stratify_patient(
    patient_id="PATIENT_001",
    genomic_data={
        "sequence_features": seq_features,
        "topology_features": topo_features,
        "expression": expression,
        "crispr_scores": crispr_scores,
    },
)

print(f"Risk level: {stratification.risk_level}")
print(f"ecDNA probability: {stratification.ecdna_formation_probability:.3f}")
print(f"Treatment considerations: {stratification.treatment_considerations}")
```

## Training

### Train ecDNA-Former

```python
from src.training import ECDNAFormerTrainer
from src.data import ECDNADataset, create_dataloader

# Create dataset
dataset = ECDNADataset.from_loaders(...)
train_loader = create_dataloader(dataset, batch_size=32)

# Train
trainer = ECDNAFormerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
)
trainer.train(num_epochs=100)
```

## Project Structure

```
eclipse/
├── data/                          # Downloaded data
│   ├── amplicon_repository/       # ecDNA annotations
│   ├── cytocell_db/              # Cell line ecDNA status
│   ├── depmap/                   # CRISPR, expression, CNV
│   ├── hic/                      # Hi-C contact maps
│   └── supplementary/            # Fragile sites, COSMIC
├── src/
│   ├── data/                     # Data loading and processing
│   │   ├── download.py           # Data downloaders
│   │   ├── loaders.py            # Data loaders
│   │   ├── processing.py         # Feature extraction
│   │   └── datasets.py           # PyTorch datasets
│   ├── models/
│   │   ├── ecdna_former/         # Module 1
│   │   │   ├── sequence_encoder.py
│   │   │   ├── topology_encoder.py
│   │   │   ├── fragile_site_encoder.py
│   │   │   ├── fusion.py
│   │   │   ├── heads.py
│   │   │   └── model.py
│   │   ├── circular_ode/         # Module 2
│   │   │   ├── dynamics.py
│   │   │   ├── treatment.py
│   │   │   └── model.py
│   │   ├── vuln_causal/          # Module 3
│   │   │   ├── causal_encoder.py
│   │   │   ├── invariant_predictor.py
│   │   │   ├── causal_graph.py
│   │   │   ├── intervention.py
│   │   │   └── model.py
│   │   └── eclipse.py            # Unified framework
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── schedulers.py
│   └── utils/                    # Utilities
│       ├── genomics.py
│       ├── graphs.py
│       └── metrics.py
├── requirements.txt
└── README.md
```

## Expected Performance

| Task | Metric | Expected | Baseline |
|------|--------|----------|----------|
| ecDNA Formation | AUROC | 0.80-0.85 | 0.65-0.70 |
| ecDNA Formation | AUPRC | 0.55-0.65 | 0.35-0.40 |
| Oncogene Prediction | Macro-F1 | 0.70-0.75 | 0.50-0.55 |
| Trajectory Prediction | MSE (log CN) | 0.3-0.5 | 0.8-1.2 |
| Vulnerability Ranking | Precision@20 | 0.40-0.50 | 0.15-0.20 |

## Citation

If you use ECLIPSE in your research, please cite:

```bibtex
@article{eclipse2026,
  title={ECLIPSE: Extrachromosomal Circular DNA Learning for Integrated
         Prediction of Synthetic-lethality and Expression},
  author={ECLIPSE Team},
  journal={bioRxiv},
  year={2026}
}
```

## References

Key papers informing this work:

1. Kim H, et al. "Extrachromosomal DNA is associated with oncogene amplification and poor outcome across multiple cancers." *Nature Genetics* 2020.
2. Turner KM, et al. "Extrachromosomal oncogene amplification drives tumour evolution and genetic heterogeneity." *Nature* 2017.
3. Hung KL, et al. "ecDNA hubs drive cooperative intermolecular oncogene expression." *Nature* 2021.
4. Rajkumar U, et al. "CytoCellDB: a comprehensive resource for exploring extrachromosomal DNA in cancer cell lines." *NAR Cancer* 2024.

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
