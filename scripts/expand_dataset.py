#!/usr/bin/env python3
"""
Expand ecDNA training dataset by merging additional label sources.

Current: CytoCellDB ECDNA='Y' only (123 ecDNA+ in training intersection).

Expansions:
1. Include CytoCellDB ECDNA='P' (Possible) samples with AA confirmation as ecDNA+
2. Merge Kim et al. 2020 FISH-validated ecDNA labels for cell lines in DepMap
3. Remove unlabeled samples (ECDNA=NaN) that are currently treated as negative
4. Report statistics on expanded dataset

Usage:
    python scripts/expand_dataset.py                    # Report only
    python scripts/expand_dataset.py --apply            # Apply and re-extract features
    python scripts/expand_dataset.py --include-possible # Include P samples
    python scripts/expand_dataset.py --labeled-only     # Only use labeled (Y/N/P) samples
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cytocelldb(data_dir: Path):
    """Load CytoCellDB labels."""
    cyto = pd.read_excel(data_dir / "cytocell_db" / "CytoCellDB_Supp_File1.xlsx")
    cyto = cyto.dropna(subset=["DepMap_ID"])
    return cyto


def load_kim2020(data_dir: Path):
    """Load Kim et al. 2020 FISH-validated cell line labels."""
    kim_file = data_dir / "amplicon_repository" / "41588_2020_678_MOESM2_ESM.xlsx"
    if not kim_file.exists():
        logger.warning(f"Kim 2020 file not found: {kim_file}")
        return pd.DataFrame()

    kim2 = pd.read_excel(kim_file, sheet_name="Supplementary Table 2")

    # Cell lines with at least one Circular amplicon = ecDNA+
    ecdna_lines = set(kim2[kim2["FinalClass"] == "Circular"]["Sample"].unique())
    # Cell lines assessed but no Circular = ecDNA-
    all_lines = set(kim2["Sample"].unique())
    non_ecdna_lines = all_lines - ecdna_lines

    logger.info(f"Kim 2020 Table 2: {len(all_lines)} cell lines, "
                f"{len(ecdna_lines)} ecDNA+, {len(non_ecdna_lines)} ecDNA-")
    return kim2, ecdna_lines, non_ecdna_lines


def get_depmap_intersection(data_dir: Path):
    """Get DepMap cell line IDs available for feature extraction."""
    cnv_ids = set(pd.read_csv(data_dir / "depmap" / "copy_number.csv", usecols=[0]).iloc[:, 0])
    expr_ids = set(pd.read_csv(data_dir / "depmap" / "expression.csv", usecols=[0]).iloc[:, 0])
    return cnv_ids & expr_ids


def main():
    parser = argparse.ArgumentParser(description="Expand ecDNA dataset")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--include-possible", action="store_true",
                        help="Include ECDNA='P' samples with AA confirmation as ecDNA+")
    parser.add_argument("--labeled-only", action="store_true",
                        help="Only use samples with ECDNA label (Y/N/P), exclude NaN")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes and re-extract features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    depmap_ids = get_depmap_intersection(data_dir)

    # === Current dataset ===
    cyto = load_cytocelldb(data_dir)
    cyto_in_depmap = cyto[cyto["DepMap_ID"].isin(depmap_ids)]

    logger.info(f"\n{'='*60}")
    logger.info("CURRENT DATASET")
    logger.info(f"{'='*60}")
    logger.info(f"CytoCellDB total rows: {len(cyto)}")
    logger.info(f"  With DepMap_ID: {len(cyto.dropna(subset=['DepMap_ID']))}")
    logger.info(f"  In DepMap CNV+Expr: {len(cyto_in_depmap)}")
    logger.info(f"  ECDNA labels: Y={len(cyto_in_depmap[cyto_in_depmap['ECDNA']=='Y'])}, "
                f"N={len(cyto_in_depmap[cyto_in_depmap['ECDNA']=='N'])}, "
                f"P={len(cyto_in_depmap[cyto_in_depmap['ECDNA']=='P'])}, "
                f"NaN={cyto_in_depmap['ECDNA'].isna().sum()}")

    current_pos = len(cyto_in_depmap[cyto_in_depmap["ECDNA"] == "Y"])
    current_total = len(cyto_in_depmap)
    current_unlabeled = cyto_in_depmap["ECDNA"].isna().sum()
    logger.info(f"\n  Current training setup: {current_pos} ecDNA+ / {current_total} total")
    logger.info(f"  WARNING: {current_unlabeled} samples have NO ecDNA label (treated as negative)")

    # === Expansion 1: Include 'P' (Possible) with AA confirmation ===
    logger.info(f"\n{'='*60}")
    logger.info("EXPANSION 1: Include 'P' (Possible) samples with AA confirmation")
    logger.info(f"{'='*60}")

    p_samples = cyto_in_depmap[cyto_in_depmap["ECDNA"] == "P"]
    p_with_aa = p_samples[p_samples["AA prediction"] == "Y"]
    logger.info(f"  'P' samples in DepMap: {len(p_samples)}")
    logger.info(f"  'P' with AA prediction=Y: {len(p_with_aa)}")
    if len(p_with_aa) > 0:
        logger.info(f"  Cell lines: {p_with_aa['CCLE_Name_Format'].tolist()}")

    # === Expansion 2: Merge Kim 2020 FISH labels ===
    logger.info(f"\n{'='*60}")
    logger.info("EXPANSION 2: Merge Kim et al. 2020 FISH-validated labels")
    logger.info(f"{'='*60}")

    kim_result = load_kim2020(data_dir)
    if isinstance(kim_result, tuple):
        kim2, kim_ecdna_lines, kim_non_ecdna_lines = kim_result

        # Match Kim cell line names to CytoCellDB CCLE_Name_Format
        cyto_name_to_id = {}
        for _, row in cyto_in_depmap.iterrows():
            name = str(row["CCLE_Name_Format"]).split("_")[0].upper()
            cyto_name_to_id[name] = row["DepMap_ID"]

        kim_matches = 0
        kim_new_pos = 0
        for cell_line in kim_ecdna_lines:
            name_upper = cell_line.upper().replace("-", "")
            depmap_id = cyto_name_to_id.get(name_upper)
            if depmap_id:
                current_label = cyto_in_depmap[cyto_in_depmap["DepMap_ID"] == depmap_id]["ECDNA"].values[0]
                kim_matches += 1
                if current_label != "Y":
                    kim_new_pos += 1
                    logger.info(f"  {cell_line} → {depmap_id}: currently {current_label}, Kim says ecDNA+")

        logger.info(f"  Kim ecDNA+ lines matched to DepMap: {kim_matches}")
        logger.info(f"  New ecDNA+ labels (not already Y): {kim_new_pos}")

    # === Expansion 3: Remove unlabeled samples ===
    logger.info(f"\n{'='*60}")
    logger.info("EXPANSION 3: Remove unlabeled (NaN) samples")
    logger.info(f"{'='*60}")

    labeled_only = cyto_in_depmap[cyto_in_depmap["ECDNA"].isin(["Y", "N", "P"])]
    logger.info(f"  Labeled samples in DepMap: {len(labeled_only)}")
    logger.info(f"    Y: {(labeled_only['ECDNA']=='Y').sum()}")
    logger.info(f"    N: {(labeled_only['ECDNA']=='N').sum()}")
    logger.info(f"    P: {(labeled_only['ECDNA']=='P').sum()}")
    logger.info(f"  Removed: {len(cyto_in_depmap) - len(labeled_only)} unlabeled samples")

    # === Summary of proposed expansion ===
    logger.info(f"\n{'='*60}")
    logger.info("PROPOSED EXPANSION SUMMARY")
    logger.info(f"{'='*60}")

    if args.include_possible:
        new_pos = current_pos + len(p_with_aa)
        logger.info(f"  ecDNA+ after including P+AA: {new_pos} (was {current_pos})")
    else:
        new_pos = current_pos

    if args.labeled_only:
        new_total = len(labeled_only)
        if args.include_possible:
            new_neg = len(labeled_only[labeled_only["ECDNA"] == "N"])
            new_p_neg = len(labeled_only[labeled_only["ECDNA"] == "P"]) - len(p_with_aa)
            new_total_neg = new_neg + max(0, new_p_neg)
        else:
            new_total_neg = len(labeled_only[labeled_only["ECDNA"].isin(["N"])])
            new_total = new_pos + new_total_neg + len(labeled_only[labeled_only["ECDNA"] == "P"])
        logger.info(f"  Total (labeled only): {new_total}")
    else:
        new_total = current_total
        logger.info(f"  Total: {new_total}")

    logger.info(f"  Positive rate: {new_pos/max(new_total,1):.1%}")
    logger.info(f"\n  85/15 split:")
    n_val = int(new_total * 0.15)
    n_train = new_total - n_val
    logger.info(f"    Train: ~{n_train}, Val: ~{n_val}")

    # === Apply if requested ===
    if args.apply:
        logger.info(f"\n{'='*60}")
        logger.info("APPLYING CHANGES - Re-extracting features")
        logger.info(f"{'='*60}")

        # Build new label set
        new_labels = cyto.copy()

        if args.include_possible:
            # Upgrade P+AA to Y
            mask = (new_labels["ECDNA"] == "P") & (new_labels["AA prediction"] == "Y")
            n_upgraded = mask.sum()
            new_labels.loc[mask, "ECDNA"] = "Y"
            logger.info(f"  Upgraded {n_upgraded} P+AA samples to Y")

        if args.labeled_only:
            new_labels = new_labels[new_labels["ECDNA"].isin(["Y", "N", "P"])]
            logger.info(f"  Filtered to labeled-only: {len(new_labels)} samples")

        # Save modified CytoCellDB as temporary file for feature extraction
        output_file = data_dir / "cytocell_db" / "CytoCellDB_Supp_File1_expanded.xlsx"
        new_labels.to_excel(output_file, index=False)
        logger.info(f"  Saved expanded labels to {output_file}")
        logger.info(f"  ecDNA+: {(new_labels['ECDNA']=='Y').sum()}")
        logger.info(f"  ecDNA-: {(new_labels['ECDNA']=='N').sum()}")
        logger.info(f"  Possible: {(new_labels['ECDNA']=='P').sum()}")

        logger.info("\n  To re-extract features with expanded labels:")
        logger.info("  1. Back up current features: cp data/features/module1_features_*.npz data/features/backup/")
        logger.info("  2. Edit extract_nonleaky_features.py to load CytoCellDB_Supp_File1_expanded.xlsx")
        logger.info("  3. Run: python scripts/extract_nonleaky_features.py")
        logger.info("  4. Retrain: python main.py train --module former --epochs 200 --patience 30")
    else:
        logger.info("\nRun with --apply to save expanded labels.")
        logger.info("Run with --include-possible --labeled-only --apply for recommended expansion.")


if __name__ == "__main__":
    main()
