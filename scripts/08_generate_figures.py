"""Step 8: Generate all figures and tables for the paper.

Usage:
    PYTHONPATH=. python scripts/08_generate_figures.py --config configs/geometry.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def fig_intrinsic_dim(id_results, out_dir):
    """Bar chart: intrinsic dimension per encoder, face vs voice."""
    face_encs = []
    voice_encs = []
    face_ids = []
    voice_ids = []

    for enc, data in id_results.items():
        val = data.get("global_mle")
        if val is None:
            continue
        if enc in ["arcface", "siglip", "dinov2", "clip"]:
            face_encs.append(enc)
            face_ids.append(val)
        else:
            voice_encs.append(enc)
            voice_ids.append(val)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_face = np.arange(len(face_encs))
    x_voice = np.arange(len(voice_encs)) + len(face_encs) + 0.5

    ax.bar(x_face, face_ids, color="#2196F3", label="Face", width=0.6)
    ax.bar(x_voice, voice_ids, color="#FF5722", label="Voice", width=0.6)

    ax.set_xticks(np.concatenate([x_face, x_voice]))
    ax.set_xticklabels(face_encs + voice_encs, rotation=30, ha="right")
    ax.set_ylabel("Intrinsic Dimensionality (MLE)")
    ax.set_title("Intrinsic Dimensionality by Encoder")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "intrinsic_dim_bar.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_curvature_distributions(curv_dir, encoders, dataset, out_dir):
    """Curvature distributions per encoder."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    axes = axes.flatten()

    for i, enc in enumerate(encoders):
        path = os.path.join(curv_dir, f"{enc}_{dataset}_raw.npz")
        if not os.path.exists(path):
            axes[i].set_title(f"{enc}\n(no data)")
            continue
        data = np.load(path)
        mc = data["mean_curvatures"]
        mc = np.clip(mc, np.percentile(mc, 2), np.percentile(mc, 98))

        axes[i].hist(mc, bins=50, alpha=0.7, color="#4CAF50")
        axes[i].set_title(enc)
        axes[i].set_xlabel("Mean curvature")
        if i % 4 == 0:
            axes[i].set_ylabel("Count")

    fig.suptitle("Local Mean Curvature Distributions", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "curvature_distributions.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def fig_cross_modal_heatmaps(cm_results, face_encs, voice_encs, out_dir):
    """4x4 heatmaps for GW distance and CKA."""
    for metric, label, cmap in [
        ("gw_distance", "Gromov-Wasserstein Distance", "YlOrRd"),
        ("cka", "CKA Similarity", "YlGnBu"),
    ]:
        matrix = np.full((len(face_encs), len(voice_encs)), np.nan)
        for i, fe in enumerate(face_encs):
            for j, ve in enumerate(voice_encs):
                key = f"{fe}__{ve}"
                if key in cm_results:
                    val = cm_results[key].get(metric)
                    if val is not None and np.isfinite(val):
                        matrix[i, j] = val

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=".3f", cmap=cmap,
            xticklabels=voice_encs, yticklabels=face_encs,
        )
        ax.set_xlabel("Voice Encoder")
        ax.set_ylabel("Face Encoder")
        ax.set_title(label)

        path = os.path.join(out_dir, f"heatmap_{metric}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def fig_geometry_vs_eer(corr_data, out_dir):
    """Scatter plots: each geometric metric vs. EER (the money figure)."""
    pairs = corr_data["pairs"]
    metrics = ["gw_distance", "spectral_gap", "cka", "id_mismatch"]
    labels = ["GW Distance", "Spectral Gap", "CKA", "ID Mismatch"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, metric, label in zip(axes, metrics, labels):
        x = [p.get(metric) for p in pairs]
        y = [p["eer"] for p in pairs]
        valid = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and np.isfinite(xi)]
        if not valid:
            ax.set_title(f"{label}\n(no data)")
            continue

        x_v, y_v = zip(*valid)
        ax.scatter(x_v, y_v, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)

        # Add regression line
        x_arr = np.array(x_v)
        y_arr = np.array(y_v)
        if len(x_arr) >= 3:
            z = np.polyfit(x_arr, y_arr, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5)

        # Show correlation
        corr_info = corr_data["correlations"].get(metric, {})
        r = corr_info.get("spearman_r", float("nan"))
        pval = corr_info.get("spearman_p", float("nan"))
        ax.set_title(f"{label}\nr={r:.3f}, p={pval:.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("EER")

    fig.suptitle("Geometric Metrics vs. Cross-Modal EER", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "geometry_vs_eer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def generate_tables(id_results, curv_results, cm_results, eer_results, corr_data, out_dir):
    """Generate LaTeX tables."""
    # Table 1: Per-encoder geometric properties
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Geometric properties of pretrained encoders.}",
        r"\label{tab:encoder_geometry}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Encoder & Modality & Ambient Dim & ID (MLE) & Mean Curvature \\",
        r"\midrule",
    ]
    all_encs = list(id_results.keys())
    for enc in all_encs:
        id_data = id_results.get(enc, {})
        curv_data = curv_results.get(enc, {})
        modality = "Face" if enc in ["arcface", "siglip", "dinov2", "clip"] else "Voice"
        ambient = id_data.get("ambient_dim", "—")
        id_mle = id_data.get("global_mle")
        id_str = f"{id_mle:.1f}" if id_mle else "—"
        mc = curv_data.get("mean_curvatures_median")
        mc_str = f"{mc:.4f}" if mc else "—"
        lines.append(f"  {enc} & {modality} & {ambient} & {id_str} & {mc_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    path = os.path.join(out_dir, "table_encoder_geometry.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved {path}")

    # Table 2: Cross-modal metrics + EER
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-modal metrics and EER for all encoder pairs.}",
        r"\label{tab:cross_modal}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Face & Voice & GW Dist. & Spectral Gap & CKA & ID Mismatch & EER \\",
        r"\midrule",
    ]
    for pair_name, cm in sorted(cm_results.items()):
        if "error" in cm:
            continue
        eer_data = eer_results.get(pair_name, {})
        face_enc = cm.get("face_encoder", pair_name.split("__")[0])
        voice_enc = cm.get("voice_encoder", pair_name.split("__")[1])
        gw = cm.get("gw_distance")
        sg = cm.get("spectral_gap")
        cka = cm.get("cka")
        idm = cm.get("id_mismatch")
        eer = eer_data.get("eer")

        def fmt(v):
            return f"{v:.3f}" if v is not None and np.isfinite(v) else "—"

        lines.append(
            f"  {face_enc} & {voice_enc} & {fmt(gw)} & {fmt(sg)} & {fmt(cka)} & {fmt(idm)} & {fmt(eer)} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    path = os.path.join(out_dir, "table_cross_modal.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved {path}")

    # Table 3: Correlation coefficients
    if corr_data and "correlations" in corr_data:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Correlation of geometric metrics with cross-modal EER.}",
            r"\label{tab:correlations}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Metric & Spearman $\rho$ & $p$-value & 95\% CI \\",
            r"\midrule",
        ]
        for metric, data in corr_data["correlations"].items():
            r = data["spearman_r"]
            p = data["spearman_p"]
            ci = data.get("bootstrap_ci_95", [float("nan"), float("nan")])
            lines.append(
                f"  {metric} & {r:.3f} & {p:.4f} & [{ci[0]:.3f}, {ci[1]:.3f}] \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        path = os.path.join(out_dir, "table_correlations.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--dataset", default="voxceleb1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    fig_dir = os.path.join(cfg.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    table_dir = os.path.join(cfg.results_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    all_encoders = list(cfg.encoders.face) + list(cfg.encoders.voice)

    # Load results
    id_path = os.path.join(cfg.results_dir, "intrinsic_dim", f"intrinsic_dim_{args.dataset}.json")
    curv_path = os.path.join(cfg.results_dir, "curvature", f"curvature_{args.dataset}.json")
    cm_path = os.path.join(cfg.results_dir, "cross_modal", f"cross_modal_{args.dataset}.json")
    eer_path = os.path.join(cfg.results_dir, "cross_modal", f"eer_{args.dataset}.json")
    corr_path = os.path.join(cfg.results_dir, "tables", "correlation_analysis.json")

    id_results = json.load(open(id_path)) if os.path.exists(id_path) else {}
    curv_results = json.load(open(curv_path)) if os.path.exists(curv_path) else {}
    cm_results = json.load(open(cm_path)) if os.path.exists(cm_path) else {}
    eer_results = json.load(open(eer_path)) if os.path.exists(eer_path) else {}
    corr_data = json.load(open(corr_path)) if os.path.exists(corr_path) else {}

    # Generate figures
    if id_results:
        fig_intrinsic_dim(id_results, fig_dir)

    if curv_results:
        fig_curvature_distributions(
            os.path.join(cfg.results_dir, "curvature"),
            all_encoders, args.dataset, fig_dir,
        )

    if cm_results:
        fig_cross_modal_heatmaps(
            cm_results,
            list(cfg.encoders.face),
            list(cfg.encoders.voice),
            fig_dir,
        )

    if corr_data and "pairs" in corr_data:
        fig_geometry_vs_eer(corr_data, fig_dir)

    # Generate tables
    generate_tables(id_results, curv_results, cm_results, eer_results, corr_data, table_dir)


if __name__ == "__main__":
    main()
