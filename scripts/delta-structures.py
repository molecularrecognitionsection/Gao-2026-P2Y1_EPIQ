#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
pdb_dir = "./"  # folder containing cluster rep PDBs: BPTU-C1.pdb, EPIQ-C2.pdb, ...

# Ligand -> TTCLUST log mapping
ttclust_logs = {
    "BPTU": "clustering-BPTU.log",
    "EPIQ": "clustering-EPIQ.log",
}

# Colors by ligand (paper-friendly)
LIGAND_COLORS = {
    "EPIQ": "#1f77b4",  # blue
    "BPTU": "#ff7f0e",  # orange
}
DEFAULT_COLOR = "#7f7f7f"

output_csv = "delta_with_cluster_frequency.csv"
output_png = "delta_freq_scatter.png"

# Delta definition
# d1 = V88 - I263
# d2 = T143 - L323
res_pairs = {"d1": (88, 263), "d2": (143, 323)}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def ca_distance(universe: mda.Universe, res1: int, res2: int) -> float:
    """Compute CA–CA distance between two residues."""
    a1 = universe.select_atoms(f"resid {res1} and name CA")
    a2 = universe.select_atoms(f"resid {res2} and name CA")
    if len(a1) == 0 or len(a2) == 0:
        raise ValueError(f"Missing CA atom for residues {res1} or {res2}")
    return float(np.linalg.norm(a1.positions[0] - a2.positions[0]))

def infer_ligand_from_label(pdb_label: str) -> str:
    """From labels like 'BPTU-C1' returns 'BPTU'."""
    parts = re.split(r"[_\-]", pdb_label)
    return parts[0].upper() if parts and parts[0] else pdb_label.upper()

def infer_cluster_id_from_label(pdb_label: str) -> int | None:
    """Extract cluster number from labels like BPTU-C12 -> 12."""
    m = re.search(r"(?:^|[-_])C(\d+)(?:$|[-_])", pdb_label, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_ttclust_log(logfile: str, ligand: str) -> pd.DataFrame:
    """
    Parse TTCLUST log:
      - reads Number of frames
      - reads cluster N and size = X
    Returns:
      pdb_label, ligand, cluster, frequency, freq_pct
    """
    if not os.path.exists(logfile):
        raise FileNotFoundError(f"TTCLUST log not found: {logfile}")

    total_frames = None
    clusters = []
    current_cluster = None

    with open(logfile, "r") as f:
        for raw in f:
            line = raw.strip()

            m_frames = re.match(r"Number of frames\s*:\s*(\d+)", line)
            if m_frames:
                total_frames = int(m_frames.group(1))
                continue

            m_cluster = re.match(r"cluster\s+(\d+)", line, flags=re.IGNORECASE)
            if m_cluster:
                current_cluster = int(m_cluster.group(1))
                continue

            m_size = re.match(r"size\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if m_size and current_cluster is not None:
                size = int(m_size.group(1))
                clusters.append({
                    "pdb_label": f"{ligand}-C{current_cluster}",
                    "ligand": ligand,
                    "cluster": current_cluster,
                    "frequency": size,
                })

    if total_frames is None:
        total_frames = sum(c["frequency"] for c in clusters) if clusters else 1

    df = pd.DataFrame(clusters)
    if len(df) == 0:
        return df.assign(freq_pct=pd.Series(dtype=float))

    df["freq_pct"] = 100.0 * df["frequency"] / float(total_frames)
    return df

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    # 1) Parse TTCLUST logs into one frequency table
    freq_tables = []
    for lig, log in ttclust_logs.items():
        df_lig = parse_ttclust_log(log, lig)
        freq_tables.append(df_lig)
    df_freq = pd.concat(freq_tables, ignore_index=True) if freq_tables else pd.DataFrame()

    # 2) Load PDBs and compute delta per structure
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if len(pdb_files) == 0:
        raise RuntimeError(f"No PDB files found in: {os.path.abspath(pdb_dir)}")

    rows = []
    for pdb_path in pdb_files:
        pdb_name = os.path.basename(pdb_path)
        pdb_label = pdb_name[:-4] if pdb_name.lower().endswith(".pdb") else pdb_name
        ligand = infer_ligand_from_label(pdb_label)
        cluster_id = infer_cluster_id_from_label(pdb_label)

        u = mda.Universe(pdb_path)
        d1 = ca_distance(u, *res_pairs["d1"])
        d2 = ca_distance(u, *res_pairs["d2"])
        delta = d1 - d2

        rows.append({
            "pdb": pdb_name,
            "pdb_label": pdb_label,
            "ligand": ligand,
            "cluster": cluster_id,
            "d1_V88_I263": d1,
            "d2_T143_L323": d2,
            "delta": delta,
        })

    df = pd.DataFrame(rows)

    # 3) Merge frequencies onto delta table
    if len(df_freq) > 0:
        df = df.merge(
            df_freq[["pdb_label", "frequency", "freq_pct"]],
            on="pdb_label",
            how="left"
        )
    else:
        df["frequency"] = np.nan
        df["freq_pct"] = np.nan

    df["frequency"] = df["frequency"].fillna(0).astype(float)
    df["freq_pct"] = df["freq_pct"].fillna(0.0).astype(float)

    # 4) Normalize delta (min-max)
    dmin = df["delta"].min()
    dmax = df["delta"].max()
    if dmax == dmin:
        df["delta_norm"] = 0.5
    else:
        df["delta_norm"] = (df["delta"] - dmin) / (dmax - dmin)

    # 5) NEW: cluster-only labels (C1, C2, ...) for plotting text
    df["cluster_label"] = df["pdb_label"].str.extract(r"(C\d+)", expand=False)

    # 6) Sort by ligand then highest frequency
    df = df.sort_values(by=["ligand", "frequency"], ascending=[True, False]).reset_index(drop=True)

    # 7) Colors by ligand
    df["color"] = df["ligand"].map(LIGAND_COLORS).fillna(DEFAULT_COLOR)

    # 8) Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    # --------------------------------------------------
    # Bubble strip with labels on top of bubbles and stacked legends (ligand on top)
    # --------------------------------------------------
    import matplotlib.patheffects as pe

    plt.figure(figsize=(8.5, 4.5))

    # Map ligand to y-row
    ligand_order = sorted(df["ligand"].unique())
    y_map = {lig: i for i, lig in enumerate(ligand_order)}
    df["y_row"] = df["ligand"].map(y_map)

    # Bubble size scaled by frequency (%)
    size_scale = 25
    sizes = 40 + size_scale * df["freq_pct"]

    # Draw bubbles
    plt.scatter(
        df["delta_norm"],
        df["y_row"],
        c=df["color"],
        s=sizes,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
        zorder=2
    )

    plt.xlabel("Normalized$_{min\\text{-}max}$ Δ (activation)")
    plt.yticks(range(len(ligand_order)), ligand_order)
    plt.ylabel("Ligand")
    plt.title("P2Y1 MD Clusters Receptor Activation Distribution")

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.6, len(ligand_order) - 0.4)

    # Labels centered ON TOP of bubbles
    for _, row in df.iterrows():
        label = row["cluster_label"] if pd.notna(row["cluster_label"]) else ""
        if not label:
            continue

        txt = plt.text(
            row["delta_norm"],
            row["y_row"],
            label,
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
            color="white"
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=1.5, foreground="black"),
            pe.Normal()
        ])

    # --------------------------------------------------
    # Legends stacked on the right OUTSIDE the plot
    # --------------------------------------------------
    from matplotlib.patches import Patch
    ax = plt.gca()

    # Ligand color legend — TOP
    color_handles = [
        Patch(facecolor=color, edgecolor="black", label=lig)
        for lig, color in LIGAND_COLORS.items()
    ]
    legend_ligand = ax.legend(
        handles=color_handles,
        title="Ligand",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.65)
    )
    ax.add_artist(legend_ligand)

    # Bubble size legend — BOTTOM
    size_legend_pcts = [5, 10, 20, 30]
    size_handles = [
        plt.scatter([], [], s=40 + size_scale*pct,
                    facecolor="white", edgecolor="black",
                    label=f"{pct}%")
        for pct in size_legend_pcts
    ]
    ax.legend(
        handles=size_handles,
        title="Cluster size",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.40)
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_png}")






if __name__ == "__main__":
    main()

