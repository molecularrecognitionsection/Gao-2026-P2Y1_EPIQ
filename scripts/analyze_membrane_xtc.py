#!/usr/bin/env python3
"""
analyze_membrane_xtc.py

Membrane equilibration metrics from:
  - PSF (topology; bonds help for some selections)
  - XTC (trajectory; contains per-frame unit cell dimensions)
  - Optional PDB (coordinates/atom naming reference; not required if PSF has coords)

This version **does not read an XSC file**. For NPT runs, box lengths (Lx, Ly, Lz)
are taken directly from each XTC frame (ts.dimensions).

Metrics (time series):
  - Lx, Ly (Å) and Area = Lx*Ly (Å^2)
  - Area per lipid (APL) for upper and lower leaflets using phosphate atoms
    (one P per phospholipid/sphingomyelin residue). Cholesterol is excluded.
  - Thickness = mean z(P_upper) - mean z(P_lower) (Å)
  - Cholesterol order parameter S = 0.5*(3*cos^2(theta)-1) where theta is the
    angle between a chosen sterol axis and the +z axis; reported overall and per leaflet.
  - Protein backbone RMSD  

Outputs:
  - CSV with time series
  - PNG with 5 stacked panels (box, APL, thickness, cholesterol order, protein backbone RMSD)

Notes:
  - This script assumes your trajectory is already wrapped/aligned as desired.
    (Do NOT align inside the analysis loop unless you absolutely need to.)
  - Units: MDAnalysis uses Å for coordinates and box lengths; time is in ns here.

Example:
    python analyze_membrane_xtc.py -p structure.psf -x wrapped.xtc --dt 200 --lipid-resnames "POPC POPS POPE POPI PSM" --out metrics.png --csv metrics.csv
"""

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional
    tqdm = None


# ---------------------------- small utilities ----------------------------

def linear_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    return np.nan if denom == 0 else np.dot(x0, y - y.mean()) / denom


def monotonic_score(y):
    """Mean sign of finite diffs; in [-1, 1]."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return np.nan
    dy = np.diff(y)
    dy = dy[np.abs(dy) > 1e-12]
    if dy.size == 0:
        return 0.0
    return float(np.mean(np.sign(dy)))


def _delta_first_last(y):
    y = np.asarray(y, float)
    if y.size < 2:
        return np.nan
    return float(y[-1] - y[0])

def _delta_from_slope(slope, t):
    # t is time array in ns
    t = np.asarray(t, float)
    if t.size < 2 or not np.isfinite(slope):
        return np.nan
    span = float(t[-1] - t[0])
    return float(slope * span)

def _pf(val, thresh):
    if not np.isfinite(val):
        return "NA"
    return "PASS" if abs(val) <= thresh else "FAIL"

def running_mean(x, window):
    x = np.asarray(x, float)
    if window is None or window <= 1:
        return None
    out = np.full_like(x, np.nan, dtype=float)
    half = window // 2
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out


def block_average(t, y, block_ns):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if t.size == 0 or block_ns is None or block_ns <= 0:
        return None
    t0 = t.min()
    idx = np.floor((t - t0) / block_ns).astype(int)
    blocks = np.unique(idx)
    tc, ym = [], []
    for b in blocks:
        mb = idx == b
        tc.append(t[mb].mean())
        ym.append(y[mb].mean())
    return np.array(tc), np.array(ym)


# ---------------------------- cholesterol axis precompute ----------------------------

_CANDIDATE_AXES = [
    ("O3", "C17"),
    ("C3", "C17"),
    ("O3", "C25"),
    ("C3", "C25"),
    ("O", "C17"),
    ("O", "C25"),
]

def _pick_axis_atom_indices(res, axis_mode="auto", a1_name=None, a2_name=None):
    names = set(a.name for a in res.atoms)
    if axis_mode == "manual":
        if a1_name in names and a2_name in names:
            a1 = res.atoms.select_atoms(f"name {a1_name}")[0]
            a2 = res.atoms.select_atoms(f"name {a2_name}")[0]
            return a1.ix, a2.ix
        return None

    for a1n, a2n in _CANDIDATE_AXES:
        if a1n in names and a2n in names:
            a1 = res.atoms.select_atoms(f"name {a1n}")[0]
            a2 = res.atoms.select_atoms(f"name {a2n}")[0]
            return a1.ix, a2.ix
    return None


def precompute_chol_axes(u, chol_sel, axis_mode="auto", a1_name=None, a2_name=None):
    """Return (chol_residues, axis_index_pairs)."""
    chol_atoms = u.select_atoms(chol_sel)
    if len(chol_atoms) == 0:
        return [], []
    reslist = list(chol_atoms.residues)
    pairs = []
    kept = []
    for res in reslist:
        pair = _pick_axis_atom_indices(res, axis_mode=axis_mode, a1_name=a1_name, a2_name=a2_name)
        if pair is not None:
            kept.append(res)
            pairs.append(pair)
    return kept, pairs


def chol_order_from_pairs(u, axis_pairs, z_split=None):
    """Compute mean S overall and by leaflet, using precomputed axis atom index pairs."""
    if not axis_pairs:
        return np.nan, np.nan, np.nan

    zhat = np.array([0.0, 0.0, 1.0], dtype=float)
    S_all, S_up, S_lo = [], [], []

    # Fast access to positions array
    pos = u.atoms.positions

    for a1_ix, a2_ix in axis_pairs:
        v = pos[a2_ix] - pos[a1_ix]
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        cos_t = float(np.dot(v / nv, zhat))
        S = 0.5 * (3.0 * cos_t * cos_t - 1.0)
        S_all.append(S)

        if z_split is not None:
            if pos[a1_ix, 2] >= z_split:
                S_up.append(S)
            else:
                S_lo.append(S)

    def mean_or_nan(arr):
        return float(np.mean(arr)) if arr else np.nan

    return mean_or_nan(S_all), mean_or_nan(S_up), mean_or_nan(S_lo)


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--psf", required=True, help="Topology PSF")
    ap.add_argument("-x", "--xtc", required=True, help="Trajectory XTC (must contain box per frame)")
    ap.add_argument("--pdb", default=None, help="Optional PDB for coordinates/names (rarely needed)")
    ap.add_argument("--dt", type=float, default=None, help="ps per frame (optional; inferred if possible)")
    ap.add_argument("--skip", type=int, default=0, help="frames to skip at start")
    ap.add_argument("--stride", type=int, default=1, help="analyze every Nth frame")
    ap.add_argument("--runmean", type=int, default=11, help="running mean window (frames); 0/1=off")
    ap.add_argument("--block", type=float, default=10.0, help="block average size in ns; 0=off")


    # Go/No-Go thresholds (defaults are conservative for membrane-protein production readiness)
    ap.add_argument("--thresh-dL", type=float, default=0.5, help="GO threshold for |ΔLx| and |ΔLy| over analyzed window (Å)")
    ap.add_argument("--thresh-dAPL", type=float, default=1.0, help="GO threshold for |ΔAPL| over analyzed window (Å^2)")
    ap.add_argument("--thresh-dThk", type=float, default=0.5, help="GO threshold for |ΔThickness| over analyzed window (Å)")
    ap.add_argument("--thresh-dCholS", type=float, default=0.05, help="GO threshold for |ΔCholS| over analyzed window (unitless)")
    ap.add_argument("--thresh-dRMSD", type=float, default=1.5, help="GO threshold for |ΔBackboneRMSD| over analyzed window (Å)")
    ap.add_argument("--thresh-sRMSD", type=float, default=0.05, help="GO threshold for |RMSD slope| (Å/ns)")
    ap.add_argument("--thresh-mono", type=float, default=0.2, help="GO threshold for |monotonicity| (dimensionless)")
    ap.add_argument("--no-go-check", action="store_true", help="Skip go/no-go evaluation output")

    # Lipid phosphate selection: default targets common phospholipids + sphingomyelin, excludes cholesterol.
    ap.add_argument("--lipid-resnames",
                    default="POPC POPS POPE POPI PSM",
                    help="space-separated resnames treated as 'lipids with phosphate' for APL/thickness")
    ap.add_argument("--pname", default="P", help="phosphate atom name (default P). Use e.g. P1 if needed")

    ap.add_argument("--chol-sel", default="resname CHOL CHL1 CHL ROH CLR",
                    help="cholesterol selection (MDAnalysis syntax)")
    ap.add_argument("--chol-axis", choices=["auto", "manual"], default="auto")
    ap.add_argument("--chol-a1", default=None, help="manual axis atom1 (e.g., O3)")
    ap.add_argument("--chol-a2", default=None, help="manual axis atom2 (e.g., C17)")

    # Protein RMSD (backbone) options
    ap.add_argument("--rmsd-sel", default="protein and backbone", help="selection for RMSD (default: protein and backbone)")
    ap.add_argument("--rmsd-ref", choices=["first", "pdb"], default="first", help="reference for RMSD: first frame of trajectory or a PDB")
    ap.add_argument("--rmsd-ref-pdb", default=None, help="reference PDB if --rmsd-ref pdb")

    ap.add_argument("--csv", default="metrics.csv")
    ap.add_argument("--out", default="metrics.png")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    args = ap.parse_args()

    # --- load universe
    if args.pdb:
        u = mda.Universe(args.psf, args.pdb, args.xtc)
    else:
        u = mda.Universe(args.psf, args.xtc)

    n_total = len(u.trajectory)
    if args.skip >= n_total:
        raise RuntimeError("skip >= total frames; nothing to analyze.")
    if args.stride < 1:
        raise RuntimeError("--stride must be >= 1")

    # --- dt (ps per frame)
    dt_ps = args.dt
    if dt_ps is None:
        # try MDAnalysis dt metadata
        try:
            dt_ps = float(u.trajectory.dt)
        except Exception:
            dt_ps = None

    if dt_ps is None or not np.isfinite(dt_ps) or dt_ps <= 0:
        raise RuntimeError("Could not infer dt from trajectory. Please provide --dt (ps per frame).")

    dt_ns = dt_ps / 1000.0

    # --- selections
    lipid_resnames = args.lipid_resnames.split()
    lipid_sel = f"resname {' '.join(lipid_resnames)} and name {args.pname}"
    psel = u.select_atoms(lipid_sel)
    if len(psel) == 0:
        raise RuntimeError(
            f"No phosphate atoms found for selection: '{lipid_sel}'. "
            f"Check --lipid-resnames and/or --pname."
        )

    # Precompute cholesterol axis pairs once (fast, avoids reselect each frame)
    chol_residues, chol_axis_pairs = precompute_chol_axes(
        u,
        args.chol_sel,
        axis_mode=args.chol_axis,
        a1_name=args.chol_a1,
        a2_name=args.chol_a2,
    )
    if args.chol_axis == "manual" and (args.chol_a1 is None or args.chol_a2 is None):
        raise RuntimeError("--chol-axis manual requires --chol-a1 and --chol-a2")

    # --- frame list to analyze
    frame_indices = list(range(args.skip, n_total, args.stride))
    n = len(frame_indices)
    t = np.arange(n, dtype=float) * (dt_ns * args.stride)


    # --- RMSD setup (protein backbone by default)
    rmsd_sel = args.rmsd_sel
    mobile = u.select_atoms(rmsd_sel)
    if len(mobile) == 0:
        raise RuntimeError(f"No atoms found for RMSD selection: '{rmsd_sel}'")

    if args.rmsd_ref == "pdb":
        if not args.rmsd_ref_pdb:
            raise RuntimeError("--rmsd-ref pdb requires --rmsd-ref-pdb")
        ref_u = mda.Universe(args.psf, args.rmsd_ref_pdb)
        ref_atoms = ref_u.select_atoms(rmsd_sel)
        if len(ref_atoms) != len(mobile):
            raise RuntimeError(
                f"RMSD selection mismatch: mobile has {len(mobile)} atoms, reference has {len(ref_atoms)}. "
                "Ensure the reference PDB corresponds to the same topology and selection."
            )
        ref_u.trajectory[0]
        ref_coords = ref_atoms.positions.copy()
    else:
        # Reference is the first analyzed frame
        u.trajectory[frame_indices[0]]
        ref_coords = mobile.positions.copy()


    # --- allocate arrays
    Lx = np.full(n, np.nan)
    Ly = np.full(n, np.nan)
    Lz = np.full(n, np.nan)
    Area = np.full(n, np.nan)
    apl_u = np.full(n, np.nan)
    apl_l = np.full(n, np.nan)
    thickness = np.full(n, np.nan)
    chol_S = np.full(n, np.nan)
    chol_Su = np.full(n, np.nan)
    chol_Sl = np.full(n, np.nan)

    rmsd_bb = np.full(n, np.nan)
    # --- progress bar
    iterator = frame_indices
    if (not args.no_progress) and tqdm is not None:
        iterator = tqdm(frame_indices, total=n, desc="Analyzing", unit="frame")

    # --- main loop
    for i, fr in enumerate(iterator):
        u.trajectory[fr]
        ts = u.trajectory.ts

        # Box from XTC (per-frame)
        dims = ts.dimensions  # [lx, ly, lz, alpha, beta, gamma]
        if dims is None or len(dims) < 3 or dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
            raise RuntimeError(
                "No valid unit cell found in XTC frame. "
                "Your XTC may not contain box vectors, or the reader didn't load them."
            )

        lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
        Lx[i], Ly[i], Lz[i] = lx, ly, lz
        Area[i] = lx * ly

        # Leaflet split plane from phosphate z-median
        zP = psel.positions[:, 2]
        z_split = float(np.median(zP))
        up_mask = zP >= z_split
        lo_mask = ~up_mask

        # Thickness
        z_up = zP[up_mask]
        z_lo = zP[lo_mask]
        if z_up.size > 0 and z_lo.size > 0:
            thickness[i] = float(z_up.mean() - z_lo.mean())

        # APL: number of unique lipid residues per leaflet (based on P atoms)
        up_resids = np.unique(psel.resids[up_mask])
        lo_resids = np.unique(psel.resids[lo_mask])
        n_up = int(up_resids.size)
        n_lo = int(lo_resids.size)
        if n_up > 0:
            apl_u[i] = Area[i] / n_up
        if n_lo > 0:
            apl_l[i] = Area[i] / n_lo

        # Cholesterol order parameter (overall and per leaflet)
        S_all, S_up, S_lo = chol_order_from_pairs(u, chol_axis_pairs, z_split=z_split)
        chol_S[i], chol_Su[i], chol_Sl[i] = S_all, S_up, S_lo

        # Protein backbone RMSD (Å) with per-frame superposition to reference coords
        # Reports conformational change independent of global translation/rotation.
        rmsd_bb[i] = float(rms.rmsd(mobile.positions, ref_coords, superposition=True))

        if tqdm is not None and hasattr(iterator, "set_postfix") and (not args.no_progress):
            iterator.set_postfix(
                A=f"{Area[i]:.0f}",
                APLu=f"{apl_u[i]:.1f}" if np.isfinite(apl_u[i]) else "nan",
                Thk=f"{thickness[i]:.1f}" if np.isfinite(thickness[i]) else "nan",
            )

    # --- diagnostics
    lx_s = linear_slope(t, Lx)
    ly_s = linear_slope(t, Ly)
    aplu_s = linear_slope(t, apl_u)
    apll_s = linear_slope(t, apl_l)
    th_s = linear_slope(t, thickness)
    S_s = linear_slope(t, chol_S)
    rmsd_s = linear_slope(t, rmsd_bb)

    print("\n=== Drift diagnostics (time in ns) ===")
    print(f"Lx slope: {lx_s:.4g} Å/ns   monotonic: {monotonic_score(Lx):.3f}")
    print(f"Ly slope: {ly_s:.4g} Å/ns   monotonic: {monotonic_score(Ly):.3f}")
    print(f"APL upper slope: {aplu_s:.4g} Å²/ns  monotonic: {monotonic_score(apl_u):.3f}")
    print(f"APL lower slope: {apll_s:.4g} Å²/ns  monotonic: {monotonic_score(apl_l):.3f}")
    print(f"Thickness slope: {th_s:.4g} Å/ns  monotonic: {monotonic_score(thickness):.3f}")
    print(f"Chol S slope: {S_s:.4g} 1/ns  monotonic: {monotonic_score(chol_S):.3f}")
    print(f"Backbone RMSD slope: {rmsd_s:.4g} Å/ns  monotonic: {monotonic_score(rmsd_bb):.3f}")


    if not args.no_go_check:
        span = float(t[-1] - t[0]) if t.size > 1 else float("nan")

        # Actual first->last deltas
        dLx = _delta_first_last(Lx)
        dLy = _delta_first_last(Ly)
        dAPL_u = _delta_first_last(apl_u)
        dAPL_l = _delta_first_last(apl_l)
        dThk = _delta_first_last(thickness)
        dChS = _delta_first_last(chol_S)
        dRMSD = _delta_first_last(rmsd_bb)

        # Deltas implied by slope (useful if series is noisy)
        dLx_s = _delta_from_slope(lx_s, t)
        dLy_s = _delta_from_slope(ly_s, t)
        dAPL_u_s = _delta_from_slope(aplu_s, t)
        dAPL_l_s = _delta_from_slope(apll_s, t)
        dThk_s = _delta_from_slope(th_s, t)
        dChS_s = _delta_from_slope(S_s, t)
        dRMSD_s = _delta_from_slope(rmsd_s, t)

        # Monotonicity
        mLx = abs(monotonic_score(Lx))
        mLy = abs(monotonic_score(Ly))
        mAPL_u = abs(monotonic_score(apl_u))
        mAPL_l = abs(monotonic_score(apl_l))
        mThk = abs(monotonic_score(thickness))
        mChS = abs(monotonic_score(chol_S))
        mRMSD = abs(monotonic_score(rmsd_bb))

        print("\n=== Go/No-Go check (window: {:.2f} ns) ===".format(span))
    print("Thresholds: |ΔL|≤{:.3g} Å, |ΔAPL|≤{:.3g} Å², |ΔThk|≤{:.3g} Å, |ΔCholS|≤{:.3g}, |ΔRMSD|≤{:.3g} Å, |RMSD slope|≤{:.3g} Å/ns, |mono|≤{:.3g}".format(
        args.thresh_dL, args.thresh_dAPL, args.thresh_dThk, args.thresh_dCholS, args.thresh_dRMSD, args.thresh_sRMSD, args.thresh_mono
    ))

    rows = [
        ("Lx", dLx, dLx_s, args.thresh_dL, mLx),
        ("Ly", dLy, dLy_s, args.thresh_dL, mLy),
        ("APL upper", dAPL_u, dAPL_u_s, args.thresh_dAPL, mAPL_u),
        ("APL lower", dAPL_l, dAPL_l_s, args.thresh_dAPL, mAPL_l),
        ("Thickness", dThk, dThk_s, args.thresh_dThk, mThk),
        ("Chol S", dChS, dChS_s, args.thresh_dCholS, mChS),
        ("Backbone RMSD", dRMSD, dRMSD_s, args.thresh_dRMSD, mRMSD),
    ]

    # Print a compact table
    print("{:<14s} {:>12s} {:>12s} {:>8s} {:>8s} {:>6s}".format("Metric", "Δ (data)", "Δ (slope)", "|mono|", "limit", "PF"))
    overall_pf = "PASS"
    for name, d_data, d_slope, lim, mono in rows:
        # Pass if both |Δ(data)| and |Δ(slope)| are within limit, and |mono| within mono threshold
        pf_delta = (np.isfinite(d_data) and abs(d_data) <= lim) or (np.isfinite(d_slope) and abs(d_slope) <= lim)
        pf_mono = (np.isfinite(mono) and mono <= args.thresh_mono)
        pf = "PASS" if (pf_delta and pf_mono) else "FAIL"
        if name == "Backbone RMSD":
            # Also enforce RMSD slope magnitude threshold
            if np.isfinite(rmsd_s) and abs(rmsd_s) > args.thresh_sRMSD:
                pf = "FAIL"
        if pf == "FAIL":
            overall_pf = "FAIL"
        print("{:<14s} {:>12.4g} {:>12.4g} {:>8.3f} {:>8.3g} {:>6s}".format(
            name,
            d_data if np.isfinite(d_data) else float("nan"),
            d_slope if np.isfinite(d_slope) else float("nan"),
            mono if np.isfinite(mono) else float("nan"),
            lim,
            pf
        ))

    print("Overall:", overall_pf)
    if overall_pf == "PASS":
        print("Go/No-Go verdict: GO (metrics within thresholds for this window)")
    else:
        print("Go/No-Go verdict: NO-GO (one or more metrics exceed thresholds)")

    # --- smoothing / block averages
    Lx_rm = running_mean(Lx, args.runmean) if args.runmean and args.runmean > 1 else None
    Ly_rm = running_mean(Ly, args.runmean) if args.runmean and args.runmean > 1 else None
    apl_u_rm = running_mean(apl_u, args.runmean) if args.runmean and args.runmean > 1 else None
    apl_l_rm = running_mean(apl_l, args.runmean) if args.runmean and args.runmean > 1 else None
    th_rm = running_mean(thickness, args.runmean) if args.runmean and args.runmean > 1 else None
    S_rm = running_mean(chol_S, args.runmean) if args.runmean and args.runmean > 1 else None
    rmsd_rm = running_mean(rmsd_bb, args.runmean) if args.runmean and args.runmean > 1 else None

    blocks = {}
    if args.block and args.block > 0:
        blocks["Lx"] = block_average(t, Lx, args.block)
        blocks["Ly"] = block_average(t, Ly, args.block)
        blocks["APL_u"] = block_average(t, apl_u, args.block)
        blocks["APL_l"] = block_average(t, apl_l, args.block)
        blocks["Thk"] = block_average(t, thickness, args.block)
        blocks["CholS"] = block_average(t, chol_S, args.block)

    # --- CSV
    header = "time_ns,Lx_A,Ly_A,Lz_A,Area_A2,APL_upper_A2,APL_lower_A2,Thickness_A,CholS,CholS_upper,CholS_lower\n"
    with open(args.csv, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(
                f"{t[i]:.6f},{Lx[i]:.6f},{Ly[i]:.6f},{Lz[i]:.6f},{Area[i]:.6f},"
                f"{apl_u[i]:.6f},{apl_l[i]:.6f},{thickness[i]:.6f},"
                f"{chol_S[i]:.6f},{chol_Su[i]:.6f},{chol_Sl[i]:.6f}\n"
            )
    print(f"Wrote CSV: {args.csv}")

    # --- Plot
    fig = plt.figure(figsize=(13, 15))

    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(t, Lx, label=f"Lx (slope={lx_s:.3g} Å/ns)")
    ax1.plot(t, Ly, label=f"Ly (slope={ly_s:.3g} Å/ns)")
    if Lx_rm is not None:
        ax1.plot(t, Lx_rm, linewidth=2, label="Lx run-mean")
        ax1.plot(t, Ly_rm, linewidth=2, label="Ly run-mean")
    if blocks.get("Lx") is not None:
        bx, by = blocks["Lx"]
        ax1.plot(bx, by, marker="o", linestyle="None", label=f"Lx block({args.block} ns)")
        bx, by = blocks["Ly"]
        ax1.plot(bx, by, marker="o", linestyle="None", label=f"Ly block({args.block} ns)")
    ax1.set_ylabel("Box length (Å)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(t, apl_u, label=f"APL upper (slope={aplu_s:.3g} Å²/ns)")
    ax2.plot(t, apl_l, label=f"APL lower (slope={apll_s:.3g} Å²/ns)")
    ax2.axhline(np.nanmean(apl_u), linestyle="--", linewidth=1, label="Mean upper")
    ax2.axhline(np.nanmean(apl_l), linestyle="--", linewidth=1, label="Mean lower")
    if apl_u_rm is not None:
        ax2.plot(t, apl_u_rm, linewidth=2, label="APL upper run-mean")
        ax2.plot(t, apl_l_rm, linewidth=2, label="APL lower run-mean")
    if blocks.get("APL_u") is not None:
        bx, by = blocks["APL_u"]
        ax2.plot(bx, by, marker="o", linestyle="None", label=f"APL upper block({args.block} ns)")
        bx, by = blocks["APL_l"]
        ax2.plot(bx, by, marker="o", linestyle="None", label=f"APL lower block({args.block} ns)")
    ax2.set_ylabel("Area per lipid (Å²)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(t, thickness, label=f"Thickness (slope={th_s:.3g} Å/ns)")
    ax3.axhline(np.nanmean(thickness), linestyle="--", linewidth=1, label="Mean")
    if th_rm is not None:
        ax3.plot(t, th_rm, linewidth=2, label="Thickness run-mean")
    if blocks.get("Thk") is not None:
        bx, by = blocks["Thk"]
        ax3.plot(bx, by, marker="o", linestyle="None", label=f"Thickness block({args.block} ns)")
    ax3.set_ylabel("Bilayer thickness (Å)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(t, chol_S, label=f"Chol S (slope={S_s:.3g} 1/ns)")
    ax4.plot(t, chol_Su, label="Chol S upper")
    ax4.plot(t, chol_Sl, label="Chol S lower")
    ax4.axhline(np.nanmean(chol_S), linestyle="--", linewidth=1, label="Mean")
    if S_rm is not None:
        ax4.plot(t, S_rm, linewidth=2, label="Chol S run-mean")
    if blocks.get("CholS") is not None:
        bx, by = blocks["CholS"]
        ax4.plot(bx, by, marker="o", linestyle="None", label=f"Chol S block({args.block} ns)")
    ax4.set_ylabel("Chol order S")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    ax5.plot(t, rmsd_bb, label=f"Backbone RMSD (slope={rmsd_s:.3g} Å/ns)")
    ax5.axhline(np.nanmean(rmsd_bb), linestyle="--", linewidth=1, label="Mean")
    if rmsd_rm is not None:
        ax5.plot(t, rmsd_rm, linewidth=2, label="RMSD run-mean")
    if blocks.get("RMSD") is not None:
        bx, by = blocks["RMSD"]
        ax5.plot(bx, by, marker="o", linestyle="None", label=f"RMSD block({args.block} ns)")
    ax5.set_ylabel("RMSD (Å)")
    ax5.set_xlabel("Time (ns)")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    # Optional: keep logs clean without hiding everything.
    warnings.filterwarnings("ignore", message=".*Bio.Application.*", category=Warning)
    main()
