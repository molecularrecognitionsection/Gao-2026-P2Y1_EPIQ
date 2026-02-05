#!/data/Software/anaconda3/envs/mdanalysis/bin/python3

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import MDAnalysis as mda
from MDAnalysis.transformations import unwrap, center_in_box, wrap
from MDAnalysis.analysis import align

from tqdm.auto import tqdm  # <- progress bar

u = mda.Universe("structure.psf", "output.xtc")
ref = mda.Universe("structure.psf", "structure.pdb")

protein = u.select_atoms("protein")
not_protein = u.select_atoms("not protein")

transforms = [
    unwrap(u.atoms),
    center_in_box(protein, center="geometry"),
    wrap(not_protein, compound="residues"),
]
u.trajectory.add_transformations(*transforms)

# Your slice: frames 4000 to (last-1), step 10  (matches your script)
traj_slice = u.trajectory[2500:-1:10]

# tqdm needs a total; MDAnalysis slices typically support len(...)
total = len(traj_slice)

with mda.Writer("wrapped.xtc", u.atoms.n_atoms) as W:
    for ts in tqdm(traj_slice, total=total, desc="Wrapping+aligning", unit="frame"):
        old_rmsd, new_rmsd = align.alignto(
            u, ref, select="protein and backbone", weights="mass"
        )
        W.write(u.atoms)

