import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm
from numba import njit


@njit
def calculate_surface(coords, radii, grid, x, y, z, threshold):
    num_atoms = coords.shape[0]

    # Precompute the 3D grid coordinates once
    grid_coords = np.zeros(3)

    for i in range(len(x)):
        grid_coords[0] = x[i]
        for j in range(len(y)):
            grid_coords[1] = y[j]
            for k in range(len(z)):
                grid_coords[2] = z[k]

                # Compute squared differences for each atom and the current grid point
                min_dist = np.inf
                for atom_idx in range(num_atoms):
                    dx = coords[atom_idx, 0] - grid_coords[0]
                    dy = coords[atom_idx, 1] - grid_coords[1]
                    dz = coords[atom_idx, 2] - grid_coords[2]

                    dist_sq = dx * dx + dy * dy + dz * dz
                    dist = np.sqrt(dist_sq) - radii[atom_idx]

                    # Track the minimum distance
                    if dist < min_dist:
                        min_dist = dist

                # Update the grid based on the threshold condition
                grid[i, j, k] = np.abs(min_dist) < threshold

    return grid

class Contacts:
    def __init__(self,rec_path, lig_path):
        self.receptor = Chem.MolFromPDBFile(rec_path)
        self.ligands = Chem.SDMolSupplier(lig_path)
        self.vdwradii = {6: 1.7, 1: 1.2, 7: 1.55, 8: 1.52}
        self.overlaps=[]
        self.reloverlap=[]

    def run(self):
        min_coords, max_coords = self._get_bounding_box(self.ligands)
        grid, x, y, z = self._create_grid(min_coords, max_coords)
        receptor_grid = self._populate_grid_with_molecule(self.receptor, np.copy(grid), x, y, z)
        self.overlaps=[]
        self.reloverlap=[]

        for lig in tqdm(self.ligands, total=len(self.ligands)):
            ligand_grid=self._populate_grid_with_molecule(lig,np.copy(grid),x,y,z)
            self.overlaps.append(self._compute_grid_overlap(receptor_grid, ligand_grid))
            self.reloverlap.append(self._compute_grid_overlap(receptor_grid, ligand_grid)/np.sum(ligand_grid))

        return (self.overlaps,self.reloverlap)

    def _get_bounding_box(self,ligands, padding=0.0):
        conformers= [mol.GetConformer() for mol in tqdm(ligands)]
        print("Calculating bounding box of ligands...")
        mins=[]
        maxs=[]
        for m,mol in tqdm(enumerate(conformers)):
            coords=mol.GetPositions()
            mins.append(coords.min(axis=0))
            maxs.append(coords.max(axis=0))

        min_coords = np.min(np.vstack(mins), axis=0)
        max_coords = np.max(np.vstack(maxs), axis=0)
        return min_coords - padding, max_coords + padding


    def _create_grid(self, min_coords, max_coords, spacing=.5):
        x = np.arange(min_coords[0], max_coords[0], spacing)
        y = np.arange(min_coords[1], max_coords[1], spacing)
        z = np.arange(min_coords[2], max_coords[2], spacing)
        grid = np.zeros((len(x), len(y), len(z)))
        print(f"Created grid with {np.prod(grid.size)} points")
        return grid, x, y, z

    def _populate_grid_with_molecule(self,mol, grid,x,y,z, threshold=0.25):
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        radii = np.array([self.vdwradii.get(atom.GetAtomicNum(), 1.5)  for atom in mol.GetAtoms()])
        return calculate_surface(coords,radii,grid, x,y,z,threshold)


    def _compute_grid_overlap(self,a,b):
        return np.sum(a * b)