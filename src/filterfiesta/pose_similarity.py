import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolAlign import CalcRMS # !!! changed AllChem to (alleged) correct submodule containing CalcRMS
from rdkit.Geometry import Point3D
import numpy as np
from tqdm import tqdm

class Similarity:
    def __init__(self, ligands,scores):
        self.ligands=ligands
        self.scores=scores
        # Initialize "Group RMSD" column in dataframe
        self.scores["Group RMSD"]=pd.NA

    def groupByN(self, N=10): # !!! N=10 is too restrictive, assumes compounds are ordered by name
        """
        Groups ligands from `self.ligands` into batches of size N, computes the RMS distance for each ligand
        in a group relative to the average coordinates of the group, and updates `self.scores` with the mean RMS
        distance of each group. Both `self.scores` and `self.ligands` must be of the same length and represent
        the same molecules in the same order.

        Process:
        1. The ligands are divided into groups of N consecutive molecules from `self.ligands`.
        2. For each group of ligands, the RMS distance of each ligand to the average ligand (based on their coordinates)
        is calculated.
        3. The mean RMS distance of each group is computed.
        4. The mean RMS value for each group is added to the corresponding index in `self.scores`.

        Parameters:
        - N (int): The number of ligands to group together for RMS distance calculation.

        Notes:
        - `self.ligands` and `self.scores` must have the same length, with molecules in the same order.
        - The RMS distance measures how much a ligand's coordinates deviate from the average coordinates of the group.
        """

        # Create a dataframe with all the unique molecule titles, each with the number of occurences in score dataframe: it corresponds to the number of conformers for each molecule
        size_df = self.scores.groupby(self.scores["Title"].tolist(),as_index=False).size().sort_values('index', ascending=True, ignore_index=True, key=lambda s: s.map(lambda x: int(x.split("_")[-1])))


        RMSDs=[]
        print(f"Calculating pose similarities for each group of conformers...")
        for i in tqdm(range(len(size_df))):
            Mol_RMSD=[]
            MolGroup=[]

            # Work on N ligands at a time, should all be conformers of the same molecule
            for j in range(size_df["size"][i]):
                MolGroup.append(self.ligands[next(iter(self.scores["Supplier order"]))]) # uses the indexes of the supplier, but ordered with respect to molecule title

            # Calculate the average atomic positions across the 10 conformers
            conformers=[mol.GetConformer() for mol in MolGroup]
            positions= np.stack([conf.GetPositions() for conf in conformers])
            average_pos=np.mean(positions,axis=0)

            # Create a reference molecule with average atom positions
            reference_molecule= Chem.Mol(MolGroup[0])
            reference_conf=reference_molecule.GetConformer()
            for atomIndex in range(reference_molecule.GetNumAtoms()):
                x, y, z = average_pos[atomIndex]
                reference_conf.SetAtomPosition(atomIndex, Point3D(x, y, z))

            # Calculate RMSD between each N-th ligand and the average reference molecule
            for mol in MolGroup:
                rmsd = CalcRMS(mol,reference_molecule)
                Mol_RMSD.append(rmsd)

            # Average over N calculated RMSDs
            for j in range(size_df["size"][i]):
                RMSDs.append(np.mean(Mol_RMSD))
            RMSD_group=[]
            MolGroup=[]
        self.scores["Group RMSD"]=RMSDs


    def writeBestScore(self, ScoreColumnName="score", MolColumn="Title", cutoff=1, ascending=True,key=None):
        """
        Writes the best scoring ligands to an SDF file and their corresponding scores to a CSV file.

        Parameters:
        - sdf_path (str): Path to save the filtered ligands in SDF format.
        - score_path (str): Path to save the scores of the filtered ligands in CSV format.
        - ScoreColumnName (str): The name of the column containing the scores. Default is "score".
        - MolColumn (str): The name of the column used to identify ligands (e.g., molecule titles). Default is "Title".
        - ascending (bool): Whether to sort scores in ascending order. Default is True.
        - key (callable) : Apply the key function to the values before sorting by Molecule name at the last step.

        Steps:
        1. Sorts `self.scores` by `ScoreColumnName` in the specified order (ascending/descending).
        2. Drops duplicate ligands based on `MolColumn`, keeping only the best score for each unique molecule.
        3. Writes the filtered scores to the specified `score_path` as a CSV file.
        4. Writes the corresponding ligands to the specified `sdf_path` in SDF format.
        """
        # Create a copy of scores to work with
        bestscore = self.scores.copy()
        bestscore.reset_index(inplace=True)

        # Sort by the score and remove duplicates based on the molecule title
        bestscore.sort_values(ScoreColumnName, inplace=True, ascending=ascending)
        bestscore.drop_duplicates(subset=MolColumn, inplace=True)
        bestscore.sort_values(MolColumn, inplace=True, ascending=ascending,key=key)

        # Update dataframe with only molecules within selected RMSD cutoff
        bestscore = bestscore[(bestscore["Group RMSD"]<cutoff)]
        # !!! removed score save
        # Save the best scores to a CSV file
        #print(f"Writing file: {score_path}")
        #bestscore.to_csv(score_path, index=False)

        # Write the corresponding ligands to the SDF file
        # !!! removed sdf save
        #writer = Chem.SDWriter(sdf_path)
        #print(f"Writing file: {sdf_path}")
        #for i in bestscore.index:
            #m = Chem.rdmolops.AddHs(self.ligands[i],addCoords=True) # !!! added hydrogens before saving molecules
            #writer.write(m)

        #writer.close()

        return bestscore