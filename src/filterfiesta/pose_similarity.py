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

    def groupByN(self, N=10):
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
        RMSDs=[]
        print(f"Calculating pose similarities for each group of {N} molecules...")
        for i in tqdm(range(len(self.scores)//N)):
            Mol_RMSD=[]
            MolGroup=[]
            for j in range(N):
                MolGroup.append(next(self.ligands))

            conformers=[mol.GetConformer() for mol in MolGroup]
            positions= np.stack([conf.GetPositions() for conf in conformers])
            average_pos=np.mean(positions,axis=0)



            reference_molecule= Chem.Mol(MolGroup[0])
            reference_conf=reference_molecule.GetConformer()
            for atomIndex in range(reference_molecule.GetNumAtoms()):
                x, y, z = average_pos[atomIndex]
                reference_conf.SetAtomPosition(atomIndex, Point3D(x, y, z))

            for mol in MolGroup:
                rmsd = CalcRMS(mol,reference_molecule)
                Mol_RMSD.append(rmsd)

            for j in range(N):
                RMSDs.append(np.mean(Mol_RMSD))
            RMSD_group=[]
            MolGroup=[]
        self.scores["Group RMSD"]=RMSDs

    def groupByName(self, name_column):
        """
        Groups ligands from `self.ligands` into batches of molecules with the same name, computes the RMS distance for each ligand
        in a group relative to the average coordinates of the group, and updates `self.scores` with the mean RMS
        distance of each group. Both `self.scores` and `self.ligands` must be of the same length and represent
        the same molecules in the same order.

        Process:
        1. The ligands are divided into groups of molecules from `self.ligands`.
        2. For each group of ligands, the RMS distance of each ligand to the average ligand (based on their coordinates)
        is calculated.
        3. The mean RMS distance of each group is computed.
        4. The mean RMS value for each group is added to the corresponding index in `self.scores`.

        Parameters:
        - name_column (str):  The column containing mol names used to group together for RMS distance calculation.

        Notes:
        - `self.ligands` and `self.scores` must have the same length, with molecules in the same order.
        - The RMS distance measures how much a ligand's coordinates deviate from the average coordinates of the group.
        """
        names_df=self.scores[name_column].value_counts()
        names=names_df.index
        self.scores["Group RMSD"]=pd.NA
        print(f"Calculating pose similarities for {len(names_df)} unique molecules...")
        for i in tqdm(range(len(names_df))):
            filtered_scores=self.scores[self.scores[name_column]==names[i]]
            ids=filtered_scores["Supplier order"]
            Mol_RMSD=[]
            MolGroup=[]
            for id_order in ids:
                MolGroup.append(self.ligands[id_order])

            conformers=[mol.GetConformer() for mol in MolGroup]
            positions= np.stack([conf.GetPositions() for conf in conformers])
            average_pos=np.mean(positions,axis=0)



            reference_molecule= Chem.Mol(MolGroup[0])
            reference_conf=reference_molecule.GetConformer()
            for atomIndex in range(reference_molecule.GetNumAtoms()):
                x, y, z = average_pos[atomIndex]
                reference_conf.SetAtomPosition(atomIndex, Point3D(x, y, z))

            for mol in MolGroup:
                rmsd = CalcRMS(mol,reference_molecule)
                Mol_RMSD.append(rmsd)

            for index in filtered_scores["Group RMSD"].index:
                self.scores.loc[index, "Group RMSD"]=np.mean(Mol_RMSD)



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