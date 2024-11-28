from oddt.fingerprints import InteractionFingerprint
import pandas as pd
from rdkit import Chem
import oddt
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import numpy as np

class Fingerprint:
    def __init__(self,proteinfile,ligandfile):

        protein_rdkitmol=Chem.rdmolfiles.MolFromPDBFile(proteinfile)
        self.protein = oddt.toolkit.Molecule(protein_rdkitmol)
        self.protein.protein=True

        self.ligands=Chem.rdmolfiles.SDMolSupplier(ligandfile)
        self.pd_fp_explicit = None

    def get_residues(self):

        # CREATE DESCRIPTION FOR FINGERPRINT BITS
        fp_column_descriptor = []
        ia_type = [
        "VDW",
        "AR-FF",
        "AR-EF",
        "HBD",
        "HBA",
        "I+",
        "I-",
        "ME",
        ]
        res_code = ["{}:{}{}".format(res.chain,res.name,res.number) for res in self.protein.residues]
        for res in res_code:
            for ia in ia_type:
                fp_column_descriptor.append("{}-{}".format(res,ia))

        return fp_column_descriptor

    def plif(self, column_descriptor):
        fp=[]
        for mol in tqdm(self.ligands):
            mol=oddt.toolkit.Molecule(mol)
            plif = InteractionFingerprint(mol,self.protein)
            fp.append(plif)

        self.pd_fp_explicit = pd.DataFrame(fp,columns=column_descriptor)
        self.pd_fp_explicit = (self.pd_fp_explicit > 0).astype(int)

        return self.pd_fp_explicit


    def calculate_jaccard(self,reference_vector):
        """
        Calculates the Jaccard score between a binary reference vector and each row of the pd_fp_explicit DataFrame.

        Parameters:
        - reference_vector (list or np.array): A binary vector (0s and 1s) with the same length as the number of columns
                                               in the pd_fp_explicit DataFrame.

        Returns:
        - List of Jaccard scores: One score for each row in the pd_fp_explicit DataFrame.
        """
        if self.pd_fp_explicit is None:
            raise ValueError("Make sure to calculate the fingerprints first.")

        if len(reference_vector) != self.pd_fp_explicit.shape[1]:
            raise ValueError("The reference vector length must match the number of columns in the fingerprint, i.e., 8 time the number of residues")

        # Convert the reference vector to a numpy array if it's not already
        reference_vector = np.array(reference_vector)
        # Calculate Jaccard scores for each row
        jaccard_scores = []
        for index, row in tqdm(self.pd_fp_explicit.iterrows(),total=len(self.pd_fp_explicit)):
            score = jaccard_score(reference_vector, row.values)
            jaccard_scores.append(score)

        return jaccard_scores

    def save(self,path):
        if self.pd_fp_explicit is None:
            raise ValueError("Make sure to calculate the fingerprints first.")

        np.save(path,self.pd_fp_explicit.to_numpy())

    def load(self,path,columns=None):
        array=np.load(path)
        if not columns:
            columns=range(array.shape[1])
        self.pd_fp_explicit=pd.DataFrame(array,columns=columns)