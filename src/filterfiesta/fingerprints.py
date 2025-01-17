from oddt.fingerprints import InteractionFingerprint
import pandas as pd
from rdkit import Chem
import oddt
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import numpy as np

from sklearn.utils.multiclass import type_of_target

class Fingerprint:
    def __init__(self,proteinfile,ligands,scores):

        protein_rdkitmol=Chem.rdmolfiles.MolFromPDBFile(proteinfile)
        self.protein = oddt.toolkit.Molecule(protein_rdkitmol)
        self.protein.protein=True

        self.ligands=ligands
        self.scores=scores
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
        help(next(self.protein.residues))
        res_code = ["{}:{}{}".format(res.chain,res.name,res.number) for res in self.protein.residues]
        for res in res_code:
            for ia in ia_type:
                fp_column_descriptor.append("{}-{}".format(res,ia))
        return fp_column_descriptor

    def plif(self):
        standard_resids=[res.idx0 for res in self.protein.residues] # ids of standard residues
        #[res.resid for res in self.protein.residues] this and also res.id both gave errors


        keep_columns=[] #create boolean vector to filter fingerprint columns
        for resid in np.unique(self.protein.atom_dict['resid']): #for each id of all residues
            if resid in standard_resids:
                keep_columns+=[True]*8 #retain the 8 bits of fignerprints corresponding to the standard residue
            else:
                keep_columns+=[False]*8 #discard the 8 bits for non standard residue

        column_descriptor = self.get_residues()
        fp=[]
        for i in tqdm(self.scores["Supplier order"]):
            mol0 = self.ligands[i]
            mol=oddt.toolkit.Molecule(mol0)
            plif = InteractionFingerprint(mol,self.protein)
            plif = plif[keep_columns]

            fp.append(plif)

        print("hasn't crashed yet")
        self.pd_fp_explicit = pd.DataFrame(fp,columns=column_descriptor)
        self.pd_fp_explicit = (self.pd_fp_explicit > 0).astype(int)

        return self.pd_fp_explicit


    def calculate_jaccard(self,reference_vector, fingerprint):
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

        #!!!! moved to the protocol file since it would have always given an error no matter what
        #if len(reference_vector) != self.pd_fp_explicit.shape[1]:
            #raise ValueError("The reference vector length must match the number of columns in the fingerprint, i.e., 8 time the number of residues")

        # Convert the reference vector to a 1D numpy array if it's not already
        #reference_vector = np.array(reference_vector.loc[0].values)
        # Calculate Jaccard scores for each row
        reference_vector = reference_vector.values.flatten()  # Already 2D if it's a DataFrame


        jaccard_scores = []
        for index, row in tqdm(fingerprint.iterrows(),total=len(fingerprint)):

            row = row.values.flatten()

            score = jaccard_score(reference_vector, row, average='binary')
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