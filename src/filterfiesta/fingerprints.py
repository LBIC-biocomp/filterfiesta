from oddt.fingerprints import InteractionFingerprint
import pandas as pd
from rdkit import Chem
import oddt
from tqdm import tqdm

class Fingerprint:
    def __init__(self,proteinfile,ligandfile):
        protein_rdkitmol=Chem.rdmolfiles.MolFromPDBFile(proteinfile)
        self.protein = oddt.toolkit.Molecule(protein_rdkitmol)
        self.protein.protein=True

        self.ligands=Chem.rdmolfiles.SDMolSupplier(ligandfile)
        self.pd_fp_explicit = None

    def PLIF(self):

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


        fp=[]
        for mol in tqdm(self.ligands):
            mol=oddt.toolkit.Molecule(mol)
            PLIF = InteractionFingerprint(mol,self.protein)
            fp.append(PLIF)

        self.pd_fp_explicit = pd.DataFrame(fp,columns=fp_column_descriptor)

        return self.pd_fp_explicit