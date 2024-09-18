from filterfiesta.cluster import Cluster
from filterfiesta.fingerprints import Fingerprint
from filterfiesta.contacts import *
import pandas as pd
from rdkit import Chem

class FilterFiesta:
    def __init__(self, csv_path=None, columns=None, receptors=None, ligands=None, scores=None):
        """
        Initializes the Fiesta class. Accepts either:
        1. A CSV path and a list of three column names.
        2. Three lists: one for receptors, one for ligands, and one for scores.

        Parameters:
        - csv_path (str): Path to the CSV file.
        - columns (list of str): List of three column names in the CSV file corresponding to receptors, ligands, and scores.
        - receptors (list): List of receptor file paths.
        - ligands (list): List of ligand file paths.
        - scores (list): List of scores file paths.
        """

        # Validate input
        if csv_path is not None and columns is not None:
            self.load_from_csv(csv_path, columns)
        elif receptors is not None and ligands is not None and scores is not None:
            self.load_from_lists(receptors, ligands, scores)
        else:
            raise ValueError("Provide either a CSV path with column names or three lists (receptors, ligands, scores).")

    def load_from_csv(self,csv_path,columns):
        """
        Loads data from a CSV file given column names for receptors, ligands, and scores.

        Parameters:
        - csv_path (str): Path to the CSV file.
        - columns (list of str): List of three column names in the CSV file.
        """

        df = pd.read_csv(csv_path)[columns]
        if len(columns) != 3:
            raise ValueError("You must provide exactly three column names.")
        self.load_from_lists(df[columns[0]].tolist(),df[columns[1]].tolist(),df[columns[2]].tolist())

    def load_from_lists(self,receptors,ligands,scores):
        self.receptors = []
        self.ligands = []
        self.scores = []
        for receptor_path in receptors:
            receptor=Chem.rdmolfiles.MolFromPDBFile(receptor_path)
            self.receptors.append(receptor)
        for ligand_path in ligands:
            ligand=Chem.rdmolfiles.SDMolSupplier(ligand_path)
            self.ligands.append(ligand)
        for score_path in scores:
            score=pd.read_csv(score_path)
            self.scores.append(score)
