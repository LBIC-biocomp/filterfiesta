from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina # type: ignore

class Cluster:
    def __init__(self,path):
        suppl = Chem.SDMolSupplier(path)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024)
        print("Generating Morgan fingerprints...")
        self.fps = [morgan_gen.GetFingerprint(mol) for mol in tqdm(suppl) if mol is not None]

    def calculate_cluster(self,cutoff=0.2):
    # first generate the distance matrix:
        dm = []
        number_of_fps = len(self.fps)
        for i in tqdm(range(1,number_of_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(self.fps[i],self.fps[:i])
            dm.extend([1-x for x in sims])
        self.cs = Butina.ClusterData(dm,number_of_fps,cutoff,isDistData=True)
        return self.cs

    def cluster(self,cutoff=0.2):
        self.calculate_cluster(cutoff)
        cluster_number=np.zeros((len(self.fps)))
        cluster_centroid=np.zeros((len(self.fps)))
        for i,cluster in enumerate(self.cs):
            for molecule in cluster:
                cluster_number[molecule]=i
            cluster_centroid[cluster[0]]=1
        return (cluster_number,cluster_centroid)