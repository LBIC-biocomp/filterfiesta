from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina # type: ignore

class Cluster:
    def __init__(self,ligands,scores):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # !!! to separate into its own method
        print("Generating Morgan fingerprints...")
        self.morgan_fingerprints = [morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None]



    def calculate_cluster(self,cutoff=0.2):
        # generate the distance matrix:
        distance_matrix = []
        number_of_fps = len(self.morgan_fingerprints)

        for i in tqdm(range(1,number_of_fps)): # from 1 and not 0 to avoid calculating similarity of a molecule with itself

            # calculate tanimoto similarity values
            similarities = DataStructs.BulkTanimotoSimilarity(self.morgan_fingerprints[i],
                                                              self.morgan_fingerprints[:i]) # [:i] is to avoid calculating two times the same distances

            # from similarities calculate tanimoto distances: distance = (1 - similarity)
            distance_matrix.extend([1-x for x in similarities]) # avoids having a list of lists of progressively longer length, Butina.ClusterData accepts only a monodimensional list
        self.clusters = Butina.ClusterData(distance_matrix,number_of_fps,cutoff,isDistData=True)
        return self.clusters



    def cluster(self,cutoff=0.2):
        distance_matrix = []
        number_of_fps = len(self.morgan_fingerprints)

        for i in tqdm(range(1,number_of_fps)): # from 1 and not 0 to avoid calculating similarity of a molecule with itself

            # calculate tanimoto similarity values
            similarities = DataStructs.BulkTanimotoSimilarity(self.morgan_fingerprints[i],
                                                              self.morgan_fingerprints[:i]) # [:i] is to avoid calculating two times the same distances
            # from similarities calculate tanimoto distances: distance = (1 - similarity)
            distance_matrix.extend([1-x for x in similarities]) # avoids having a list of lists of progressively longer length, Butina.ClusterData accepts only a monodimensional list
        clusters = Butina.ClusterData(distance_matrix,number_of_fps,cutoff,isDistData=True)

        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number=np.zeros((number_of_fps))
        cluster_centroid=np.zeros((number_of_fps))


        for i,cluster in enumerate(clusters):
            for molecule in cluster:
                cluster_number[molecule]=i
            cluster_centroid[cluster[0]]=1

        #for i in range(10): !!! what was the purpose?
            #print(len(clusters[i]))

        return (cluster_number,cluster_centroid)