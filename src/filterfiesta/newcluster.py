from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
from array import array
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina # type: ignore
import os
from functools import partial
from multiprocessing import Pool
import time

class NewCluster:
    def __init__(self,ligands,scores=None, issimple=False):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # !!! to separate into its own method
        print("Generating Morgan fingerprints...")

        if issimple and scores==None:
            self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(i) for i in tqdm(ligands)])}
        elif not issimple and scores != None:
            self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None])}
        else:
            raise ValueError(f"Incompatible arguments: scores = {scores}, issimple = {issimple}")

    @staticmethod
    def _compute_row(i, fingerprints, cutoff):
        """Worker function: compute neighbor matches for index i against
        fingerprints[:i]. Pure function - no shared state touched, so this
        is safe to run in a separate process. Returns (i, matched_positions)
        so the caller can always tell which row a result belongs to,
        regardless of the order results come back in."""
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        )
        distances = 1 - similarities
        matched = np.nonzero(distances <= cutoff)[0]
        return i, matched.tolist()

    @staticmethod
    def zigzag_order(n):
        """Interleave small and large i values (1, n-1, 2, n-2, ...) so that
        consecutive tasks - and therefore chunks built from consecutive
        tasks - have roughly constant total cost, instead of ascending order
        which puts all the cheap tasks first and all the expensive tasks last."""
        order = []
        lo, hi = 1, n - 1
        while lo <= hi:
            order.append(lo)
            if hi != lo:
                order.append(hi)
            lo += 1
            hi -= 1
        return order


    def calculate_neighbors_parallel(self, cutoff, number_of_fingerprints, n_workers=None, chunksize=None):
        """Parallel neighbor counting for self.morgan_fingerprints, equivalent
        to the sequential 'calculate number of neighbors for all molecules'
        block in NewClusterDajeDarko. Must run on the pristine, unpopped
        from_idx_to_fp (same precondition as the sequential version)."""
        fingerprints = list(self.from_idx_to_fp.values())
        from_idx_to_neighbors = {i: 0 for i in range(number_of_fingerprints)}

        if n_workers is None:
            n_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            n_workers = min(8, n_workers)

        order = NewCluster.zigzag_order(number_of_fingerprints)

        if chunksize is None:
            # aim for ~20 chunks per worker, so the pool has room to balance
            # load even with the zig-zag ordering already smoothing things out
            chunksize = max(1, len(order) // (n_workers * 20))

        worker = partial(NewCluster._compute_row, fingerprints=fingerprints, cutoff=cutoff)

        with Pool(processes=n_workers) as pool:
            for i, matched_positions in tqdm(pool.imap_unordered(worker, order, chunksize=chunksize), total=len(order)):
                from_idx_to_neighbors[i] += len(matched_positions)
                for j in matched_positions:
                    from_idx_to_neighbors[j] += 1

        return from_idx_to_neighbors


    def NewClusterDajeDarko(self, cutoff=0.2):
        '''New plan: be more like Darko.
        - Fingerprints are calculated and stored in memory
        - BulkTanimotoSimilarity is used to create a list/array of tuples: (molecule, number_of_neighbours)
        - List is sorted based on neighbours
        - First element: BulkTanimoto on every other element of the list --> if match, remove from list
        - Save first element as cluster centroid of cluster #1
        - Take the next available element and repeat BulkTanimoto --> gets faster each time
        - Profit
        '''
        start = time.time()

        n = len(self.from_idx_to_fp)
        cluster_counter = 0
        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number=np.zeros((n))
        cluster_centroid=np.zeros((n))

        # calculate number of neighbors for all molecules (only once as per Butina original implementation), parallelized
        print(f"Counting neighbours for {n} molecules...")
        from_idx_to_neighbors = self.calculate_neighbors_parallel(cutoff, n, n_workers=8)

        # main loop: iteratively finds cluster centroids, assigns as cluster members all molecules within cutoff, then removes them from the dicts
        while len(self.from_idx_to_fp) > 0:
            print(f"Finding members for cluster {cluster_counter}...")

            # find centroid, i.e. molecule with the highest number of neighbours
            max_pos = len(from_idx_to_neighbors) -1 - np.argmax(list(from_idx_to_neighbors.values())[::-1])
            centroid_idx = list(from_idx_to_neighbors)[max_pos]
            centroid_fp = self.from_idx_to_fp[centroid_idx]

            # remove centroid from dicts
            self.from_idx_to_fp.pop(centroid_idx)
            from_idx_to_neighbors.pop(centroid_idx)

            # find cluster members
            similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(centroid_fp, list(self.from_idx_to_fp.values())))
            distances = 1 - similarities
            mask = distances <= cutoff

            # retrieve cluster member array using numpy fancy indexing
            cluster_members = np.asarray(list(self.from_idx_to_fp.keys()), dtype=int)[mask]

            # remove cluster members from dictionaries
            self.from_idx_to_fp = {key:self.from_idx_to_fp[key] for key in self.from_idx_to_fp.keys() if key not in set(cluster_members)}
            from_idx_to_neighbors = {key:from_idx_to_neighbors[key] for key in self.from_idx_to_fp.keys()}

            # update arrays with cluster members and cluster centroid
            cluster_number[cluster_members] = cluster_counter
            cluster_number[centroid_idx] = cluster_counter
            cluster_centroid[centroid_idx] = 1

            # increase cluster counter
            cluster_counter += 1

        print(f"Done. Found {len(np.nonzero(cluster_centroid)[0])} clusters")

        end = time.time()

        duration = end - start

        print("Cluster_number:\n", cluster_number)

        print("Time:", duration)

        return (cluster_number, cluster_centroid, duration)

if __name__ == '__main__':
    import pandas as pd
    from cluster import Cluster

    ligands_df = pd.read_csv("/home/luca/projects/2026_filterfiesta_debugging/filterfiesta/notebooks/cluster_test.smi", sep=" ")
    ligands_smi = ligands_df[ligands_df.columns[0]]

    print("Uploading ligands...")
    ligands = [Chem.MolFromSmiles(smi) for smi in tqdm(ligands_smi)]

    nc = NewCluster(ligands, scores=None, issimple=True)
    c = Cluster(ligands, scores=None, issimple=True)

    new_cluster_number, new_cluster_centroid, new_time = nc.NewClusterDajeDarko(cutoff=0.7)
    del nc

    cluster_number, cluster_centroid, c_time = c.cluster(cutoff=0.7)
    del c

    print("Old time:", c_time)
    print("New time:", new_time)
    print("Difference in cluster numbers:", len(np.nonzero(new_cluster_number - cluster_number)[0]))
    print("Difference in cluster Centroids:", len(np.nonzero(new_cluster_centroid - cluster_centroid)[0]))