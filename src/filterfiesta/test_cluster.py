from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
from array import array
import numpy as np
from rdkit.ML.Cluster import Butina # type: ignore
import os
import pickle
from multiprocessing import Pool

# Read-only data shared with the worker processes. Set ONCE per worker by
# NewCluster._init_worker (instead of being re-pickled with every chunk of
# tasks, which is what happens when the data is carried inside a partial).
_WORKER_FPS = None
_WORKER_CUTOFF = None


'''class NewClusterTest:
    def __init__(self, ligands, scores=None, issimple=False):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # !!! to separate into its own method
        print("Generating Morgan fingerprints...")

        if issimple and scores==None:
            self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(i) for i in tqdm(ligands)])}
        elif not issimple and scores != None:
            self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None])}
        else:
            raise ValueError(f"Incompatible arguments: scores = {scores}, issimple = {issimple}")'''

class NewClusterTest:
    def __init__(self, ligands, scores):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # !!! to separate into its own method
        print("Generating Morgan fingerprints...")

        self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None])}


    @staticmethod
    def _init_worker(fingerprints, cutoff):
        """Pool initializer: runs once in each worker process at startup."""
        global _WORKER_FPS, _WORKER_CUTOFF
        _WORKER_FPS = fingerprints
        _WORKER_CUTOFF = cutoff

    @staticmethod
    def _compute_row(i):
        """Worker function: compute neighbor matches for index i against
        fingerprints[:i]. Reads the fingerprints/cutoff from the per-worker
        globals set by _init_worker, so the only thing sent with each task
        is the integer i. Returns (i, matched_positions) so the caller can
        always tell which row a result belongs to, regardless of the order
        results come back in."""
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(_WORKER_FPS[i], _WORKER_FPS[:i]))
        distances = 1 - similarities
        matched = np.nonzero(distances <= _WORKER_CUTOFF)[0]
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

    @staticmethod
    def resolve_n_workers(n_workers=None):
        """Explicit value wins. Otherwise: SLURM allocation if running under
        SLURM with --cpus-per-task, else the CPUs this process may use, else
        the machine's core count."""
        if n_workers is not None:
            return n_workers
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            return int(slurm_cpus)
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return os.cpu_count()


    def calculate_neighbors_parallel(self, cutoff, number_of_fingerprints, n_workers=None, chunksize=None, checkpoint_path=None):
        """Parallel neighbor counting for self.from_idx_to_fp. Must run on the
        pristine, unpopped from_idx_to_fp (keys 0..n-1 contiguous), since
        rows are addressed by position.

        If checkpoint_path is given, the counts are pickled there as soon as
        they are computed (write-only: this function never loads from it, so
        a stale file from another cutoff/dataset can never be reused silently).
        """
        fingerprints = list(self.from_idx_to_fp.values())
        from_idx_to_neighbors = {i: 0 for i in range(number_of_fingerprints)}

        n_workers = NewClusterTest.resolve_n_workers(n_workers)
        order = NewClusterTest.zigzag_order(number_of_fingerprints)

        if chunksize is None:
            # aim for ~20 chunks per worker, so the pool has room to balance
            # load even with the zig-zag ordering already smoothing things out
            chunksize = max(1, len(order) // (n_workers * 20))

        print(f"Using {n_workers} worker processes, chunksize {chunksize}")

        with Pool(processes=n_workers, initializer=NewClusterTest._init_worker, initargs=(fingerprints, cutoff)) as pool:
            # mininterval: seconds between bar refreshes; keeps SLURM log files readable (lower it for interactive use)
            for i, matched_positions in tqdm(pool.imap_unordered(NewClusterTest._compute_row, order, chunksize=chunksize), total=len(order), mininterval=30):
                from_idx_to_neighbors[i] += len(matched_positions)
                for j in matched_positions:
                    from_idx_to_neighbors[j] += 1

        if checkpoint_path is not None:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(from_idx_to_neighbors, f)

        return from_idx_to_neighbors


    def NewClusterDajeDarko(self, n_workers=None, cutoff=0.6, checkpoint_path=None):
        '''New plan: be more like Darko.
        - Fingerprints are calculated and stored in memory
        - BulkTanimotoSimilarity is used to create a list/array of tuples: (molecule, number_of_neighbours)
        - List is sorted based on neighbours
        - First element: BulkTanimoto on every other element of the list --> if match, remove from list
        - Save first element as cluster centroid of cluster #1
        - Take the next available element and repeat BulkTanimoto --> gets faster each time
        - Profit
        '''

        n = len(self.from_idx_to_fp)
        cluster_counter = 0
        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number = np.zeros(n, dtype=np.int64)
        cluster_centroid = np.zeros(n, dtype=bool)

        # calculate number of neighbors for all molecules (only once as per Butina original implementation), parallelized
        print(f"Counting neighbours for {n} molecules...")
        from_idx_to_neighbors = self.calculate_neighbors_parallel(cutoff, n, n_workers=n_workers, checkpoint_path=checkpoint_path)

        # main loop: iteratively finds cluster centroids, assigns as cluster members all molecules within cutoff, then removes them from the dicts
        while len(self.from_idx_to_fp) > 0:
            if cluster_counter % 1000 == 0:
                print(f"Finding members for cluster {cluster_counter} ({len(self.from_idx_to_fp)} molecules left)...", flush=True)

            # find centroid, i.e. molecule with the highest number of neighbours
            max_pos = len(from_idx_to_neighbors) - 1 - np.argmax(list(from_idx_to_neighbors.values())[::-1])
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

            # remove cluster members from dictionaries (set built ONCE, outside the comprehension)
            members_set = set(cluster_members.tolist())
            self.from_idx_to_fp = {key: fp for key, fp in self.from_idx_to_fp.items() if key not in members_set}
            from_idx_to_neighbors = {key: from_idx_to_neighbors[key] for key in self.from_idx_to_fp.keys()}

            # update arrays with cluster members and cluster centroid
            cluster_number[cluster_members] = cluster_counter
            cluster_number[centroid_idx] = cluster_counter
            cluster_centroid[centroid_idx] = True

            # increase cluster counter
            cluster_counter += 1

        print(f"Done. Found {np.count_nonzero(cluster_centroid)} clusters")

        return (cluster_number, cluster_centroid)