from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from tqdm import tqdm
from array import array
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina # type: ignore
import time

class Cluster:
    def __init__(self, ligands, scores=None, issimple=False):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=1024) # !!! to separate into its own method
        print("Generating Morgan fingerprints...")
        #self.morgan_fingerprints = [morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None]
        #self.from_idx_to_fp = {i:fp for i,fp in enumerate([morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None])}

        if issimple and scores==None:
            self.morgan_fingerprints = [morgan_gen.GetFingerprint(i) for i in tqdm(ligands)]
        elif not issimple and scores != None:
            self.morgan_fingerprints = [morgan_gen.GetFingerprint(ligands[i]) for i in tqdm(scores["Supplier order"]) if ligands[i] is not None]
        else:
            raise ValueError(f"Incompatible arguments: scores = {scores}, issimple = {issimple}")

    def ClusterData(self, data, nPts, distThresh, isDistData=False, reordering=False):
        """  clusters the data points passed in and returns the list of clusters

        **Arguments**

          - data: a list, tuple, or numpy array of items with the input data
            (see discussion of _isDistData_ argument for the exception)

          - nPts: the number of points to be used

          - distThresh: elements within this range of each other are considered
            to be neighbors

          - isDistData: set this toggle when the data passed in is a
              distance matrix.  The distance matrix should be stored
              in one of two formats: as an nxn NumPy array, or as a
              symmetrically stored list or 1D array generated using a
              similar process to the example below:

                dists = []
                for i in range(nPts):
                  for j in range(i):
                    dists.append( distfunc(i,j) )

        **Returns**

          - a tuple of tuples containing information about the clusters:
             ( (cluster1_elem1, cluster1_elem2, ...),
               (cluster2_elem1, cluster2_elem2, ...),
               ...
             )
             The first element for each cluster is its centroid.

        """
        if isDistData:
            # Check if data is a supported type
            if not isinstance(data, (list, tuple, np.ndarray)):
                raise TypeError(f"Unsupported type for data, {type(data)}")

            # Check if data is a 1D array or list
            if isinstance(data, (list, tuple)) or (isinstance(data, np.ndarray) and data.ndim == 1):
                # Check if data length matches the required number of points
                if len(data) != (nPts * (nPts - 1)) // 2:
                    raise ValueError("Mismatched input data dimension and nPts")

                # Create a distance matrix from the 1D data
                dist_matrix = np.zeros((nPts, nPts))
                idx = np.tril_indices(nPts, -1)
                dist_matrix[idx] = data
                dist_matrix += dist_matrix.T
            else:
                # Check if data is a matrix of the correct shape and use it as distance matrix
                if data.shape != (nPts, nPts):
                    raise ValueError(f"Input data with shape {data.shape} is not a matrix of the required shape {(nPts, nPts)}")
                dist_matrix = data
        else:
            raise ValueError

        # Initialize neighbor lists
        neighbor_lists = [np.where(dist_matrix[i] <= distThresh)[0].tolist() for i in range(nPts)]

        # Sort points by the number of neighbors in descending order
        sorted_indices = [(len(neighbors), idx) for idx, neighbors in enumerate(neighbor_lists)]
        sorted_indices.sort(reverse=True)

        # Initialize clusters and a seen array to keep track of processed points
        clusters = []
        seen = np.zeros(nPts, dtype=bool)

        # Process all candidate clusters that have at least two members
        while sorted_indices and sorted_indices[0][0] > 1:
            _, idx = sorted_indices.pop(0)
            if seen[idx]:
                continue

            # Create a new cluster and mark points as seen
            cluster = [idx]
            seen[idx] = True
            for neighbor in neighbor_lists[idx]:
                if not seen[neighbor]:
                    cluster.append(neighbor)
                    seen[neighbor] = True

            clusters.append(tuple(cluster))

        # Process any remaining single-point clusters
        while sorted_indices:
            _, idx = sorted_indices.pop(0)
            if seen[idx]:
                continue
            clusters.append(tuple([idx]))
        return tuple(clusters)

    def cluster(self,cutoff=0.2):

        start = time.time()

        distance_matrix = []
        number_of_fps = len(self.morgan_fingerprints)

        print("Calculating similarities...")
        for i in tqdm(range(1,number_of_fps)): # from 1 and not 0 to avoid calculating similarity of a molecule with itself

            # calculate tanimoto similarity values
            similarities = DataStructs.BulkTanimotoSimilarity(self.morgan_fingerprints[i],
                                                              self.morgan_fingerprints[:i]) # [:i] is to avoid calculating two times the same distances
            # from similarities calculate tanimoto distances: distance = (1 - similarity)
            distance_matrix.extend([1-x for x in similarities]) # avoids having a list of lists of progressively longer length, Butina.ClusterData accepts only a monodimensional list
        print("Forming clusters...")
        clusters = self.ClusterData(distance_matrix,number_of_fps,cutoff,isDistData=True)

        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number=np.zeros((number_of_fps))
        cluster_centroid=np.zeros((number_of_fps))


        for i,cluster in enumerate(clusters):
            for molecule in cluster:
                cluster_number[molecule]=i
            cluster_centroid[cluster[0]]=1

        end = time.time()

        duration = end - start

        print(f"Done. Found {len(clusters)} clusters")

        print("Cluster_number:\n", cluster_number)

        print("Time:",duration)

        return (cluster_number,cluster_centroid, duration)


    '''def calculate_cluster(self,cutoff=0.2):
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
    '''


    '''def NewCluster(self,cutoff=0.2):
        number_of_fps = len(self.morgan_fingerprints)
        distance_matrix = np.empty(number_of_fps * (number_of_fps - 1) // 2, dtype=bool)

        for i in tqdm(range(1,number_of_fps)): # from 1 and not 0 to avoid calculating similarity of a molecule with itself

            # calculate tanimoto similarity values
            similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(self.morgan_fingerprints[i],
                                                              self.morgan_fingerprints[:i])) # [:i] is to avoid calculating two times the same distances
            # from similarities calculate tanimoto distances: distance = (1 - similarity)
            start = i * (i - 1) // 2
            distance_matrix[start:start + i] = (1 - similarities) <= cutoff # avoids having a list of lists of progressively longer length, Butina.ClusterData accepts only a monodimensional list
            del similarities
        clusters = self.NewClusterData(distance_matrix,number_of_fps,cutoff,isDistData=True)

        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number=np.zeros((number_of_fps))
        cluster_centroid=np.zeros((number_of_fps), dtype=bool)


        for i,cluster in enumerate(clusters):
            for molecule in cluster:
                cluster_number[molecule]=i
            cluster_centroid[cluster[0]]=1
            del cluster

        return (cluster_number,cluster_centroid)

    def _neighbor_lists_from_fingerprints(self, cutoff, ):
        n = len(self.morgan_fingerprints)
        neighbor_lists = ([array('i') for _ in range(n)], [])

        for i in tqdm(range(n)):
            if i > 0:
                similarities = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(self.morgan_fingerprints[i], self.morgan_fingerprints[:i])
                )
                distances = 1 - similarities  # identical arithmetic to the original code
                matched = np.nonzero(distances <= cutoff)[0]
                for j in matched:
                    j = int(j)
                    neighbor_lists[0][j].append(i)
                    neighbor_lists[0][i].append(j)

        lengths = [len(neighbor_lists[i]) for i in range(len(neighbor_lists))]

        neighbor_lists[1].extend(lengths)

        return neighbor_lists

    def _Darko_neighbor_lists_from_fingerprints(self, cutoff):
        n = len(self.morgan_fingerprints)
        neighbor_lists = [(i,0) for i in self.morgan_fingerprints]


        for i in tqdm(range(n)):
            if i > 0:
                similarities = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(neighbor_lists[i][0], neighbor_lists[:i][0])
                )
                distances = 1 - similarities  # identical arithmetic to the original code
                matched = np.nonzero(distances <= cutoff)[0]
                for j in matched:
                    j = int(j)
                    neighbor_lists[j][1] += 1
                    neighbor_lists[i][1] += 1
        return neighbor_lists

    def NewClusterDajeDarko(self,cutoff=0.2):
        New plan: be more like Darko.
        - Fingerprints are calculated and stored in memory
        - BulkTanimotoSimilarity is used to create a list/array of tuples: (molecule, number_of_neighbours)
        - List is sorted based on neighbours
        - First element: BulkTanimoto on every other element of the list --> if match, remove from list
        - Save first element as cluster centroid of cluster #1
        - Take the next available element and repeat BulkTanimoto --> gets faster each time
        - Profit

        n = len(self.from_idx_to_fp)
        # from_idx_to_fp = {i:fp for i,fp in enumerate(self.morgan_fingerprints)}
        cluster_counter = 0
        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids
        cluster_number=np.zeros((n))
        cluster_centroid=np.zeros((n))

        # calculate number of neighbors for all molecules (only once as per Butina original implementation)
        from_idx_to_neighbors = {i:0 for i in self.from_idx_to_fp.keys()}

        for i in list(from_idx_to_neighbors.keys())[1:]:
            similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(self.from_idx_to_fp[i], list(self.from_idx_to_fp.values())[:i]))
            distances = 1 - similarities  # identical arithmetic to the original code
            matched = np.nonzero(distances <= cutoff)[0]
            for j in matched:
                j = int(j)
                from_idx_to_neighbors[j] += 1
                from_idx_to_neighbors[i] += 1

        # main loop: iteratively finds cluster centroids, assigns as cluster members all molecules within cutoff, then removes them from the dicts
        while len(self.from_idx_to_fp) > 0:

            # find centroid, i.e. molecule with the highest number of neighbours
            centroid_idx = list(from_idx_to_neighbors)[np.argmax(list(from_idx_to_neighbors.values()))]
            centroid_fp = self.from_idx_to_fp[centroid_idx]

            # remove centroid from dicts
            self.from_idx_to_fp.pop(centroid_idx)
            from_idx_to_neighbors.pop(centroid_idx)

            # find cluster members
            similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(centroid_fp, list(self.from_idx_to_fp.values())))
            distances = 1 - similarities
            mask = distances <= cutoff

            # retrieve cluster member array using numpy fancy indexing
            cluster_members = np.asarray(list(self.from_idx_to_fp.keys()))[mask]

            # remove cluster members from dictionaries
            self.from_idx_to_fp = {key:self.from_idx_to_fp[key] for key in self.from_idx_to_fp.keys() if key not in set(cluster_members)}
            from_idx_to_neighbors = {key:from_idx_to_neighbors[key] for key in self.from_idx_to_fp.keys()}

            # update arrays with cluster members and cluster centroid
            cluster_number[cluster_members] = cluster_counter
            cluster_centroid[centroid_idx] = 1

            # increase cluster counter
            cluster_counter += 1

        return cluster_number, cluster_centroid



        neighbor_lists = self._neighbor_lists_from_fingerprints(cutoff)

        cluster_number=np.zeros((number_of_fps))
        cluster_centroid=np.zeros((number_of_fps), dtype=bool)

        cluster_counter = 1

        while len(neighbor_lists) > 1:
            centroid = max(neighbor_lists[1])

            for i in neighbor_lists[0][centroid]:
                cluster_number[i] = cluster_counter
                neighbor_lists[0].pop(i)
                neighbor_lists[1].pop(i)

            cluster_centroid[centroid] = 1
            neighbor_lists[0].pop(centroid)
            neighbor_lists[0].pop(centroid)

            cluster_counter += 1




        clusters = self.NewClusterData(distance_matrix,number_of_fps,cutoff,isDistData=True)

        # create arrays filled with 0s, to be later filled with cluster numbers and bools for cluster centroids



        for i,cluster in enumerate(clusters):
            for molecule in cluster:
                cluster_number[molecule]=i
            cluster_centroid[cluster[0]]=1
            del cluster

        return (cluster_number,cluster_centroid)'''