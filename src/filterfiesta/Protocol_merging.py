import glob
from filterfiesta.pose_similarity import Similarity
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from filterfiesta.fingerprints import Fingerprint
import gzip
import numpy as np
import csv

from sklearn.utils.multiclass import type_of_target

from filterfiesta.cluster import Cluster




#######################################################
# 0-POSE SIMILARITY

# fetching paths for ligands' .sdf files and docking score files
ligands=glob.glob("*sdf")
ligands.sort()
ligands=[x for x in ligands if "best" not in x]
scores=[lig.replace(".sdf","_score.txt") for lig in ligands]
scores.sort()

print("Ligand paths are:")
[print(f"-{x}") for x in ligands]
print("Score paths are:")
[print(f"-{x}") for x in scores]


# Convert .sdf paths into RDKit's suppliers and .csv paths to pandas dataframes
suppliers=[Chem.rdmolfiles.SDMolSupplier(lig) for lig in ligands]
scoredfs=[pd.read_csv(score,sep="\t") for score in scores]


# !!! removed i +=1 iteration. Would the original solution be more elegant?
for supplier, score_df, score_path, lig in zip(suppliers,scoredfs,scores,ligands): # zip() receives iterables as arguments and pairs in order every i-th element of each iterable as a tuple

	sdf_out=lig.replace("sdf","bestpose.sdf")
	csv_out=score_path.replace("txt","bestpose.csv")
	print(f"output will be {sdf_out}")

	print(f"Calculating conformer similarity for ligands in {lig} ...")
	f=Similarity(supplier,score_df)
	f.groupByN()
	f.writeBestScore(sdf_out,csv_out,"FRED Chemgauss4 score","Title",key=lambda s: s.map(lambda x: int(x.split("_")[-1])))

print(f"Done.")







#############################################################
# 1-CALCULATE FINGERPRINT, 2-COMPARE FINGERPRINT

# !!!!! insert here existing file check

pd.set_option('future.no_silent_downcasting', True)

# Load all input files, 20 initial molecules and full database files are processed together
best_ligands=glob.glob("*best*sdf")
best_ligands.sort()
receptors=glob.glob("*pdb")
receptors.sort()
ref_residues=glob.glob("reference*txt")
ref_residues.sort()

print("The receptors are:")
[print(f"-{x}") for x in receptors]
print("The ligands are:")
[print(f"-{x}") for x in best_ligands]
print("The references are:")
[print(f"-{x}") for x in ref_residues]

# Ensure the receptor list and file list match in length and type
if len(best_ligands) / len(receptors) != 1:
	for i in range(int((len(best_ligands) / len(receptors)) -1)): # !!!! limited to cases where each rec has the same number of groups of docked ligands
		receptors += [i for i in list(set(receptors))]
		receptors.sort()



# !!!!!! insert here existing All_residues check
# get all residues from the used receptors into a list of lists
all_residues=[]
for rec,lig in zip(receptors,best_ligands):
	f=Fingerprint(rec,lig)
	residues=f.get_residues()
	all_residues += [residues]

# convert list of lists into flat list (list of lists is needed later)
flat_all_residues = [x for res in all_residues for x in res]

# keep only unique residues and sort them by residue number
unique_residues = list(set(flat_all_residues))
unique_residues.sort(key=lambda x: int(x[5:].split('-')[0]))

# Save the complete list of unique residues
with open("All_residues.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(unique_residues)

empty_df = pd.DataFrame(columns=unique_residues) # Create an empty dataframe with the unique residues as columns, will be the template for the following ones
print(f"type check: {type_of_target(empty_df)}") # !!! GIVES "UNKNOWN" TYPE



# Create reference vectors
reference_vectors = []
ia_types = ["VDW","AR-FF","AR-EF","HBD","HBA","I+","I-","ME",]

print(f"Creating reference vectors...")
for ref in ref_residues:
    # Read the original columns from the reference file
    residues_df = pd.read_csv(ref, header=None)[0]

    # Create a new DataFrame for the new columns
    residues_ia = []
    for col in residues_df:
        for ia in ia_types:
            residues_ia.append(f"{col}-{ia}")

    # Initialize the new DataFrame with all values set to 1
    new_df = pd.DataFrame(1, index=[0], columns=residues_ia)

    print(f"type check: {type_of_target(new_df)}")

    # Concatenate with the empty DataFrame
    reference_vector = pd.concat([empty_df, new_df], ignore_index=True).fillna(0)
    print(f"type check: {type_of_target(reference_vector)}") # !!! CHANGES TYPE FROM MULTI-LABEL TO UNKNOWN
    reference_vectors.append(reference_vector)

    # Save the new DataFrame as a .npy file
    output_file = ref.replace(".txt", ".ref_fingerprint.npy")
    np.save(output_file, reference_vector.to_numpy())
print(f"Done.")



# Create fingerprints and directly write them to a file
for rec,lig,res in zip(receptors,best_ligands,all_residues):
	similarities = []

	print(f"Calculating fingerprint for {lig} ...")
	f=Fingerprint(rec,lig)
	table=f.plif(res)
	print(f"Creating table")
	sorted_table = pd.concat([empty_df, table], ignore_index=True).fillna(0)
	output_file=lig.replace("sdf","fingerprint.csv.gz") # !!!! output name now derives from ligands name and not rec name

	print(f"Writing output to {output_file}")
	with gzip.open(output_file, 'wt') as gz_file: # inizializza la variabile gz_file e cancella la memoria dopo che il blocco indentato ha finito di eseguire
		sorted_table.to_csv(gz_file, index=False)


	# Calculate Jaccard index
	for ref in reference_vectors:
		print(f"type check: {type_of_target(ref)}")
		if ref.shape[1] != sorted_table.shape[1]:
			raise ValueError("The reference vector length must match the number of columns in the fingerprint, i.e., 8 time the number of residues")

		print(f"Calculating Jaccard index between {lig} and reference fingerprints  ...")
		similarity=f.calculate_jaccard(ref, sorted_table) # !!! NOT WORKING YET
		print(f"type check: {type_of_target(ref)}")
		similarities.append(similarity)

	similarities=pd.DataFrame(similarities).T # Transpose dataframe (columns to indexes and vice versa)
	similarities.columns= [path.split("/")[-1].replace(".txt","") for path in ref_residues]
	similarities.to_csv(lig.split("/")[-1].replace("sdf","overlap.csv"),index=False)


print(f"Done. Finished.")





#######################################################
# 3-CLUSTER


file="..\\0-PoseSimilarity\\OK_C1_Top10Percent_dockHD.bestpose.sdf"

for lig in best_ligands:

	c=Cluster(lig)
	numbers, centroids=c.cluster(cutoff=0.3)

	p=pd.DataFrame([numbers,centroids]).T
	p.columns=["Cluster Number","Cluster centroid"]

	output_file = lig.replace(".sdf", f".Clusters_cutoff0.3.csv") # !!! hot to automatically include selected cutoff  value in output path?
	p.to_csv("Clusters_cutoff0.3.csv",index=False)
