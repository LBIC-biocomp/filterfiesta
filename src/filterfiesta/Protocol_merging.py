import glob
from filterfiesta.pose_similarity import Similarity
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import tempfile
import os

from filterfiesta.fingerprints import Fingerprint
import gzip
import numpy as np
import csv

from filterfiesta.cluster import Cluster






#######################################################
# 0-INPUT FILE PREPARATION


# fetching paths for ligands' .sdf files and docking score files
ligands=glob.glob("*sdf")
ligands.sort()
ligands=[x for x in ligands if "selection_output" not in x]

score_paths=[lig.replace(".sdf","_score.txt") for lig in ligands]
score_paths.sort()

receptors=glob.glob("*pdb")
receptors.sort()

print("The receptors are:")
[print(f"-{x}") for x in receptors]
print("Ligand paths are:")
[print(f"-{x}") for x in ligands]
print("Score paths are:")
[print(f"-{x}") for x in score_paths]

# Convert .sdf paths into RDKit's suppliers and .csv paths to pandas dataframes
suppliers=[Chem.rdmolfiles.SDMolSupplier(lig) for lig in ligands]
score_dfs_0=[pd.read_csv(score,sep="\t") for score in score_paths]

# create new empty list to be populated with the updated score dataframes, repeat at each step, since modifying them inside the for loop will leave them unchanged outside of it
score_dfs_1 = []

# Match the order of score dataframes and suppliers, add receptor column
for score_df, supplier, rec in zip(score_dfs_0,suppliers, receptors):

	# Check if lenghts match
	if len(supplier) != len(score_df):
		raise ValueError("Lenghts of score table and sdf file are not matching.")

	# Sort score dataframe by title and then by score, the latter ensures the best scoring conformer of each group is always the first
	# Ignores index
	score_df = score_df.sort_values(by=["Title", "FRED Chemgauss4 score"], ascending=True, ignore_index=True, key=lambda s: s.map(lambda x: int(x.split("_")[-1])) if s.name == "Title" else s)


	# Create a dataframe containing titles and scores properties from the sdf
	df = pd.DataFrame()

	titles = []
	scores = []

	for i in range(len(supplier)):
		mol = supplier[i]

		title = mol.GetProp("_Name")
		titles.append(title)

		score = mol.GetProp("FRED Chemgauss4 score")
		scores.append(float(score))

	df["Title"] = titles
	df["FRED Chemgauss4 score"] = scores

	# Sort dataframe of sdf properties, the index is not updated as it reflects the real order of molecules in the sdf
	df = df.sort_values(by=["Title", "FRED Chemgauss4 score"], ascending=True, key=lambda s: s.map(lambda x: int(x.split("_")[-1])) if s.name == "Title" else s)


	# Check if the two dataframe match
	if score_df["Title"].to_list() != df["Title"].to_list():
		raise ValueError("Title columns not matching.")
	else:
		print("Title columns matching.")

	if score_df["FRED Chemgauss4 score"].to_list() != df["FRED Chemgauss4 score"].to_list():
		raise ValueError("Score columns not matching.")
	else:
		print("Score columns matching.")


	# Add column to score dataframe with the index of sdf properties, which represent the correct order to sample the supplier to match the score df
	score_df["Supplier order"] = df.index

	# Add column with receptor name to score dataframe
	rec_name = rec.replace(".pdb","")
	score_df = score_df.assign(Receptor=rec_name)

	score_dfs_1.append(score_df)



score_dfs_0 = []
#######################################################
# 1-DOCKING SCORE

score_dfs_2 = []
score_cutoff = -12 # !!! needs to be adapted to user input

for score_df,supplier in zip(score_dfs_1,suppliers):

    #filter based on selected score threshold
	score_df = score_df[(score_df["FRED Chemgauss4 score"]<score_cutoff)]
	print(f"Score Saved molecules: {len(score_df)}")
	score_dfs_2.append(score_df)

score_dfs_1 = []


#######################################################
# 2-AVERAGE CONFORMER RMSD

score_dfs_3 = []
rmsd_cutoff = 1 # !!! needs to adapt to user input


for supplier, score_df, lig in zip(suppliers,score_dfs_2,ligands): # zip() receives iterables as arguments and pairs in order every i-th element of each iterable as a tuple

	# Calculate average conformer RMSD
	print(f"Calculating conformer similarity for ligands in {lig} ...")
	f=Similarity(supplier,score_df)
	f.groupByN()

	bestscore = f.writeBestScore("FRED Chemgauss4 score","Title", cutoff=rmsd_cutoff, key=lambda s: s.map(lambda x: int(x.split("_")[-1]))) # !!! Key Da rivedere, forse da togliere o da rendere una funzione a se stante
	print(f"RMSD Saved molecules: {len(bestscore)}")
	# Update score dataframe
	score_dfs_3.append(bestscore)

print(f"Done.")

score_dfs_2 = []




#############################################################
# 3-INTERACTION FINGERPRINTS

score_dfs_4 = []
fp_cutoffs = [0] # !!! needs to adapt to user input


# !!!!! insert here existing file check

pd.set_option('future.no_silent_downcasting', True)

ref_residues=glob.glob("reference*txt")
ref_residues.sort()

print("The references are:")
[print(f"-{x}") for x in ref_residues]


# !!!!!! insert here existing All_residues check

# get all residues from the used receptors
all_residues=[]

for rec, supplier, score_df in zip(receptors, suppliers, score_dfs_3):
	f=Fingerprint(rec,supplier,score_df)
	residues=f.get_residues()
	all_residues += residues

# keep only unique residues and sort them by residue number
unique_residues = list(set(all_residues))
unique_residues.sort(key=lambda x: int(x[5:].split('-')[0]))


empty_df = pd.DataFrame(columns=unique_residues) # !!! Add dictionary with datatype bool. Create an empty dataframe with the unique residues as columns, will be the template for the following ones
empty_df = empty_df.astype(int) # !!! DAJE


ia_types = ["VDW","AR-FF","AR-EF","HBD","HBA","I+","I-","ME",]


# Create fingerprints and directly write them to a file
for rec,lig,supplier,score_df in zip(receptors,ligands,suppliers,score_dfs_3):

	print(f"Calculating fingerprint for {lig} ...")
	f=Fingerprint(rec,supplier,score_df)

	table=f.plif()

	print(f"Creating table")
	sorted_table = pd.concat([empty_df, table], ignore_index=True).fillna(0)

	# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SEEMS THAT ALL FINGERPRINT BITS ARE == 0, UNSURE WHY, COULD BE THE ISSUE AS TO WHY ALL MOLECULES GET DISCARDED

	#print(f"fingerprint columns {list(table.columns)}")
	#for i in range(len(sorted_table)):
		#if sorted_table.iloc[i].sum() != 0:
			#print(i)
		#else:
			#print("no")

	# Create reference vectors and calculate Jaccard index
	for ref in ref_residues:
		print(f"Creating reference vectors...")
  		# Read the original columns from the reference file
		residues_df = pd.read_csv(ref, header=None)[0]

		# Create a new DataFrame for the new columns
		residues_ia = []
		for col in residues_df:
			for ia in ia_types:
				residues_ia.append(f"{col}-{ia}")

		# Initialize the new DataFrame with all values set to 1
		new_df = pd.DataFrame(1, index=[0], columns=residues_ia)
		#OK print(f"reference dataframe {list(new_df.loc[0])}")

		# Concatenate with the empty DataFrame
		reference_vector = pd.concat([empty_df, new_df], ignore_index=True).fillna(0)
		#OK print(f"reference dataframe {list(reference_vector.loc[0].astype(int))}")

		# Check lengths
		if reference_vector.shape[1] != sorted_table.shape[1]:
			raise ValueError("The reference vector length must match the number of columns in the fingerprint, i.e., 8 time the number of residues")

		# Calculate fingerprint similarity to reference
		print(f"Calculating Jaccard index between {lig} and reference fingerprints  ...")
		similarity=f.calculate_jaccard(reference_vector, sorted_table)

		# Add fingerprint similarity column to score dataframe
		col_name = ref.split("/")[-1].replace(".txt","")
		score_df[col_name] = similarity
		score_df.to_csv()
       	# Select only molecules complying with the selected cutoff for the specific reference fingerprint
		score_df = score_df[(score_df[col_name]>fp_cutoffs[0])] # !!! needs to adapt to user input
		print(f"Molecule saved: {len(score_df)}")

	print(f"plif Saved molecules: {len(score_df)}")
	score_dfs_4.append(score_df)

print(f"Done.")

score_dfs_3 = []




#######################################################
# 3-BUTINA CLUSTERING

cluster_cutoff = 0.6 # !!! needs to be adapted to user input file="..\\0-PoseSimilarity\\OK_C1_Top10Percent_dockHD.bestpose.sdf"


for lig, supplier, score_df, score_path in zip(ligands, suppliers, score_dfs_4, score_paths):

	csv_out = score_path.replace("txt","selection_output.csv")
	sdf_out = lig.replace(".sdf", "selection_output.sdf")

	c=Cluster(supplier, score_df)


	numbers, centroids=c.cluster(cutoff=cluster_cutoff)

	#p=pd.DataFrame([numbers,centroids]).T
	#p.columns=["Cluster Number","Cluster centroid"]
	score_df["Cluster Number"] = numbers.astype(int)
	score_df["Cluster centroid"] = centroids.astype(int)

	score_df = score_df[(score_df["Cluster centroid"] == True)]

	# Save selected molecules to sdf
	writer = Chem.rdmolfiles.SDWriter(sdf_out)
	for i in score_df["Supplier order"]:
		if supplier[i] is not None: writer.write(supplier[i]) # !!! is the sintax correct?
	writer.close()


	# Save final score table with filtration metrics
	columns = [i for i in score_df.columns if i != "Supplier order"]
	final_df = score_df[columns]
	final_df.to_csv(csv_out, index=False)

print(f"Done. Finished.")