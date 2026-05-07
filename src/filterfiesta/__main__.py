#!/usr/bin/env python3

import argparse
import pandas as pd
from rdkit import Chem
import time
from itertools import chain
from pathlib import Path

from filterfiesta.fingerprints import Fingerprint
from filterfiesta.cluster import Cluster
from filterfiesta.pose_similarity import Similarity


class RunLogger:
    def __init__(self, output_suffix):
        self.lines = []
        self.output_suffix = output_suffix
        self.run_start = time.time()

    def _header(self, title):
        bar = "=" * 60
        self.lines.append(f"\n{bar}\n  {title}\n{bar}")

    def _subheader(self, title):
        self.lines.append(f"\n--- {title} ---")

    def add(self, text):
        self.lines.append(text)

    def save(self, path=None):
        if path is None:
            path = _get_unique_path(Path(f"filterfiesta_{self.output_suffix}.log"))
        total = int(time.time() - self.run_start)
        self.lines.append(f"\n{'='*60}")
        self.lines.append(f"  Total run time: {total} seconds")
        self.lines.append(f"{'='*60}")
        with open(path, "w") as f:
            f.write("\n".join(self.lines))
        print(f"Log saved to {path}")


def create_parser():
    parser = argparse.ArgumentParser(description="Filter docked molecules based on various parameters.")

    parser.add_argument(
        "-d", "--docked_molecules",
        nargs="+",
        default=["docked_molecules.sdf"],
        help="File(s) containing docked molecules. Defaults to 'docked_molecules.sdf'."
    )
    parser.add_argument(
        "-s", "--docked_scores",
        nargs="+",
        default=["docked_scores.csv"],
        help="File(s) containing docked scores. Defaults to 'docked_scores.csv'."
    )
    parser.add_argument(
        "-r", "--receptors",
        nargs="+",
        default=["receptor.pdb"],
        help="File(s) containing receptor structures. Defaults to 'receptor.pdb'."
    )
    parser.add_argument(
        "-t", "--title_column",
        required=True,
        help="Name of the column containing molecule names in the CSV files."
    )
    parser.add_argument(
        "-c", "--score_column",
        required=True,
        help="Name of the column containing the scores in the CSV files."
    )
    parser.add_argument(
        "--sdf_title_property",
        default="_Name",
        help="SDF property containing the title of the conformer. Defaults to '_Name'."
    )
    parser.add_argument(
        "--sdf_score_property",
        default=None,
        help="SDF property containing the score of the conformer. Defaults to the score_column if not set."
    )
    parser.add_argument(
        "--residues_reference",
        default="binding_residues.txt",
        help="File containing reference residues in the format 'chain:three-letter-residue and residue number'. Defaults to 'binding_residues.txt'."
    )
    parser.add_argument(
        "--rmsd_cutoff",
        type=float,
        default=1.0,
        help="RMSD cutoff to retain molecule groups. Defaults to 1.0."
    )
    parser.add_argument(
        "--score_cutoff",
        type=float,
        default=-12.0,
        help="Score cutoff to retain molecule groups. Defaults to -12."
    )
    parser.add_argument(
        "--plif_cutoff",
        type=float,
        default=0.0,
        help="Jaccard similarity cutoff (range 0-1) between protein-ligand interaction fingerprint of the molecule and reference residues. \
        -1 to deactivate the filter, 0 keep molecules with at least one common interaction with reference, 1 only molecules with a perfect match. Defaults to 0."
    )
    parser.add_argument(
        "--clustering_cutoff",
        type=float,
        default=0.6,
        help="Jaccard similarity cutoff (range 0-1) for Butina clustering. Defaults to 0.6."
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=500,
        help="Number of molecules to save after filtering. Defaults to 500."
    )
    parser.add_argument(
        "--output_suffix",
        default="selection_output",
        help="String to append to input names to obtain output names. Defaults to 'selection_output'."
    )
    parser.add_argument(
        "-sep","--separator",
        default=",",
        help="Separator used to load and convert the score files into dataframes."
    )

    parser.add_argument(
        "-split","--split",
        action="store_true",
        help="If specified, all filtered inputs are NOT grouped together before clustering, resulting in a single output for each input file."
    )

    parser.add_argument(
        "-sv","--save_all",
        action="store_true",
        help="If specified, the input molecules are NOT filtered in the steps before clustering, resulting in an additional output table containing all the measured properties for all the molecules."
    )


    args = parser.parse_args()
    return args


def match_molecule_order(input_score_dfs, docked_molecules, receptors, title_column, score_column,sdf_molecule_name,sdf_score_property, log):
    log._header("STEP 1 — MOLECULE MATCHING")

    new_score_dfs = []
    supplier_lengths = []
    start_time = time.time()
    print("Party planners are checking the molecule list…")

    # Match the order of score dataframes and suppliers, add receptor column
    for score_df, docked_molecule, rec in zip(input_score_dfs,docked_molecules, receptors):
        # Sort score dataframe by title and then by score, the latter ensures the best scoring conformer of each group is always the first
        score_df = score_df.sort_values(by=[title_column, score_column], ascending=True, ignore_index=True)

        # Create a dataframe containing titles and scores properties from the sdf
        df = pd.DataFrame()
        titles = []
        scores = []
        mol_index = 0
        mol_count = 0
        start_time = time.time()

        found_title = False
        found_property = False
        new_molecule=True

        with open(docked_molecule, "r") as file:
            for line in file:
                if new_molecule and (sdf_molecule_name=="_Name"):
                    titles.append(line.strip())
                    found_title = True
                    new_molecule=False
                elif line.strip() == "> <"+sdf_molecule_name+">":
                    titles.append(next(file).strip())
                    found_title = True

                if line.strip() == "> <"+sdf_score_property+">":
                    scores.append(float(next(file).strip()))
                    found_property = True

                if line.strip() == "$$$$":
                    if not found_title:
                        raise ValueError(
                            f"Molecule {mol_index} in {docked_molecule} is missing "
                            f"the property tag '{sdf_molecule_name}'. "
                            f"Check --sdf_title_property is set correctly."
                        )
                    if not found_property:
                        raise ValueError(
                            f"Molecule {mol_index} in {docked_molecule} is missing "
                            f"the property tag '{sdf_score_property}'. "
                            f"Check --sdf_score_property is set correctly."
                        )

                    mol_index += 1
                    mol_count += 1
                    found_title = False
                    found_property = False
                    new_molecule=True
        supplier_lengths.append(mol_count)

        df[title_column] = titles
        df[score_column] = scores

        # Sort dataframe of sdf properties, the index is not updated as it reflects the real order of molecules in the sdf
        df = df.sort_values(by=[title_column, score_column], ascending=True)

        # Check if the two dataframe match
        if score_df[title_column].to_list() != df[title_column].to_list():
            raise ValueError("Title columns not matching.")

        if score_df[score_column].to_list() != df[score_column].to_list():
            raise ValueError("Score columns not matching.")

        # Add column to score dataframe with the index of sdf properties, which represent the correct order with which to sample the supplier to match the score df
        score_df["Supplier order"] = df.index

        log.add(f"  {docked_molecule}: {len(score_df)} molecules matched")

        # Add column with receptor name to score dataframe
        rec_name = rec.replace(".pdb","")
        score_df = score_df.assign(Receptor=rec_name)

        new_score_dfs.append(score_df)
    print("Molecules perfectly lined up!")
    print("--- Done. %s seconds ---" % int(time.time() - start_time))

    log.add(f"Completed in {int(time.time() - start_time)} seconds")
    return new_score_dfs, supplier_lengths


def average_conformer_rmsd(input_score_dfs, suppliers,docked_molecules, title_column, score_column, rmsd_cutoff,filter, log):
    log._header("STEP 2 — POSE COHERENCE (RMSD) FILTER")
    log.add(f"RMSD cutoff: {rmsd_cutoff} | Filter active: {filter}")

    start_time = time.time()
    new_score_dfs=[]
    print(f"\nCalculating average conformer RMSD...")
    for supplier, score_df, lig in zip(suppliers,input_score_dfs,docked_molecules):

        # Calculate average conformer RMSD
        f=Similarity(supplier,score_df)
        f.groupByName(name_column=title_column)

        grouped_scores = f.writeBestScore(score_column,title_column, cutoff=rmsd_cutoff, filter=filter)

        # update log
        n_in = len(score_df)
        n_out = len(grouped_scores)
        log.add(f"  {lig}: {n_in} poses in → {n_out} unique molecules retained "
                f"({'filter off' if not filter else f'{n_in - n_out} removed'})")

        # Update score dataframe
        new_score_dfs.append(grouped_scores)

    for supplier, score_df, lig in zip(suppliers,new_score_dfs,docked_molecules):
        print(f"{lig}... kept {len(score_df)} unique molecules.")

    print("--- Done. %s seconds ---" % int(time.time() - start_time))
    log.add(f"Completed in {int(time.time()- start_time)} seconds")
    return new_score_dfs


def docking_score_filter(input_dfs, suppliers, docked_molecules, score_column, score_cutoff, filter, log):
    log._header("STEP 3 — DOCKING SCORE FILTER")
    log.add(f"Score cutoff: {score_cutoff} | Filter active: {filter}")
    start_time = time.time()
    new_score_dfs=[]
    print("\nIt's time for limbo!")
    print(f"Retaining molecules with score below {score_cutoff}:")

    for lig,score_df,supplier in zip(docked_molecules,input_dfs,suppliers):
        #filter based on selected score threshold
        if filter:
            score_df = score_df[(score_df[score_column]<score_cutoff)]
            # update log
        n_in = len(input_dfs[docked_molecules.index(lig)])  # molecules coming in
        log.add(f"  {lig}: {n_in} in → {len(score_df)} passed "
                f"({'filter off' if not filter else f'{n_in - len(score_df)} removed'})")

        print(f"{lig}... {len(score_df)} passed.")

        new_score_dfs.append(score_df)

    print("--- Done. %s seconds ---\n\n" % int(time.time() - start_time))
    log.add(f"Completed in {int(time.time() - start_time)} seconds")
    return new_score_dfs


def plif_filter(input_dfs,reference_file, plif_cutoff,receptors, suppliers, docked_molecules,filter, log):
    log._header("STEP 4 — PLIF FILTER")
    log.add(f"PLIF cutoff: {plif_cutoff} | Filter active: {filter}")
    log.add(f"Reference file: {reference_file}")
    start_time = time.time()
    new_score_dfs = []
    pd.set_option('future.no_silent_downcasting', True)

    ref_residues=[reference_file]

    # get all residues from the used receptors
    all_residues=[]
    for rec, supplier, score_df in zip(receptors, suppliers, input_dfs):
        f=Fingerprint(rec,supplier,score_df)
        residues=f.get_residues()
        all_residues += residues

    # keep only unique residues and sort them by residue number
    unique_residues = list(set(all_residues))
    unique_residues.sort(key=lambda x: int(x[5:].split('-')[0]))
    log.add(f"Unique residues across all receptors: {len(unique_residues)}")

    # Create an empty dataframe with the unique residues as columns, will be the template for the following ones
    empty_df = pd.DataFrame(columns=unique_residues) # !!! Add dictionary with datatype bool.
    empty_df = empty_df.astype(int)
    ia_types = ["VDW","AR-FF","AR-EF","HBD","HBA","I+","I-","ME",]

    # Create fingerprints
    for rec,lig,supplier,score_df in zip(receptors,docked_molecules,suppliers,input_dfs):
        print(f"Calculating fingerprint for {lig} ...")
        f=Fingerprint(rec,supplier,score_df)
        table=f.plif()

        print(f"Creating table")
        sorted_table = pd.concat([empty_df, table], ignore_index=True).fillna(0)

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

            # Concatenate with the empty DataFrame
            reference_vector = pd.concat([empty_df, new_df], ignore_index=True).fillna(0)

            # Check lengths
            if reference_vector.shape[1] != sorted_table.shape[1]:
                raise ValueError("The reference vector length must match the number of columns in the fingerprint, i.e., 8 time the number of residues")

            # Calculate fingerprint similarity to reference
            print(f"Calculating Jaccard index between {lig} and reference fingerprints  ...")
            similarity=f.calculate_jaccard(reference_vector, sorted_table)

            # update log
            log.add(f"  {lig} | ref {ref}: Jaccard scores — "
                    f"min={min(similarity):.3f}, max={max(similarity):.3f}, "
                    f"mean={sum(similarity)/len(similarity):.3f}")

            # Add fingerprint similarity column to score dataframe
            col_name = ref.split("/")[-1].replace(".txt"," similarity")
            score_df[col_name] = similarity

            # Select only molecules complying with the selected cutoff for the specific reference fingerprint
            print(f"filtering based on {col_name}")
            if filter:
                score_df = score_df[(score_df[col_name]>plif_cutoff)]
                log.add(f"  {lig}: {len(score_df)} molecules passed PLIF filter")

            print(f"{lig}... {len(score_df)} passed.")
        new_score_dfs.append(score_df)

    print("--- Done. %s seconds ---\n\n" % int(time.time() - start_time))
    log.add(f"Completed in {int(time.time() - start_time)} seconds")
    return new_score_dfs


def cluster_filter(input_dfs, supplier_lengths, suppliers, clustering_cutoff, clusters_to_save, grouped, log):
    log._header("STEP 5 — CLUSTERING")
    log.add(f"Clustering cutoff: {clustering_cutoff} | Grouped: {grouped} | Max clusters to save: {clusters_to_save}")
    start_time = time.time()
    print("Clustering molecules....")
    new_score_dfs = []

    if grouped:
        concatenated_dfs=input_dfs[0].copy()
        concatenated_dfs["Original Supplier order"] = concatenated_dfs["Supplier order"].map(int)

        for i in range(1,len(input_dfs)):
            df= input_dfs[i].copy()
            df["Original Supplier order"]=df["Supplier order"].map(int)
            df["Supplier order"]=df["Supplier order"] + sum(supplier_lengths[:i])
            concatenated_dfs=pd.concat([concatenated_dfs,df])

        new_supplier=[]

        for sup in suppliers:
            new_supplier+=list(sup)
        suppliers=[new_supplier]
        input_dfs=[concatenated_dfs]

    for supplier, score_df in zip(suppliers, input_dfs):
        c=Cluster(supplier, score_df)
        numbers, centroids=c.cluster(cutoff=clustering_cutoff)

        # update log
        n_clusters = len(set(numbers))
        n_centroids = int(sum(centroids))
        n_in = len(score_df)
        log.add(f"  Input molecules: {n_in} | Clusters found: {n_clusters} | Centroids: {n_centroids}")

        numbers = numbers.astype(int)
        centroids = centroids.astype(int)

        score_df = score_df.copy()

        score_df["Cluster number"] = numbers.astype(int)
        score_df["Cluster centroid"] = centroids.astype(int)

        # Save centroids from 500 most populated cluster
        score_df = score_df[(score_df["Cluster centroid"] == 1)]
        score_df = score_df[(score_df["Cluster number"] < clusters_to_save)]

        # update log
        log.add(f"  Molecules saved (top {clusters_to_save} clusters): {len(score_df)}")

        if 'Original Supplier order' in score_df.columns:
            score_df['Concatenated Supplier order']=score_df['Supplier order']
            score_df['Supplier order']=score_df['Original Supplier order']
            score_df.drop('Original Supplier order',axis=1,inplace=True)
        new_score_dfs.append(score_df)

    print("--- Done. %s seconds ---\n\n" % int(time.time() - start_time))
    log.add(f"Completed in {int(time.time() - start_time)} seconds")
    return new_score_dfs


def _get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def _write_output(df, supplier, csv_path, sdf_path, log):
    if 'Concatenated Supplier order' in df.columns:
        selected_mols = [supplier[i] for i in df['Concatenated Supplier order'] if supplier[i] is not None]
        df.drop('Concatenated Supplier order', axis=1, inplace=True)
    else:
        selected_mols = [supplier[i] for i in df['Supplier order'] if supplier[i] is not None]
    df.drop('Supplier order', axis=1, inplace=True)

    with Chem.SDWriter(sdf_path) as writer:
        for mol in selected_mols:
            mol = Chem.rdmolops.AddHs(mol, addCoords=True)
            writer.write(mol)

    df.to_csv(csv_path, index=False)

    # update log
    log.add(f"  CSV:  {csv_path} ({len(df)} molecules)")
    log.add(f"  SDF:  {sdf_path} ({len(selected_mols)} molecules written)")


def save(input_dfs, docked_scores, docked_molecules, suppliers, output_suffix, grouped, log):
    log._header("OUTPUT")
    start_time = time.time()
    print("Writing output files...")

    if grouped:
        new_supplier = []
        for sup in suppliers:
            new_supplier += list(sup)
        csv_path = _get_unique_path(Path(f"grouped_{output_suffix}.csv"))
        sdf_path = _get_unique_path(Path(f"grouped_{output_suffix}.sdf"))
        _write_output(input_dfs[0], new_supplier, csv_path, sdf_path, log)

    else:
        for df, input_csv, input_sdf, supplier in zip(input_dfs, docked_scores, docked_molecules, suppliers):
            input_csv_path = Path(input_csv)
            input_sdf_path = Path(input_sdf)
            csv_path = _get_unique_path(Path(f"{input_csv_path.stem}_{output_suffix}{input_csv_path.suffix}"))
            sdf_path = _get_unique_path(Path(f"{input_sdf_path.stem}_{output_suffix}{input_sdf_path.suffix}"))
            _write_output(df, supplier, csv_path, sdf_path, log)

    print("--- Done. %s seconds ---\n\n" % int(time.time() - start_time))
    log.add(f"Completed in {int(time.time()-t)} seconds")
    log.save()   # ← writes the log file

'''
def save(input_dfs,docked_scores, docked_molecules, suppliers, output_suffix, grouped):
    start_time = time.time()
    print("Writing output files...")

    if grouped:
        new_supplier=[]
        for sup in suppliers:
            new_supplier += list(sup)
        suppliers=[new_supplier]

    for (df, input_csv,input_sdf, supplier) in zip(input_dfs,docked_scores,docked_molecules,suppliers):
        input_csv_path=Path(input_csv)
        output_csv_path = _get_unique_path(Path(f"{input_csv_path.stem}_{output_suffix}{input_csv_path.suffix}"))
        input_sdf_path=Path(input_sdf)
        output_sdf_path = _get_unique_path(Path(f"{input_sdf_path.stem}_{output_suffix}{input_sdf_path.suffix}"))

        if grouped:
            output_csv_path = _get_unique_path(Path(f"grouped_{output_suffix}{input_csv_path.suffix}"))
            output_sdf_path = _get_unique_path(Path(f"grouped_{output_suffix}{input_sdf_path.suffix}"))

        if 'Concatenated Supplier order' in df.columns:
            selected_mols = [supplier[i] for i in df['Concatenated Supplier order'] if supplier[i] is not None]
            df.drop('Concatenated Supplier order',axis=1,inplace=True)
        else:
            selected_mols = [supplier[i] for i in df['Supplier order'] if supplier[i] is not None]
        df.drop('Supplier order',axis=1,inplace=True)

        with Chem.SDWriter(output_sdf_path) as writer:
            for mol in selected_mols:
                mol= Chem.rdmolops.AddHs(mol,addCoords=True)
                writer.write(mol)
        df.to_csv(output_csv_path,index=False)
    print("--- Done. %s seconds ---\n\n" % int(time.time() - start_time))
'''

def main():
    print("""    ______ ____ __   ______ ______ ____   ______ ____ ______ _____ ______ ___
   / ____//  _// /  /_  __// ____// __ \\ / ____//  _// ____// ___//_  __//   |
  / /_    / / / /    / /  / __/  / /_/ // /_    / / / __/   \\__ \\  / /  / /| |
 / __/  _/ / / /___ / /  / /___ / _, _// __/  _/ / / /___  ___/ / / /  / ___ |
/_/    /___//_____//_/  /_____//_/ |_|/_/    /___//_____/ /____/ /_/  /_/  |_|

""")
    args=create_parser()

    log = RunLogger(args.output_suffix)

    log._header("RUN PARAMETERS")
    log.add(f"Run started:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.add(f"Score files:         {args.docked_scores}")
    log.add(f"Molecule files:      {args.docked_molecules}")
    log.add(f"Receptor files:      {args.receptors}")
    log.add(f"Title column:        {args.title_column}")
    log.add(f"Score column:        {args.score_column}")
    log.add(f"SDF title property:  {args.sdf_title_property}")
    log.add(f"SDF score property:  {args.sdf_score_property}")
    log.add(f"Score cutoff:        {args.score_cutoff}")
    log.add(f"RMSD cutoff:         {args.rmsd_cutoff}")
    log.add(f"PLIF cutoff:         {args.plif_cutoff}")
    log.add(f"Clustering cutoff:   {args.clustering_cutoff}")
    log.add(f"Output size:         {args.output_size}")
    log.add(f"Output suffix:       {args.output_suffix}")
    log.add(f"Grouped mode:        {not args.split}")
    log.add(f"Save all:            {args.save_all}")
    log.add(f"Reference residues:  {args.residues_reference}")

    if not args.sdf_score_property:
        args.sdf_score_property=args.score_column


    suppliers=[Chem.rdmolfiles.SDMolSupplier(lig) for lig in args.docked_molecules]
    dfs=[pd.read_csv(score,sep=args.separator) for score in args.docked_scores]

    log._header("INPUT FILES")
    for sdf, csv, rec in zip(args.docked_molecules, args.docked_scores, args.receptors):
        mol_count = len(pd.read_csv(csv, sep=args.separator))
        log.add(f"  {sdf}: {mol_count} molecules in score table | receptor: {rec}")

    right_order_dfs, supplier_lengths=match_molecule_order(dfs,
                                         docked_molecules=args.docked_molecules,
                                         receptors=args.receptors,
                                         title_column=args.title_column,
                                         score_column=args.score_column,
                                         sdf_molecule_name=args.sdf_title_property,
                                         sdf_score_property=args.sdf_score_property,
                                         log=log
                                        )

    rmsd_filtered_dfs=average_conformer_rmsd(right_order_dfs,
                                             suppliers=suppliers,
                                             docked_molecules=args.docked_molecules,
                                             title_column=args.title_column,
                                             score_column=args.score_column,
                                             rmsd_cutoff=args.rmsd_cutoff,
                                             filter=not args.save_all,
                                             log=log
                                             )

    filtered_docking_dfs=docking_score_filter(rmsd_filtered_dfs,
                                              suppliers=suppliers,
                                              docked_molecules=args.docked_molecules,
                                              score_column=args.score_column,
                                              score_cutoff=args.score_cutoff,
                                              filter=not args.save_all,
                                              log=log
                                              )

    plif_filtered_dfs=plif_filter(filtered_docking_dfs,
                                reference_file=args.residues_reference,
                                plif_cutoff=args.plif_cutoff,
                                receptors=args.receptors,
                                suppliers=suppliers,
                                docked_molecules=args.docked_molecules,
                                filter=not args.save_all,
                                log=log
                                )

    a=plif_filtered_dfs
    print(a[0][:10])
    titles = [mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else "Untitled" for mol in suppliers[0]]
    print([titles[x] for x in a[0]["Supplier order"][:10]])


    if args.save_all:
        save(input_dfs=plif_filtered_dfs,
         docked_scores=args.docked_scores,
         docked_molecules=args.docked_molecules,
         suppliers=suppliers,
         output_suffix=args.output_suffix,
         grouped=True)

    else:
        clusterd_dfs=cluster_filter(input_dfs=plif_filtered_dfs,
                                    supplier_lengths=supplier_lengths,
                                    suppliers=suppliers,
                                    clustering_cutoff=args.clustering_cutoff,
                                    clusters_to_save=args.output_size,
                                    grouped=not args.split,
                                    log=log
                                    )

        save(input_dfs=clusterd_dfs,
            docked_scores=args.docked_scores,
            docked_molecules=args.docked_molecules,
            suppliers=suppliers,
            output_suffix=args.output_suffix,
            grouped=not args.split,
            log=log
            )
if __name__ == "__main__":
    main()
