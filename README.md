# FilterFiesta 🎉

Welcome to FilterFiesta, the ultimate tool to rescore and filter your docked ligands and turn your molecular docking results into a grand fiesta of refined and selected poses! 🎊

## 🎈 Filters That Join the Party:

- Docking Score : Set the mood with a docking score cutoff! Only the molecules with the best scores get VIP access to the fiesta. Let the high scorers shine on the dance floor! 🌟💫
- Pose coherence: For those with multiple poses per molecule, evaluate the in-group RMSD and use it as a filter. This ensures your party guests (poses) are in harmony and hitting the right notes! 🎵
- PLIF Filter: Shake up your poses by scoring them based on a selection of residue-ligand interactions. 
- Clustering: Get your molecules to dance together by clustering them based on their fingerprints. With Butina clustering at a given threshold, we select only one representative per cluster to keep the party smooth and exclusive. 💃


## 📥 Installation

Installation steps tested with Python 3.9 ✅, 3.10 ✅, 3.11 ❌
To install FilterFiesta in a virtual environment, follow these steps:

```bash
git clone https://github.com/LBIC-biocomp/filterfiesta
cd filterfiesta
python -m venv venv
source venv/bin/activate
pip install six
pip install .
```

## 📜 How To Get Started:

Prepare:
- Receptors path in PDB format
- Docked molecules in SDF format
- Scores in tabular format (.txt or .csv)
- A list of relevant residues for ligand binding to be used by the Protein-Ligand Interactions Filter

For Protein-Ligand Interactions Filter, prepare a text file with the chain Letter:3-letter residue name and residue number: - Residue name (three-letter format)

```
X:SER441
X:HID296
X:ASP345
X:ASP435
X:GLN438
X:GLY439
```

Below is an example of a program run to filter results from four different virtual screenings on four distinct receptors. FilterFiesta accepts tabular files in both `.csv` and `.txt` formats. Molecule names and docking scores should be specified in the “Title” and “Docking Score” columns, respectively. In this example, values are separated by tabs (as specified by the `-sep \t` flag). The docking score cutoff is set at -1, and the pose coherence (RMSD) cutoff is set at 1. The protein-ligand interaction filter cutoff is set to 0, ensuring that every molecule with at least one interaction compatible with the reference will pass the filter.

```
filterfiesta -s receptor1_scores.txt receptor2_scores.txt receptor3_scores.txt receptor4_scores.txt -d receptor1_library.sdf receptor2_library.sdf receptor3_library.sdf receptor4_library.sdf -t "Title" -c "Docking Score" -r receptor1.pdb receptor2.pdb receptor3.pdb receptor4.pdb -sep \t --rmsd_cutoff 1 --score_cutoff -1 --residues_reference reference.txt --plif_cutoff 0 --output_suffix mix
```

## 🎉 Let's Get This Party Started!

With FilterFiesta, your molecular docking results will not only be filtered but will dance into a new level of refinement. So, get your data ready and let the fiesta begin! 🎊

## 🆘 Need Assistance?

If you need further assistance or have any questions, you can always run the following command to get detailed help:

```
filterfiesta -h
```

This will display the help message with all available options:

```
  -h, --help            show this help message and exit
  -d DOCKED_MOLECULES [DOCKED_MOLECULES ...], --docked_molecules DOCKED_MOLECULES [DOCKED_MOLECULES ...]
                        File(s) containing docked molecules. Defaults to
                        'docked_molecules.sdf'.
  -s DOCKED_SCORES [DOCKED_SCORES ...], --docked_scores DOCKED_SCORES [DOCKED_SCORES ...]
                        File(s) containing docked scores. Defaults to
                        'docked_scores.csv'.
  -r RECEPTORS [RECEPTORS ...], --receptors RECEPTORS [RECEPTORS ...]
                        File(s) containing receptor structures. Defaults to
                        'receptor.pdb'.
  -t TITLE_COLUMN, --title_column TITLE_COLUMN
                        Name of the column containing molecule names in the CSV
                        files.
  -c SCORE_COLUMN, --score_column SCORE_COLUMN
                        Name of the column containing the scores in the CSV files.
  --sdf_title_property SDF_TITLE_PROPERTY
                        SDF property containing the title of the conformer.
                        Defaults to '_Name'.
  --sdf_score_property SDF_SCORE_PROPERTY
                        SDF property containing the score of the conformer.
                        Defaults to the score_column if not set.
  --residues_reference RESIDUES_REFERENCE
                        File containing reference residues in the format
                        'chain\:three-letter-residue and residue number'. Defaults
                        to 'binding_residues.txt'.
  --rmsd_cutoff RMSD_CUTOFF
                        RMSD cutoff to retain molecule groups. Defaults to 1.0.
  --score_cutoff SCORE_CUTOFF
                        Score cutoff to retain molecule groups. Defaults to -12.
  --plif_cutoff PLIF_CUTOFF
                        Jaccard similarity cutoff (range 0-1) between protein-
                        ligand interaction fingerprint of the molecule and
                        reference residues. -1 to deactivate the filter, 0 keep
                        molecules with at least one common interaction with
                        reference, 1 only molecules with a perfect match. Defaults
                        to 0.
  --clustering_cutoff CLUSTERING_CUTOFF
                        Jaccard similarity cutoff (range 0-1) for Butina
                        clustering. Defaults to 0.6.
  --output_size OUTPUT_SIZE
                        Number of molecules to save after filtering. Defaults to
                        500.
  --output_suffix OUTPUT_SUFFIX
                        String to append to input names to obtain output names.
                        Defaults to 'selection_output'.
  -sep SEPARATOR, --separator SEPARATOR
                        Separator used to load and convert the score files into
                        dataframes.
  -split, --split       If specified, all filtered inputs are NOT grouped together
                        before clustering, resulting in a single output for each
                        input file.
  -sv, --save_all       If specified, the input molecules are NOT filtered in the
                        steps before clustering, resulting in an additional output
                        table containing all the measured properties for all the
                        molecules.
```
