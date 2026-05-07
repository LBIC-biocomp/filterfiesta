# FilterFiesta 🎉

Welcome to FilterFiesta, the ultimate tool to rescore and filter your docked ligands and turn your molecular docking results into a grand fiesta of refined and selected poses! 🎊

## 🎈 Filters That Join the Party:

- **Docking Score**: Set the mood with a docking score cutoff! Only the molecules with the best scores get VIP access to the fiesta. Let the high scorers shine on the dance floor! 🌟💫
- **Pose Coherence**: For those with multiple poses per molecule, evaluate the in-group RMSD and use it as a filter. This ensures your party guests (poses) are in harmony and hitting the right notes! 🎵
- **PLIF Filter**: Shake up your poses by scoring them based on a selection of residue-ligand interactions.
- **Clustering**: Get your molecules to dance together by clustering them based on their fingerprints. With Butina clustering at a given threshold, we select only one representative per cluster to keep the party smooth and exclusive. 💃


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
- Receptor(s) in PDB format
- Docked molecules in SDF format
- Scores in tabular format (`.csv` or `.txt`)
- A list of relevant binding residues for the PLIF filter

For the PLIF filter, prepare a text file with one residue per line in `Chain:ResidueName+Number` format (residue name must be 3-letter long):

```
X:SER441
X:HID296
X:ASP345
X:ASP435
X:GLN438
X:GLY439
```

### Basic example

Filter results from four virtual screenings on four distinct receptors. Values are tab-separated, score cutoff is -1, RMSD cutoff is 1, and the PLIF filter keeps any molecule with at least one interaction compatible with the reference (cutoff 0):

```
filterfiesta -s receptor1_scores.txt receptor2_scores.txt receptor3_scores.txt receptor4_scores.txt \
             -d receptor1_library.sdf receptor2_library.sdf receptor3_library.sdf receptor4_library.sdf \
             -r receptor1.pdb receptor2.pdb receptor3.pdb receptor4.pdb \
             -t "Title" -c "Docking Score" \
             -sep \t \
             --rmsd_cutoff 1 --score_cutoff -1 \
             --residues_reference reference.txt --plif_cutoff 0 \
             --output_suffix mix
```

### OpenEye-style molecule names (`--name_key`)

If your molecules were named by an OpenEye application, titles typically follow the pattern `base_stereoisomerID_conformerID` (e.g. `output_529_1_55`, `Acetic acid.mol_1_1`). Use `--name_key` to parse this format: molecules are then grouped for RMSD calculation by `base_stereoisomerID`, correctly treating different stereoisomers as distinct molecules and ignoring the trailing conformer ID.

### Secondary grouping property (`--sdf_secondary_property`)

If different molecules in your SDF share the same title, provide a secondary SDF property (e.g. a generic name or database ID) via `--sdf_secondary_property`. Molecules are then grouped by both title and this property, preventing RMSD calculation across structurally different compounds.

### Saving all properties without filtering (`-sv`)

Use `-sv` / `--save_all` to skip filtering and instead produce an output table containing all measured properties (RMSD, PLIF similarity, etc.) for every molecule. Useful for inspection and threshold selection before committing to a filtered run.

### Per-file output (`-split`)

By default, molecules from all input files are pooled before clustering and a single grouped output is produced. Use `-split` to instead produce one independent output per input file.


## 🆘 Need Assistance?

```
filterfiesta -h
```

```
  -h, --help            show this help message and exit
  -d DOCKED_MOLECULES [DOCKED_MOLECULES ...], --docked_molecules DOCKED_MOLECULES [DOCKED_MOLECULES ...]
                        File(s) containing docked molecules. Defaults to 'docked_molecules.sdf'.
  -s DOCKED_SCORES [DOCKED_SCORES ...], --docked_scores DOCKED_SCORES [DOCKED_SCORES ...]
                        File(s) containing docked scores. Defaults to 'docked_scores.csv'.
  -r RECEPTORS [RECEPTORS ...], --receptors RECEPTORS [RECEPTORS ...]
                        File(s) containing receptor structures. Defaults to 'receptor.pdb'.
  -t TITLE_COLUMN, --title_column TITLE_COLUMN
                        Name of the column containing molecule names in the CSV files.
  -c SCORE_COLUMN, --score_column SCORE_COLUMN
                        Name of the column containing the scores in the CSV files.
  --sdf_title_property SDF_TITLE_PROPERTY
                        SDF property containing the title of the conformer. Defaults to '_Name'.
  --sdf_secondary_property SDF_SECONDARY_PROPERTY
                        Optional secondary SDF property used as additional grouping key for RMSD
                        calculation. Only molecules matching on both title and this property are
                        grouped together.
  --name_key            If specified, molecule names are parsed as 'base_number1_number2' and
                        grouped by 'base_number1', treating number1 as stereoisomer ID and number2
                        as a conformer ID (default naming by OpenEye applications).
  --sdf_score_property SDF_SCORE_PROPERTY
                        SDF property containing the score of the conformer. Defaults to the
                        score_column if not set.
  --residues_reference RESIDUES_REFERENCE
                        File containing reference residues in the format
                        'chain:three-letter-residue and residue number'. Defaults to
                        'binding_residues.txt'.
  --rmsd_cutoff RMSD_CUTOFF
                        RMSD cutoff to retain molecule groups. Defaults to 1.0.
  --score_cutoff SCORE_CUTOFF
                        Score cutoff to retain molecule groups. Defaults to -12.
  --plif_cutoff PLIF_CUTOFF
                        Jaccard similarity cutoff (range 0-1) between protein-ligand interaction
                        fingerprint of the molecule and reference residues. -1 to deactivate the
                        filter, 0 keep molecules with at least one common interaction with
                        reference, 1 only molecules with a perfect match. Defaults to 0.
  --clustering_cutoff CLUSTERING_CUTOFF
                        Tanimoto distance cutoff for Butina clustering (range 0-1). Defaults to 0.6.
  --output_size OUTPUT_SIZE
                        Number of cluster representatives to save. Defaults to 500.
  --output_suffix OUTPUT_SUFFIX
                        String appended to input filenames to produce output filenames. Defaults to
                        'selection_output'.
  -sep SEPARATOR, --separator SEPARATOR
                        Separator used to parse the score files. Defaults to ','.
  -split, --split       If specified, inputs are NOT pooled before clustering, producing one output
                        per input file.
  -sv, --save_all       If specified, filtering steps are skipped and an output table with all
                        measured properties for all molecules is produced instead.
```

## 🎉 Let's Get This Party Started!

With FilterFiesta, your molecular docking results will not only be filtered but will dance into a new level of refinement. So, get your data ready and let the fiesta begin! 🎊
