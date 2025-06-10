# morld_shape-pharmacophore
This repository contains the source code of the five different docking implementations of MORLD (QuickVina2, AutoDockGPU, PLANTS, GOLD and GLide) along with a novel implemenation of MORLD using shape similarity and pharmacophore alignment insted of docking. 
## Usage
The instructions to reproduce the data obtained in the paper are here presented. 
This is also serving as a tutorial on how to use the different implementations of MORLD.
1. **Install MORLD**  
   Follow the official instructions on the [MORLD GitHub page](https://github.com/wsjeon92/morld).

2. **Copy the optimization script**  
   Choose your implementation and copy its `optimize_BE_{implementation}.py` into your MORLD install:
   ```bash
   cp implementations/optimize_BE_{implementation}.py /path/to/MORLD_installation_folder/mol_dqn/chemgraph/

3. **Copy the data files**  
   Copy the contents of the corresponding data subfolder into chemgraph/:
   ```bash
   cp -r data/{implementation}/* /path/to/MORLD_installation_folder/mol_dqn/chemgraph/

4. **Edit the optimization script**  
   Open optimize_BE_{implementation}.py in your editor. Locate the blocks marked:
   ```python
   EDIT THE FOLLOWING LINES
   ```
   and update each parameter (paths, filenames, etc.) to match your system. Please, always use full paths.

5. **Configure docking inputs (only Glide and QuickVina2)**  
   - **Glide (unconstrained & constrained):**
     Edit ligand.in, setting full paths to the Glide executables and any grid/constraint files.
   - **QuickVina2:**
     Edit config.txt, specifying the protein path, box center/size, exhaustiveness, etc.

6. **Change into the working directory**
   ```bash
   cd /path/to/MORLD_installation_folder/mol_dqn/chemgraph

7. **Run the optimization**  
   1. **Set your initial SMILES:**  
      ```bash
      export INIT_MOL="$(cat initial_mol.smi)"
      ```
   2. **Launch MORLD with your edited script:**  
      ```bash
      python optimize_BE_{implementation}.py \
        --model_dir ${OUTPUT_DIR} \
        --start_molecule "${INIT_MOL}" \
        --hparams "./configs/bootstrap_dqn_step1.json"
      ```
      - Replace `${OUTPUT_DIR}` with your desired output directory.  
      - Point `--hparams` to the JSON config you wish to use.

## Tips & Troubleshooting
- **Permissions:**  
  ```bash
  chmod +x optimize_BE_{implementation}.py
