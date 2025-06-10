# coding=utf-8
# Copyright 2019 The Google Research Authors.
# Modified Copyright 2020 by Woosung Jeon and Dongsup Kim.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Lint as: python2, python3
"""Optimizes a binding affinity of a molecule against the target with MolDQN.
MORLD tries to find the molecule with the highest docking score
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os,shutil,re

# you need to add path of mol_dqn and gym-molecule. Example,
import sys

      #################################
      #                               #
      #    EDIT THE FOLLOWING LINES   #
      #                               #
      ################################# 
      
sys.path.append("/path/to/your/MORLD_installation_folder")
sys.path.append("/path/to/your/GYM_MOLECULE_installation_folder/gym-molecule")
      #################################

from absl import app
from absl import flags

from rdkit import Chem

from rdkit.Chem import QED
from tensorflow import gfile
from mol_dqn.chemgraph.dqn import deep_q_networks
from mol_dqn.chemgraph.dqn import molecules as molecules_mdp
from mol_dqn.chemgraph.dqn import run_dqn
from mol_dqn.chemgraph.dqn.tensorflow_core import core

FLAGS = flags.FLAGS

from gym_molecule.envs.sascorer import calculateScore
import pandas as pd

      #################################
      #                               #
      #    EDIT THE FOLLOWING LINES   #
      #                               #
      ################################# 

prot_name = '6XET_-_prepared_prot' # set the protein name without extension here
pose_conv_out_format = 'sdf' # set the output format for pose conversion
path_gpf4 = '/path/to/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_gpf4.py'
path_ag4 = '/path/to/autogrid4'
path_adgpu_exec = '/path/to/AutoDock-GPU-develop/bin/autodock_gpu_128wi'
docking_npts = '40,40,40'
docking_gridcenter = '116.38,89.66,5.91'

      #################################

# Molecule class for the reward of binding affinity
class BARewardMolecule(molecules_mdp.Molecule):
  """The molecule whose reward is the Bingding affinity."""

  def __init__(self, discount_factor, **kwargs):
    """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
    super(BARewardMolecule, self).__init__(**kwargs)
    self.discount_factor = discount_factor

  def _reward(self):
    """Reward of a state.

    Returns:
      intermediate reward: SA score, QED score
      final reward: Docking score (a negative value of the binding energy)
    """
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return 0.0

    # calculate SA and QED score
    sa = calculateScore(molecule)
    sa_norm = round((10-sa)/9,2) # normalize the SA score
    qed = round(QED.qed(molecule),2)
    print("SA score and QED: {}, {} : {}".format(sa_norm, qed, self._state))

    if self._counter < self.max_steps: # intermediate state
      return round((sa_norm+qed)*self.discount_factor ** (self.max_steps - self.num_steps_taken),2)

    if self._counter >= self.max_steps: # terminal state
      # create SMILES file
      with open('ligand.smi','w') as f:
        f.write(self._state)

      #################################
      #                               #
      #  START OF CUSTOMIZED SECTION  #
      #                               #
      #################################

      cvt_cmd = "obabel ligand.smi -O ligand.pdbqt --gen3D -p > cvt_log.txt"
      os.system(cvt_cmd)
      
      # Create folder for the ligand
      target_folder = '{}/poses'.format(os.getcwd())
      new_folder_index = len([d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]) + 1
      new_folder_name = f'episode_{new_folder_index}'
      new_folder_path = os.path.join(target_folder, new_folder_name)
      os.mkdir(new_folder_path)

      # Creating the maps for docking
      os.system('{} \
-l ligand.pdbqt \
-r {}.pdbqt \
-o {}.gpf \
-p npts={} \
-p gridcenter={}'.format(path_gpf4,prot_name,prot_name,docking_npts,docking_gridcenter))    
      os.system('{} \
-p {}.gpf \
-l {}.glg'.format(path_ag4,prot_name,prot_name))
     # Run docking
      os.system('{} \
--ffile {}.maps.fld \
--lfile ligand.pdbqt \
--resnam ligand_out'.format(path_adgpu_exec,prot_name))

        
      if os.path.exists('{}/ligand_out.dlg'.format(os.getcwd())):
      # Move the dlg file
        os.rename(os.path.join(os.getcwd(), 'ligand_out.dlg'), 
                  os.path.join(new_folder_path, 'ligand_out.dlg'))
    
      # Convert dlg file in the new location
        def convert_dlg(path_dlg,out_format):
          docked_lines,ids,clust_hists = [],[],[]
          with open(path_dlg) as d:
              for i,l in enumerate(d.readlines()):
                  if 'DOCKED' in l and '    FINAL DOCKED STATE:' not in l:
                      docked_lines.append(l)
                  if '_____|___________|_____|___________|_____|____:____|____:____|____:____|____:___' in l:
                      ids.append(i)
                  if '_____|___________|_____|___________|_____|______________________________________' in l:
                      ids.append(i)
          with open(path_dlg) as d:
              for l in d.readlines()[ids[0]+1:ids[1]]:
                  clust_hists.append(l)
          dict_poses = {}
          poses = ''.join(docked_lines).split('DOCKED: ENDMDL')
          poses.pop()
          for p in poses:
              #Find the string in the text
              pattern = r"DOCKED: USER\s+Run\s+=\s+(\d+)"
              match = re.search(pattern, p)
              run_number = match.group(1)
              # Clean and complete the lines
              p_clean = '\n'.join((p.replace('DOCKED: ', '') + 'ENDMDL').split('\n')[1:])
              # Append to dictionary
              dict_poses[run_number]=p_clean
          dict_clustering = {}
          for i,c in enumerate(clust_hists):
              run_numb = c.replace(' ','').split('|')[2]
              with open('{}/{}_pose{}.pdbqt'.format('/'.join(path_dlg.split('/')[:-1]),
                                                path_dlg.split('/')[-1].replace('.dlg',''),
                                                str(i+1)), 'w') as p_pdbqt:
                  p_pdbqt.write(dict_poses[run_numb])
                                                
              input_path = '{}/{}_pose{}.pdbqt'.format('/'.join(path_dlg.split('/')[:-1]),
                                                       path_dlg.split('/')[-1].replace('.dlg', ''),
                                                       str(i+1))
              output_path = '{}/{}_pose{}.{}'.format('/'.join(path_dlg.split('/')[:-1]),
                                                     path_dlg.split('/')[-1].replace('.dlg', ''),
                                                     str(i+1),
                                                     out_format)
              mol_title = '{}_pose{}'.format(path_dlg.split('/')[-1].replace('.dlg', ''),
                                             str(i+1))
              os.system('obabel {} -O {} --title "{}"'.format(input_path, output_path, mol_title))                                  
        convert_dlg(os.path.join(new_folder_path, 'ligand_out.dlg'),pose_conv_out_format)
    
        # Extract the docking score
        if os.path.exists(os.path.join(new_folder_path, 'ligand_out_pose1.pdbqt')):
          docking_score = []
          with open(os.path.join(new_folder_path, 'ligand_out_pose1.pdbqt')) as d:
            for l in d.readlines():
              if 'Estimated Free Energy of Binding' in l:
                docking_score.append(float(''.join(list(l)[45:52]).replace(' ','')))
          docking_score = docking_score[0]
        
        # Clean up the files
        for f in os.listdir(os.getcwd()):
          if ((prot_name in f and 'pdbqt' not in f) or
              ('ligand' in f and ('pdbqt' not in f or 'smi' not in f))):
              os.remove(os.path.join(os.getcwd(), f))
        for f in os.listdir(new_folder_path):
          if 'ligand_out.dlg' not in f and 'ligand_out_pose1.pdbqt' not in f and 'ligand_out_pose1.{}'.format(pose_conv_out_format) not in f:
            os.remove(os.path.join(new_folder_path, f))
            
            
      if not os.path.exists('{}/ligand_out.dlg'.format(new_folder_path)):
        # Saving problematic ligand
        shutil.copy(os.path.join(os.getcwd(), 'ligand.smi'),
                    os.path.join(new_folder_path, 'ligand.smi'))
        shutil.copy(os.path.join(os.getcwd(), 'ligand.pdbqt'),
                    os.path.join(new_folder_path, 'ligand.pdbqt'))
        # Clean up the files
        for f in os.listdir(os.getcwd()):
          if ((prot_name in f and 'pdbqt' not in f) or
              ('ligand' in f and ('pdbqt' not in f or 'smi' not in f))):
              # Remove the file
              os.remove(os.path.join(os.getcwd(), f))
        with open('./optimized_result_total.txt', 'a') as f2:
          f2.write(self._state+'\t'+str(0.0)+'\t'+str(sa_norm)+'\t'+str(qed)+'\n')
        return 0.0

      print("binding energy value: "+str(round(docking_score,2))+'\t'+self._state)

    # record a optimized result with the SMILES, docking score, SA score,
    # and QED score.
    with open('./optimized_result_total.txt', 'a') as f2:
      f2.write(self._state+'\t'+str(docking_score)+'\t'+str(sa_norm)+'\t'+str(qed)+'\n')

    return round(-docking_score, 2)
    
      #################################
      #                               #
      #   END OF CUSTOMIZED SECTION   #
      #                               #
      #################################

def main(argv):
  del argv  # unused.
  if FLAGS.hparams is not None:
    with gfile.Open(FLAGS.hparams, 'r') as f:
      hparams = deep_q_networks.get_hparams(**json.load(f))
  else:
    hparams = deep_q_networks.get_hparams()
  environment = BARewardMolecule(
      discount_factor=hparams.discount_factor,
      atom_types=set(hparams.atom_types),
      init_mol= FLAGS.start_molecule,
      allow_removal=hparams.allow_removal,
      allow_no_modification=hparams.allow_no_modification,
      allow_bonds_between_rings=hparams.allow_bonds_between_rings,
      allowed_ring_sizes=set(hparams.allowed_ring_sizes),
      max_steps=hparams.max_steps_per_episode)

  dqn = deep_q_networks.DeepQNetwork(
      input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
      q_fn=functools.partial(
          deep_q_networks.multi_layer_model, hparams=hparams),
      optimizer=hparams.optimizer,
      grad_clipping=hparams.grad_clipping,
      num_bootstrap_heads=hparams.num_bootstrap_heads,
      gamma=hparams.gamma,
      epsilon=1.0)

  run_dqn.run_training(
      hparams=hparams,
      environment=environment,
      dqn=dqn)

  core.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config_sa.json'))


if __name__ == '__main__':
  app.run(main)
