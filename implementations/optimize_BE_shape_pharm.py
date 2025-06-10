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
import os

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

maestro_path = '/path/to/your/SCHRODINGER_installation_folder'
pharm_model_name = 'hyp_IFD_anilina.phypo'
shape_template_name = 'sitemap_6XET_IFD_anilina_stand_buff2.mae'

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
      final reward: ShEP score
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
      
      # Create folder for the ligand
      target_folder = '{}/poses'.format(os.getcwd())
      new_folder_index = len([d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]) + 1
      new_folder_name = f'episode_{new_folder_index}'
      new_folder_path = os.path.join(target_folder, new_folder_name)
      os.mkdir(new_folder_path)
      
      # Convert smi file in mol2
      os.system('obabel -ismi ligand.smi -omol2 -O ligand_3d.mol2 --gen3d --title episode_{}'.format(new_folder_index))
      # Protonate mol2 file at neutral pH
      os.system('obabel -imol2 ligand_3d.mol2 -omol2 -O ligand_3d_protonated.mol2 -p7.4')
      # Convert the protonated mol2 file to maegz
      os.system('{}/utilities/structconvert ligand_3d_protonated.mol2 ligand_3d_protonated.maegz'.format(maestro_path))
      
      # Run pharmacophore alignment
      os.system('{}/phase_screen \
ligand_3d_protonated.maegz \
{} \
ligand_3d_protonated \
-flex -max 1000 -keep 999999999 -report 1 -HOST localhost:1 -TMPLAUNCHDIR -WAIT'.format(maestro_path,pharm_model_name))
      # If pharmacophore alignment failed, record the info in output file and save the log
      if not os.path.exists('ligand_3d_protonated-hits.maegz'):
        if os.path.exists('ligand_3d_protonated.log'):
          os.rename(os.path.join(os.getcwd(), 'ligand_3d_protonated.log'), 
                    os.path.join(new_folder_path, 'ligand_3d_protonated.log'))
        with open('./optimized_result_total.txt', 'a') as f2:
          f2.write(self._state+'\t'+str(0.0)+'\t'+str(sa_norm)+'\t'+str(qed)+'\n')
        return 0.0
      # If pharmacophore alignment was successful, continue with shape similarity calculation
      if os.path.exists('ligand_3d_protonated-hits.maegz'):
        os.system('{}/shape_screen \
-shape {} \
-screen ligand_3d_protonated-hits.maegz \
-JOB ligand_3d_protonated_shape_sim \
-inplace -norm 1 -HOST localhost:1 -TMPLAUNCHDIR -WAIT'.format(maestro_path,shape_template_name))
        # If shape similarity calculation failed, record the info and save log files
        if not os.path.exists('ligand_3d_protonated_shape_sim_align.maegz'):
          if os.path.exists('ligand_3d_protonated.log'):
            os.rename(os.path.join(os.getcwd(), 'ligand_3d_protonated.log'), 
                      os.path.join(new_folder_path, 'ligand_3d_protonated.log'))
          if os.path.exists('ligand_3d_protonated_shape_sim_shape.log'):
            os.rename(os.path.join(os.getcwd(), 'ligand_3d_protonated_shape_sim_shape.log'), 
                      os.path.join(new_folder_path, 'ligand_3d_protonated_shape_sim_shape.log'))
          with open('./optimized_result_total.txt', 'a') as f2:
            f2.write(self._state+'\t'+str(0.0)+'\t'+str(sa_norm)+'\t'+str(qed)+'\n')
          return 0.0
        # If shape similarity calculation was successful, convert final file to sdf and extract the data
        if os.path.exists('ligand_3d_protonated_shape_sim_align.maegz'):
          os.system('{}/utilities/structconvert ligand_3d_protonated_shape_sim_align.maegz ligand_3d_protonated_shape_sim_align.sdf'.format(maestro_path))
          scores = []
          with open('{}/ligand_3d_protonated_shape_sim_align.sdf'.format(os.getcwd())) as f:
            lines = f.readlines()
            for i,l in enumerate(lines):
              if 'r_phase_Fitness' in l:
                scores.append((float(lines[i+1].replace('\n','')) + 1) / 4)
              if 'r_phase_Shape_Sim' in l:
                scores.append(float(lines[i+1].replace('\n','')))
      
      # Retrieve the shaep score    
      combo_score = round(((scores[0]*0.5)+(scores[1]*0.5)),2)

      # Move the result files in the created folder of the ligand
      files_to_move = ['ligand_3d_protonated.log','ligand_3d_protonated_shape_sim_shape.log','ligand_3d_protonated_shape_sim_align.sdf']
      for filename in files_to_move:
        if os.path.exists(filename):
          os.rename(os.path.join(os.getcwd(), filename), 
                    os.path.join(new_folder_path, filename))
                  
      # Clean the unnecessary files
      files_to_delete = [file for file in os.listdir(os.getcwd()) if 'ligand' in file and file not in ['ligand.smi']]
      for file in files_to_delete:
        os.remove(os.path.join(os.getcwd(), file))

      # Printing the combo_score
      print("combo score value: "+str(combo_score)+'\t'+self._state)

    # record a optimized result with the SMILES, combo_score, SA score and QED score.
    with open('./optimized_result_total.txt', 'a') as f2:
      f2.write(self._state+'\t'+str(combo_score)+'\t'+str(sa_norm)+'\t'+str(qed)+'\n')

    # Return the combo_score. We DON'T use the negative of the combo_score because the higher combo_score the better.
    return combo_score

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
