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

plants_path = '/path/to/your/PLANTS1.2_64bit'
prot_name = '6XET_-_prepared.mol2'
bindingsite_center = '116.38 89.66 5.91'
bindingsite_radius = '12'

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

      cvt_cmd = "obabel ligand.smi -O ligand.mol2 --gen3D -p > cvt_log.txt"
      os.system(cvt_cmd)
      
      target_folder = '{}/poses'.format(os.getcwd())
      new_folder_index = len([d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]) + 1
      new_folder_name = f'episode_{new_folder_index}'
      new_folder_path = os.path.join(target_folder, new_folder_name)

      # ADAPT YOUR CONFIG FILE BEFORE RUNNING IF NECESSARY
      conf_text = '''# search settings
aco_ants 20
aco_evap 0.15
aco_sigma 1.0
flip_amide_bonds 0
force_planar_bond_rotation 1
force_flipped_bonds_planarity 1

# input
protein_file {}
ligand_file ligand.mol2

# output
output_dir {}
write_multi_mol2 0
write_protein_splitted 0
write_per_atom_scores 0

# binding site
bindingsite_center {}
bindingsite_radius {}

# cluster algorithm
cluster_rmsd 2.0
cluster_structures 1

# score
scoring_function chemplp
chemplp_weak_cho 1
chemplp_charged_hb_weight 1.5'''
      conf_text = conf_text.format(prot_name,new_folder_path,bindingsite_center,bindingsite_radius)
      with open('PLANTS_config.conf','w') as c:
        c.write(conf_text)

      docking_cmd = '{} --mode screen PLANTS_config.conf'.format(plants_path)
      os.system(docking_cmd)

      if os.path.exists('{}/*****_entry_00001_conf_01.mol2'.format(new_folder_path)):

        data = pd.read_csv('{}/ranking.csv'.format(new_folder_path), sep= ",")
      if not os.path.exists('{}/*****_entry_00001_conf_01.mol2'.format(new_folder_path)):
        os.rename(os.path.join(os.getcwd(), 'ligand.smi'), 
                  os.path.join(new_folder_path, 'ligand.smi'))
        os.rename(os.path.join(os.getcwd(), 'ligand.mol2'), 
                  os.path.join(new_folder_path, 'ligand.mol2'))
        files_to_delete = [file for file in os.listdir(new_folder_path) if file not in ['ligand.smi','ligand.mol2','skippedligands.csv']]
        for file in files_to_delete:
          os.remove(os.path.join(new_folder_path, file))
        return 0.0
      docking_score = round(float(data.at[0, 'TOTAL_SCORE']),2)
      
      if os.path.exists('{}/*****_entry_00001_conf_01.mol2'.format(new_folder_path)):
        files_to_delete = [file for file in os.listdir(new_folder_path) if file not in ['*****_entry_00001_conf_01.mol2', 'ranking.csv']]
        for file in files_to_delete:
          os.remove(os.path.join(new_folder_path, file))

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
