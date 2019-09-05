# Model

from __future__ import absolute_import, division
from __future__ import print_function
import sys, string, pickle, subprocess, os, datetime, gzip, time
from collections import defaultdict, OrderedDict
import glob
import numpy as np, pandas as pd

nts = list('ACGT')
nt_to_idx = {nt: nts.index(nt) for nt in nts}

model_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/lib-modeling/out/anyedit_gbtr/'

model_settings = {
  'celltype': 'mES',
  'base_editor': 'eA3A',
  '__model_nm': 'mES_12kChar_eA3A',
  '__train_test_id': '0_old',
}

model = None
init_flag = False

'''
  Intended usage: 
    (rename predict to model nm)
    import _predict_anyedit
    _predict_anyedit.init_model(base_editor = '', celltype = '')

    scalar = _predict.predict(seq)
'''

####################################################################
# Private
####################################################################
'''
  Featurization
'''
ohe_encoder = {
  'A': [1, 0, 0, 0],
  'C': [0, 1, 0, 0],
  'G': [0, 0, 1, 0],
  'T': [0, 0, 0, 1],
}
def __one_hot_encode(seq):
  ohe = []
  for nt in seq:
    ohe += ohe_encoder[nt]
  return ohe
  
def __get_one_hot_encoder_nms(start_pos, end_pos):
  nms = []
  nts = list('ACGT')
  for pos in range(start_pos, end_pos + 1):
    for nt in nts:
      nms.append('%s%s' % (nt, pos))
  return nms

dint_encoder = {
  'AA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AC': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AG': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AT': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  'GA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  'TT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}
def __dinucleotide_encode(seq):
  ohe = []
  for idx in range(len(seq) - 1):
    ohe += dint_encoder[seq[idx : idx + 2]]
  return ohe

def __get_dinucleotide_nms(start_pos, end_pos):
  nms = []
  dints = sorted(list(dint_encoder.keys()))
  for pos in range(start_pos, end_pos):
    for dint in dints:
      nms.append('%s%s' % (dint, pos))
  return nms


def __featurize(seq):
  '''
    start_pos, end_pos = -9, 21   # go up to N in NGG

  '''
  curr_x = []
  pos_to_idx = lambda pos: pos + 19
  seq = seq[pos_to_idx(-9) : pos_to_idx(21) + 1]

  # One hot encoding
  curr_x += __one_hot_encode(seq)

  # Dinucleotides
  curr_x += __dinucleotide_encode(seq)

  # Sum nucleotides
  features = [
    seq.count('A'),
    seq.count('C'),
    seq.count('G'),
    seq.count('T'),
    seq.count('G') + seq.count('C'),
  ]
  curr_x += features

  # Melting temp
  from Bio.SeqUtils import MeltingTemp as mt
  features = [
    mt.Tm_NN(seq),
    mt.Tm_NN(seq[-5:]),
    mt.Tm_NN(seq[-13:-5]),
    mt.Tm_NN(seq[-21:-13]),
  ]
  curr_x += features

  # ohe_nms = __get_one_hot_encoder_nms(start_pos, end_pos)
  # dint_nms = __get_dinucleotide_nms(start_pos, end_pos)
  # sum_nms = ['Num. A', 'Num. C', 'Num. G', 'Num. T', 'Num. GC']
  # mt_nms = ['Tm full', 'Tm -5', 'Tm -13 to -5', 'Tm -21 to -13']
  # param_nms = ['x_%s' % (ft_nm) for ft_nm in ohe_nms + dint_nms + sum_nms + mt_nms]
  # return (np.array(X_all), param_nms)
  return np.array(curr_x).reshape(1, -1)


####################################################################
# Public 
####################################################################

def predict(seq):
  assert len(seq) == 50, f'Error: Sequence provided is {len(seq)}, must be 50 (positions -19 to 30 w.r.t. gRNA (positions 1-20)'

  assert init_flag, f'Call .init_model() first.'
  seq = seq.upper()

  x = __featurize(seq)
  y = model.predict(x)
  return float(y)


def init_model(base_editor = '', celltype = ''):
  # Update global settings
  spec = {
    'base_editor': base_editor,
    'celltype': celltype,
    '__model_nm': f'{celltype}_12kChar_{base_editor}',
  }
  global model_settings
  for key in spec:
    if spec[key] != '':
      model_settings[key] = spec[key]

  # Load model
  model_pkl_fn = model_dir + model_settings['__model_nm'] + '_bestmodel.pkl'
  global model
  with open(model_pkl_fn, 'rb') as f:
    model = pickle.load(f)

  # Report
  print(f'Model successfully initialized. Settings:')
  public_settings = [key for key in model_settings if key[:2] != '__']
  for key in public_settings:
    print(f'\t{key}: {model_settings[key]}')

  global init_flag
  init_flag = True
  return

