# Model

from __future__ import absolute_import, division
from __future__ import print_function
import sys, string, pickle, subprocess, os, datetime, gzip, time
from collections import defaultdict, OrderedDict
import glob
import numpy as np, pandas as pd
from scipy.special import expit

nts = list('ACGT')
curr_fold = os.path.dirname(os.path.realpath(__file__))
nt_to_idx = {nt: nts.index(nt) for nt in nts}
models_design = pd.read_csv(curr_fold + '/models.csv', index_col = 0)
model_dir = curr_fold + '/params/'
model = None
init_flag = False

model_settings = {
  'celltype': 'mES',
  'base_editor': 'eA3A',
  '__model_nm': 'mES_12kChar_eA3A',
  '__train_test_id': '0_old',
}

model_nm_mapper = {}
for idx, row in models_design.iterrows():
  inp_set = (row['Public base editor'], row['Celltype'])
  model_nm = row['Model name']
  model_nm_mapper[inp_set] = model_nm

'''
  Intended usage: 
    import predict as be_efficiency_model
    be_efficiency_model.init_model(base_editor = '', celltype = '')

    scalar = be_efficiency_model.predict(seq)
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

def estimate_conversion_parameters(csv_fn):
  '''
    Expect columns:
      - "Target sequence", 50-nt DNA strings that are individually valid input to the model
      - "Observed fraction of sequenced reads with base editing activity". Should be in [0, 1].

    Introduces columns (will overwrite if exists):
    - Predicted frequency
  '''
  assert init_flag, f'Call .init_model() first.'

  obs_col = 'Observed fraction of sequenced reads with base editing activity'
  seq_col = 'Target sequence'

  dd = defaultdict(list)
  df = pd.read_csv(csv_fn)
  assert 0 <= min(df[obs_col]) and max(df[obs_col]) <= 1, f'Error: {obs_col} should be in the range [0, 1]'
  print(f'Making predictions for {len(df)} target sequences...')
  for idx, row in df.iterrows():
    pred = predict(row[seq_col])
    dd['Predicted frequency'].append(pred)
  print('Done')

  for col in dd:
    df[col] = dd[col]

  # Estimate mean and variance for converting logistic scalars to [0, 1] range with minimum L2 loss
  print('Estimating parameters...')

  preds = np.array(df['Predicted frequency'])
  obs = np.array(df[obs_col])
  def opt_f(params):
    [mean, std] = params
    conv_p = expit(preds * std + mean)
    return np.sum((conv_p - obs)**2)

  from scipy.optimize import minimize
  res = minimize(
    opt_f, 
    (0, 1), 
    bounds = (
      (None, None),
      (0, None),
    ),
  )
  print(f'Optimization results:\n{res}')
  [mean, var] = res['x']
  print(f'Optimization loss: {res["fun"]}')

  return {
    'Inferred mean': mean,
    'Inferred std': std,
  }


def predict(seq, mean = None, std = 1.5):
  assert len(seq) == 50, f'Error: Sequence provided is {len(seq)}, must be 50 (positions -19 to 30 w.r.t. gRNA (positions 1-20)'

  assert init_flag, f'Call .init_model() first.'
  seq = seq.upper()

  x = __featurize(seq)
  y = float(model.predict(x))

  conv_y = np.nan
  if mean is not None:
    assert std > 0, 'Error: Provided std is non-positive'
    conv_y = expit(y * std + mean)

  return {
    'Predicted logit score': y,
    'Predicted fraction of sequenced reads with base editing activity': conv_y,
  }


def init_model(base_editor = '', celltype = ''):
  # Check
  ok_editors = set(models_design['Public base editor'])
  assert base_editor in ok_editors, f'Bad base editor name\nAvailable options: {ok_editors}'
  ok_celltypes = set(models_design["Celltype"])
  assert celltype in ok_celltypes, f'Bad celltype\nAvailable options: {ok_celltypes}'

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

  model_settings['__model_nm'] = model_nm_mapper[(base_editor, celltype)]

  # Load model
  model_pkl_fn = model_dir + model_settings['__model_nm'] + '.pkl'
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

