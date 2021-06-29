# be_predict_efficiency

## Dependencies
- Python 3.7 and standard packages (pickle, scipy, numpy, pandas)
- scikit-learn==0.20.3
- Biopython==1.73

## Installation
Clone this github repository, then set up your environment to import the `predict.py` script in however is most convenient for you. In python, for instance, you may use the following at the top of your script to import the model.

```python
import sys
sys.path.append('/directory/containing/local/repo/clone/')
from be_predict_efficiency import predict as be_efficiency_model
```

## Usage
```python
from be_predict_efficiency import predict as be_efficiency_model
be_efficiency_model.init_model(base_editor = 'BE4', celltype = 'mES')
```

Note: Supported cell types are `['mES', 'HEK293']` and supported base editors are `['ABE', 'ABE-CP1040', 'BE4', 'BE4-CP1028', 'AID', 'CDA', 'eA3A', 'evoAPOBEC', 'eA3A-T44DS45A', 'BE4-H47ES48A', 'eA3A-T31A', 'eA3A-T31AT44A', 'BE4-H47ES48A']`. Not all combinations of base editors and cell types are supported -- refer to `models.csv`.

Available C-to-G base editors (CGBEs): ['CG-eA3A', 'CG-689', 'CG-APOBEC1', 'CG-POLD2-APOBEC1-X', 'CG-RBMX-eA3A-X-HF', 'CG-RBMX-eA3A-X', 'CG-X-689-X-RBMX', 'CG-X-APOBEC1-X-HF', 'CG-X-EE-X-X', 'CG-eA3A-dead', 'CG-EE']

If your cell type of interest is not included here, we recommend using mES. Major base editing outcomes are fairly consistent across cell-types, though rarer outcomes including cytosine transversions are known to depend on cell-type to some extent.

```python
pred_d = be_efficiency_model.predict(seq)
```

`seq` is a 50-nt string of DNA characters, spanning from positions -19 to 30 where positions 1-20 are the spacer, an NGG PAM occurs at positions 21-23, and position 0 is used to refer to the position directly upstream of position 1. 

`pred_d` is a dict with the following keys.
- 'Predicted logit score', centered at 0 with standard deviation = 2
- 'Predicted fraction of sequenced reads with base editing activity', either `np.nan` or in the range `[0, 1]`.

### Example usage
```python
from be_predict_efficiency import predict as be_efficiency_model
be_efficiency_model.init_model(base_editor = 'BE4', celltype = 'mES')

seq = 'TATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTACCGTGGACGTGAG'

pred_d = be_efficiency_model.predict(seq)
```

## Obtaining predictions of the fraction of sequenced reads with base editing activity
Base editing efficiency varies not just by sequence context and gRNA but also by cell-type, experimental condition, method of delivery or expression of the base editor. Providing predictions on the scale of the fraction of sequenced reads with base editing activity in your system of choice therefore requires you to provide the model with more information.

Our model uses two parameters, a mean and standard deviation, to convert the logit score into the range `[0, 1]` representing the fraction of sequenced reads with base editing activity: 

`1 / (1 + exp(-(logit_score * std + mean)))`.

Note that these models were trained on a comprehensive, minimally biased library containing sequence contexts with all 4-mers surrounding substrate nucleotides from positions 1-11. This minimally biased set of sequence contexts necessarily contains some sequence contexts that users are unlikely to select for base editing; users are likely to work with sequence contexts with above average editing efficiency in the space of all possible sequence contexts. As a result, simply using the average editing efficiency from your observed data will likely lead to predictions overestimating reality.

### Estimating mean and standard deviation from observed data
```python
conv_d = be_efficiency_model.estimate_conversion_parameters(csv_fn)
```

`csv_fn` is an absolute path to a CSV file, containing the columns:
- 'Target sequence', 50-nt DNA strings that are individually valid input to the model
- 'Observed fraction of sequenced reads with base editing activity'. Should be in [0, 1].

`conv_d` is a dict with the following keys.
- 'Inferred mean'
- 'Inferred std'

### Specifying a known mean and/or standard deviation
Values for the mean and standard deviation can be provided to the function as follows. We recommend using the inferred mean and std from your observed data to avoid predictions overestimating or underestimating reality. 

```python
pred_d = be_efficiency_model.predict(seq, mean = <float>, std = <float>)

# or

pred_d = be_efficiency_model.predict(seq, mean = <float>)
```

`pred_d` is a dict with the following keys.
- Predicted logit score, centered at 0 with standard deviation = 2
- Predicted fraction of sequenced reads with base editing activity, in the range `[0, 1]`. 

If `std` is not specified, the default value is 2.

## Contact
maxwshen at gmail.com

### License
https://www.crisprbehive.design/about
