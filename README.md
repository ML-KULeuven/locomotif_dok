<h1 align="left">LoCoMotif-DoK:

Steering the LoCoMotif: Using Domain Knowledge in Time Series Motif Discovery </h1>

This repository contains the implementation of the time series motif discovery (TSMD) method called LoCoMotif-DoK (LoCoMotif-<u>Do</u>main <u>K</u>nowledge). Domain knowledge is represented as hard and soft constraints on the motif (set)s to be discovered. Hard constraints can be defined as predicates (i.e., functions that return a boolean) of motifs, pairs of motifs, motif sets, and pairs of motif sets. Soft constraints are represented as desirability functions (that return a scalar between 0 and 1) of motif sets. See below for details.

LoCoMotif-DoK is an extension of LoCoMotif. LoCoMotif-DoK is fully backwards-compatible with LoCoMotif and supports all its properties: discovering motifs of different lengths (*variable-length* motifs), exhibit slight temporal differences (*time-warped* motifs), and span multiple dimensions (*multivariate* motifs)

LoCoMotif: 
- Daan Van Wesenbeeck, Aras Yurtman, Wannes Meert, and Hendrik Blockeel, "LoCoMotif: discovering time-warped motifs in time series," Data Mining and Knowledge Discovery, May 2024. https://doi.org/10.1007/s10618-024-01032-z
- implementation: https://github.com/ML-KULeuven/locomotif

LoCoMotif-DoK: 
- Aras Yurtman, Daan Van Wesenbeeck, Wannes Meert, and Hendrik Blockeel, "Steering the LoCoMotif: Using Domain Knowledge in Time Series Motif Discovery," arXiv, February 2025

## Representation of Domain Knowledge 🧠

We first define a motif set as a set of $k$ time segments that are similar to each other: 
$\mathcal{M} = \\{ \beta_1, \ldots, \beta_k \\}$ \
One of these segments is the representative segment ($\alpha$) of the motif set. 

Then, domain knowledge can be represented as the following types of hard and soft constraints: 

1. $H^\text{mset}(\mathcal{M})$: hard constraint on a motif set\
  Three special cases:
    - $H^\text{mot-repr}(\alpha)$ is the hard constraint on the representative motif
    - $\forall \beta \in \mathcal{M}: H^\text{mot}(\beta)$ \
      where $H^\text{mot}(\beta)$ is the hard constraint on every motif
    - $\forall \beta, \beta' \in \mathcal{M},\; \beta \neq \beta' : H^\text{mots-same}(\beta, \beta')$ \
      where $H^\text{mots-same}(\beta, \beta')$ is the hard constraint on every pair of motifs in the same motif set

2. $H^\text{msets}(\mathcal{M}, \mathcal{M}')$: hard constraint on a pair of motif sets \
  A special case:
   - $\forall \beta \in \mathcal{M},\; \forall \beta' \in \mathcal{M}' : H^\text{mots-diff}(\beta, \beta')$ \
     where $H^\text{mots-diff}(\beta, \beta')$ is the hard constraint on every pair of motifs from the two motif sets

3. $D(\mathcal{M})$: desirability function defined on a motif set

By enumerating the motif sets to be discovered as $\mathcal{M}_1, \ldots, \mathcal{M}_\kappa$, 
the constraints of types 1 and 3 can be defined separately for each motif set, and constraints of type 2 can be defined separately for each pair of distinct motif sets.


## Installation

First, clone the repository:
```
git clone https://github.com//aras-y/locomotif_weakly_supervised.git -b locomotif_dok_v3
```
Then, navigate into the directory and build the package from source:
```
pip install .
```

## Usage

We first explain how to use LoCoMotif (without domain knowledge), and then LoCoMotif-DoK (with domain knowledge).

### LoCoMotif 🚂

A time series is representated as 2d numpy array of shape `(n, d)` where `n` is the length of the time series and `d` the number of dimensions:

```python
f = open(os.path.join("..", "examples", "datasets", "mitdb_patient214.csv"))
ts = np.array([line.split(',') for line in f.readlines()], dtype=np.double)

print(ts.shape)
>>> (3600, 2)
```

To apply LoCoMotif to the time series, simply import the `locomotif` module and call the ``apply_locomotif`` method with suitable parameter values. Note that, we highly advise you to first z-normalize the time series.
```python
import locomotif.locomotif as locomotif 
ts = (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)
motif_sets = locomotif.apply_locomotif(ts, l_min=216, l_max=360, rho=0.6)
```
The parameters `l_min` and `l_max` respectively represent the minimum and maximum motif length of the representative of a motif set. The parameter ``rho`` determines the ''strictness'' of the LoCoMotif method; or in other words, how similar the subsequences in a motif set are expected to be. The best value of ``rho`` depends heavily on the application; however, in most of our experiments, a value between ``0.6`` and ``0.8`` always works relatively well.  
Optionally, we allow you to choose the allowed overlap between motifs through the `overlap` parameter (which lies between `0.0` and `0.5`), the number of motif sets to be discovered through the `nb` parameter (by default, `nb=None` and LoCoMotif finds all motifs), and whether to use time warping or not through the `warping` parameter (either `True` or `False`)

The result of LoCoMotif is a list of ``(candidate, motif_set)`` tuples, where each `candidate` is the representative subsequence (the most "central" subsequence) of the corresponding `motif_set`. Each `candidate` is a tuple of two integers `(b, e)` representing the start- and endpoint of the corresponding time segment, while each `motif_set` is a list of such tuples.

```python
print(motif_sets)
>>> [((2666, 2931), [(2666, 2931), (1892, 2136), (1038, 1332), (2334, 2665), (628, 1035), (1589, 1892), (1, 260)]), ((2931, 3155), [(2931, 3155), (2136, 2333), (1332, 1558)])]
```

### LoCoMotif-DoK 🚂🧠

The types of constraints defined above can be passed as the following input arguments: 
 - $H^\text{mset}$: `h_mset_all`
 - $H^\text{mot-repr}$: `h_mot_repr_all`
 - $H^\text{mot}$: `h_mot_all`
 - $H^\text{mots-same}$: `h_mots_same_all`
 - $D$: `desir_all`
 - $H^\text{msets}$: `h_msets_pairwise_all`
 - $H^\text{mots-diff}$: `h_mots_diff_all`

The first 5 can be a list of $\kappa$ functions to apply on each motif set to discover, and the last two can be a nested list of functions of shape $\kappa$-by-$\kappa$. 
Each argument can be provided as a single function, which is then applied to every (pair of) motif set(s) to be discovered. 
They can also be None,

Every function has to be numba-compiled for efficiency. Example functions can be found in the catalogue of constraints: `locomotif/catalogue_of_constraints.py`, which also contains functions to combine multiple constraints of the same type with AND and OR operations.

To apply LoCoMotif-DoK to the time series, call the `apply_locomotifdok` method with suitable parameter values:

```python
import locomotif.locomotif_dok as lcm_dok 
import locomotif.catalogue_of_constraints as cat

ts = (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)

# Add hard constraints on cardinalities of the two motif sets to be discovered:
h_mset_all = [cat.h_mset_cardinality_min_max(3, 7), 
              cat.h_mset_cardinality_min_max(5, 10)]

motif_sets = lcm_dok.apply_locomotifdok(ts, l_min=216, l_max=360, rho=0.6, 
                                        h_mset_all=h_mset_all)
```

### Visualization

We also include a visualization module, ``visualize``, to plot the time series together with the found motifs:
```python
import locomotif.visualize as visualize
import matplotlib.pyplot as plt

fig, ax = visualize.plot_motif_sets(ts, motif_sets)
plt.show()
```

More examples can be found in the `example` folder.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
