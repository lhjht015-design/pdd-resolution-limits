# Structural Limits of Phylogenetic Resolution in Mixed Microbial Communities

Code and reproducibility for the RECOMB 2027 submission of the same name.

Profiling a mixed microbial community against a fixed reference phylogeny yields an
**aggregate** — an abundance-weighted mixture of tree distances — from which the
constituent placements and abundances must be recovered. This repository implements the
paper's two-layer theory:

1. **Fixed-support identifiability.** After linearizing the observation, identifiability
   is a rank condition `rank J = c + rank B` on a constant Jacobian, where the rank splits
   into the number of anchor classes plus the rank of a matrix assembled from two
   path-incidence systems of the reference tree. The criterion is *topology-only*: it does
   not depend on the branch lengths (sign coherence via a balanced terminal pairing on
   the block decomposition).

2. **Global support ambiguity.** For two placements sharing an edge, the degree-2-vertex
   swap is a sufficient interior collision (Theorem 6), conjecturally the only one
   (Conjecture 1, verified exhaustively for n = 5, 6, 7). The resolution
   deficiency
   `Δ = 2k − rank J` then quantifies how much of a community is invisible to the
   measurement; a sibling pair of anchors always costs at least one degree of freedom.

The biological half instantiates the sibling obstruction in real communities (Earth
Microbiome Project, SILVA-independent validation, and a real counterfactual: two interior
community states whose aggregate observation agrees to machine precision).

## Repository layout

```
src/                  core scripts (and their full import closure)
data/README.md        where to obtain the raw data and how to lay it out
requirements.txt      Python dependencies
```

Only the scripts that produce the numbers in the paper are kept under `src/`; the many
intermediate *exploration* scripts used during the research are omitted.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `h5py`, `pandas` (see `requirements.txt`).

```
pip install -r requirements.txt
```

## Data

The scripts read raw data from a local path (see `data/README.md` for exact files and
download sources). The path is hard-coded as a module-level `RAW` constant in each script;
edit it to point at your copy. The required files are:

- Greengenes 13.8 97% OTUs: `97_otus.tree`, `97_otus.fasta`
- EMP `subset_2k`: `emp_cr_gg_13_8.subset_2k.biom`, `emp_qiime_mapping_subset_2k.tsv`
- SILVA SSU 144 NR99: `SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz`

## Reproducing the results

| Paper result | Script | Command |
|---|---|---|
| Cherry identity and $\Delta(\{a,b,c\})\ge 1$ (Table S1, rows 1–4) | `cherry_regression.py` | `python cherry_regression.py` |
| Rank split `rank J = c + rank B` (Table S1, row 5) | `delta_test.py` | `python delta_test.py` |
| Global $k=2$ collision + disjoint-support (Table S1, rows 7–9) | `global_collision.py` | `python global_collision.py` |
| Shared-edge collision necessity (Theorem 7; Supplement) | `shared_edge_regression.py` | `python shared_edge_regression.py --ns 5,6,7` |
| Necessity proof cross-check (algebraic reduction vs.\ LP oracle) | `reduction_crosscheck.py` | `python reduction_crosscheck.py --ns 5,6,7` |
| Counterfactual invisibility (Table S1, row 10; Table S2, last row) | `counterfactual.py` | `python counterfactual.py` |
| EMP sibling co-occurrence (Table S2) | `sibling_census.py` | `python sibling_census.py` |
| SILVA-independent validation (Table S2) | `silva_map.py` + `silva_agreement.py` | `python silva_map.py`, then `python silva_agreement.py` |
| Identifiability criterion, $k=2$ / general | `criterion_k2.py` / `criterion_kgeneral.py` | `python criterion_k2.py` |

`emp_probe.py` computes the environment / permutation-null *negative controls* (the
confounding load) reported in the paper as ruled-out hypotheses; it is **not** the source
of the headline prevalence numbers.

Each script prints its checks and the relevant counts; run with `--help` for options
(number of trees, seeds, length regimes, etc.). The core engines are `bigres.py` /
`resolution.py` (deficiency $\Delta$), `polytope.py` (observation polytope), and
`probe_identifiability.py` (tree + Jacobian infrastructure).

## License

MIT — see `LICENSE`.
