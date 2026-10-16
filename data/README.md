# Data setup

The scripts read raw data from a local directory. Each script has a module-level
`RAW` constant (e.g. `RAW = "E:/RECOB_DATA/data/raw"`); point it at your own copy.
The expected layout is:

```
<RAW>/
  emp/
    97_otus.tree                          # Greengenes 13.8, 97% OTU tree (Newick)
    97_otus.fasta                         # Greengenes 13.8, 97% OTU sequences (FASTA)
    emp_cr_gg_13_8.subset_2k.biom         # EMP subset_2k abundance table (BIOM/HDF5)
    emp_qiime_mapping_subset_2k.tsv       # EMP sample metadata (TSV)
  silva/
    SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz   # SILVA SSU 144, NR99, truncated
```

## Sources

- **Greengenes 13.8** (97% OTUs): <https://greengenes.secondgenome.com/> — the
  `gg_13_8_otus` release provides `97_otus.tree`, `97_otus.fasta`, and the taxonomy.
- **Earth Microbiome Project** (`subset_2k`): the `subset_2k` sample set is the one used in
  the PDD 2026 paper (*Deconvolving Phylogenetic Distance Mixtures*); the BIOM table and
  the QIIME mapping file are taken from that release (FTP/Zenodo mirror of EMP release 1).
- **SILVA SSU 144 NR99**: <https://www.arb-silva.de/download/archive/> — the
  `SSURef NR99` release 144, truncated FASTA with taxonomy in the header.

> Exact URLs should be confirmed against the latest mirrors at submission time.
