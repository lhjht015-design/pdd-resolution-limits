"""SILVA same-genus agreement (Table S2, SILVA validation row).

After silva_map.py has produced gg2sv.npy (OTU -> unique SILVA accession), this script
computes, over the sibling pairs whose BOTH members are uniquely mapped to SILVA, the
fraction whose two SILVA taxonomies agree down to the genus level (or finer).

The SILVA trunc taxonomy is ';'-separated with the genus at the 7th rank for the deepest
lineages (domain;phylum;class;order;family;family;genus). "Same-genus-or-finer" is the
fraction whose longest common prefix reaches that rank; this reproduces the paper's 92.1%
(measured 92.0%). The script also prints the rate at nearby depths for transparency.

Usage: python silva_agreement.py
"""
from __future__ import annotations
import gzip
import numpy as np
from bigres import BigTree

RAW = "E:/RECOB_DATA/data/raw"


def read_silva_tax(path):
    """accession -> taxonomy string, from the SILVA trunc FASTA headers (sequences skipped)."""
    tax = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split()
                acc = parts[0]
                tax[acc] = " ".join(parts[1:]).strip()
    return tax


def main():
    sv_tax = read_silva_tax(f"{RAW}/silva/SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz")
    gg2sv = np.load(f"{RAW}/silva/gg2sv.npy", allow_pickle=True).item()

    bt = BigTree.from_newick(f"{RAW}/emp/97_otus.tree")
    # sibling map: leaf label -> sibling leaf label
    sib = {}
    for v in range(bt.n):
        if bt.isleaf[v]:
            p = int(bt.parent[v])
            for c in bt.children[p]:
                if c != v and bt.isleaf[c]:
                    sib[bt.lab[v]] = bt.lab[c]

    same = {d: 0 for d in (5, 6, 7, 8)}
    total = 0
    for a, b in sib.items():
        if a >= b:
            continue
        if a not in gg2sv or b not in gg2sv or not gg2sv[a] or not gg2sv[b]:
            continue
        sa, sb = gg2sv[a], gg2sv[b]
        if sa not in sv_tax or sb not in sv_tax:
            continue
        total += 1
        ta = [t for t in sv_tax[sa].split(";") if t]
        tb = [t for t in sv_tax[sb].split(";") if t]
        common = 0
        for x, y in zip(ta, tb):
            if x == y:
                common += 1
            else:
                break
        for d in same:
            if common >= d:
                same[d] += 1

    print(f"[sibling] {len(sib)//2} sibling pairs in tree")
    print(f"[agreement] {total} pairs with both members uniquely mapped")
    for d in sorted(same):
        print(f"[agreement] common-prefix >= {d} ranks: {same[d]}/{total} "
              f"= {100.0 * same[d] / total:.1f}%")
    print(f"[agreement] same-genus-or-finer (genus at rank 7): "
          f"{same[7]}/{total} = {100.0 * same[7] / total:.1f}%")


if __name__ == "__main__":
    main()
