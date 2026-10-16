"""SILVA validation — Step 1: map EMP (Greengenes) OTU sequences to SILVA.

Exact-sequence mapping between the Greengenes 97_otus.fasta (OTU id -> sequence) and
the SILVA 144 SSURef NR99 trunc fasta (accession -> sequence + taxonomy in the header).
Produces, for the EMP sibling pairs of the Greengenes tree, the set of pairs that map
unambiguously to SILVA (both siblings have a unique exact SILVA hit), plus their SILVA
taxonomy — this is the "reliable high-confidence mapping" the validation needs.

Usage: python silva_map.py
"""
from __future__ import annotations

import gzip
import time

import numpy as np

from bigres import BigTree

RAW = "E:/RECOB_DATA/data/raw"


def read_greengenes(path):
    """OTU id -> sequence (uppercased, no gaps)."""
    seq = {}
    cur = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seq[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
        if cur is not None:
            seq[cur] = "".join(buf)
    return seq


def normalize(s):
    return s.replace("U", "T").replace(".", "").replace("-", "")


def read_silva(path):
    """accession -> sequence; accession -> taxonomy (streaming over the gz)."""
    acc2seq = {}
    acc2tax = {}
    cur = None
    buf = []
    openf = gzip.open if path.endswith(".gz") else open
    with openf(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    acc2seq[cur] = "".join(buf)
                header = line[1:]
                parts = header.split(None, 1)
                acc = parts[0]
                tax = parts[1].strip() if len(parts) > 1 else ""
                cur = acc
                acc2tax[acc] = tax
                buf = []
            else:
                buf.append(line.strip().upper())
        if cur is not None:
            acc2seq[cur] = "".join(buf)
    return acc2seq, acc2tax


def main():
    t0 = time.time()
    gg = read_greengenes(f"{RAW}/emp/97_otus.fasta")
    gg = {k: normalize(v) for k, v in gg.items()}
    print(f"[gg] {len(gg)} sequences ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    sv_seq, sv_tax = read_silva(f"{RAW}/silva/SILVA_144_SSURef_NR99_tax_silva_trunc.fasta.gz")
    sv_seq = {k: normalize(v) for k, v in sv_seq.items()}
    print(f"[silva] {len(sv_seq)} sequences ({time.time()-t0:.0f}s)", flush=True)

    # sequence -> list of SILVA accessions (exact match index)
    t0 = time.time()
    seq2acc = {}
    for acc, s in sv_seq.items():
        seq2acc.setdefault(s, []).append(acc)
    print(f"[index] {len(seq2acc)} distinct sequences ({time.time()-t0:.0f}s)", flush=True)

    # map each Greengenes OTU
    gg2sv = {}
    for oid, s in gg.items():
        hits = seq2acc.get(s, [])
        if len(hits) == 1:
            gg2sv[oid] = hits[0]
        elif len(hits) > 1:
            gg2sv[oid] = None  # ambiguous
        else:
            gg2sv[oid] = None  # no hit
    n_hit = sum(1 for v in gg2sv.values() if v is not None)
    print(f"[map] {n_hit}/{len(gg)} OTUs have a unique exact SILVA hit "
          f"({time.time()-t0:.0f}s)", flush=True)

    # EMP sibling pairs from the Greengenes tree -> mapped pairs
    bt = BigTree.from_newick(f"{RAW}/emp/97_otus.tree")
    sib = {}
    for v in range(bt.n):
        if bt.isleaf[v]:
            p = bt.parent[v]
            if p >= 0:
                for c in bt.children[p]:
                    if c != v and bt.isleaf[c]:
                        sib[bt.lab[v]] = bt.lab[c]
    pairs = []
    for a, b in sib.items():
        if a < b and a in gg2sv and b in gg2sv and gg2sv[a] and gg2sv[b]:
            pairs.append((a, b, gg2sv[a], gg2sv[b]))
    print(f"[sibling] {len(sib)//2} sibling pairs in tree; "
          f"{len(pairs)} have BOTH siblings uniquely mapped to SILVA")
    # taxonomy of the mapped pairs
    rank_names = {"g": "genus", "f": "family", "o": "order", "c": "class",
                  "p": "phylum", "s": "species"}
    def silva_rank(tax, rank):
        # SILVA taxonomy uses ; separated, e.g. Bacteria;Firmicutes;...
        # last token is the finest; we return the string (no rank letters in trunc fasta)
        toks = [t for t in tax.split(";") if t]
        return toks[-1] if toks else ""
    sample = pairs[:20]
    print("\nexample mapped sibling pairs (OTU_a OTU_b -> SILVA acc, taxonomy):")
    for a, b, sa, sb in sample:
        ta = sv_tax.get(sa, ""); tb = sv_tax.get(sb, "")
        # deepest common SILVA taxon
        ta_t = ta.split(";"); tb_t = tb.split(";")
        common = 0
        for x, y in zip(ta_t, tb_t):
            if x == y:
                common += 1
            else:
                break
        print(f"  {a},{b} -> {sa},{sb}  common_tax={';'.join(ta_t[:common]) or '(none)'}")
    np.save(f"{RAW}/silva/gg2sv.npy", gg2sv, allow_pickle=True)
    print(f"\n[saved] gg2sv.npy")


if __name__ == "__main__":
    main()
