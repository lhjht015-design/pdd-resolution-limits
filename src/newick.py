"""Newick parser for the Greengenes 97% reference tree.

Greengenes format: tips are `OTU_id:branch`, internal nodes are
`( ... )support:branch` where `support` is a float (bootstrap/confidence, may be
absent), branch lengths are floats after `:`. The whole string ends with `;`.

The parser returns plain parallel arrays; node 0 is the topmost node (the root of
the file, which may be a multifurcation for unrooted-style reference trees).
"""
from __future__ import annotations


def parse_newick(text):
    parent = []
    children = []
    label = []
    length = []
    isleaf = []

    def new(lab, leaf):
        parent.append(-1)
        children.append([])
        label.append(lab)
        length.append(0.0)
        isleaf.append(leaf)
        return len(parent) - 1

    root = new(None, False)
    stack = [root]
    s = text.strip()
    n = len(s)

    def read_label(i):
        """Read a (possibly single-quoted) label starting at i; return (label, i_next).
        Quoted labels may contain ':' and ',' and are read to the matching quote."""
        if i < n and s[i] in "'\"":
            q = s[i]
            j = i + 1
            while j < n and s[j] != q:
                j += 1
            return s[i + 1:j], j + 1
        j = i
        while j < n and s[j] not in ':,' and s[j] not in '()' and s[j] != ';':
            j += 1
        return s[i:j].strip(), j

    def read_length(i):
        """Read ':number' starting at i (s[i] == ':'). Return (float, i_next)."""
        i += 1
        j = i
        while j < n and s[j] not in ',()' and s[j] != ';':
            j += 1
        tok = s[i:j].strip()
        return (float(tok) if tok else 0.0), j

    i = 0
    while i < n:
        c = s[i]
        if c == '(':
            nd = new(None, False)
            parent[nd] = stack[-1]
            children[stack[-1]].append(nd)
            stack.append(nd)
            i += 1
        elif c == ')':
            nd = stack.pop()
            i += 1
            lab, i = read_label(i)
            if lab:
                label[nd] = lab
            if i < n and s[i] == ':':
                length[nd], i = read_length(i)
        elif c == ',':
            i += 1
        elif c == ';':
            i += 1
        else:
            lab, i = read_label(i)
            nd = new(lab, True)
            parent[nd] = stack[-1]
            children[stack[-1]].append(nd)
            i = i  # read_label already advanced
            if i < n and s[i] == ':':
                length[nd], i = read_length(i)
    return parent, children, label, length, isleaf, root


def load_newick(path):
    with open(path) as f:
        text = f.read()
    return parse_newick(text)


if __name__ == "__main__":
    import sys
    p, ch, lab, ln, leaf, root = load_newick(sys.argv[1])
    n = len(p)
    nleaf = sum(leaf)
    print(f"nodes={n}  leaves={nleaf}  internal={n - nleaf}")
    print(f"root={root}  root children={len(ch[root])}")
    # degree distribution of internal nodes
    from collections import Counter
    deg = Counter(len(ch[u]) for u in range(n) if not leaf[u] and u != root)
    deg[root] = len(ch[root])
    print("internal degree distribution (deg: count):",
          dict(sorted(Counter(len(ch[u]) for u in range(n) if not leaf[u]).items())))
    print("max depth:", end=" ")
    depth = {root: 0}
    stack = [root]
    while stack:
        u = stack.pop()
        for w in ch[u]:
            depth[w] = depth[u] + 1
            stack.append(w)
    print(max(depth.values()), "  avg leaf depth:",
          round(sum(depth[u] for u in range(n) if leaf[u]) / nleaf, 1))
    # sample labels
    tips = [lab[u] for u in range(n) if leaf[u]][:5]
    print("sample tip labels:", tips)
