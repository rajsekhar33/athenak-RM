#!/usr/bin/env python3
"""Complete-inventory checks for the sf runners (a missing output file must fail, not
shrink the set of files that is compared).

  sf_inventory.py select DIR   run_select.sh case: DIR/sf_scalar and DIR/sf_longtrans
                               each hold exactly sfsel.00000.sf and sfsel.00001.sf
  sf_inventory.py toggle DIR   run_sparse.sh toggle case (DIR/in.athinput, nlim cycles,
                               an sf output every cycle, restart files every
                               <output3>/dcycle cycles, restart from rst file 1):
                               DIR/cont/<sub> holds files 0..nlim+1 (initial, one per
                               cycle, final), DIR/rest/<sub> the files after the
                               checkpoint, rst_cycle+1..nlim+1, for sub in sf_scalar,
                               sf_longtrans; DIR/cont/rst/*.00001.rst exists
Exit 0 when complete, 1 otherwise (listing what is missing or unexpected).
"""
import glob
import os
import re
import sys


def compare(d, base, expect):
    have = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.sf")))
    want = [f"{base}.{i:05d}.sf" for i in expect]
    missing = [f for f in want if f not in have]
    extra = [f for f in have if f not in want]
    ok = not missing and not extra
    print(f"  {d}: {len(have)} files, expected {len(want)}"
          f"{'' if ok else f'; missing {missing} unexpected {extra}'}"
          f" {'OK' if ok else 'FAIL'}")
    return ok


def block_key(text, block, key):
    m = re.search(r"<%s>(.*?)(?=\n<|\Z)" % re.escape(block), text, re.S)
    k = re.search(r"^\s*%s\s*=\s*(\S+)" % re.escape(key), m.group(1), re.M) if m else None
    if not k:
        raise SystemExit(f"{block}/{key} not found")
    return int(k.group(1))


def main():
    kind, d = sys.argv[1], sys.argv[2]
    ok = True
    if kind == "select":
        for sub in ("sf_scalar", "sf_longtrans"):
            ok &= compare(os.path.join(d, sub), "sfsel", range(2))
    elif kind == "toggle":
        text = open(os.path.join(d, "in.athinput")).read()
        nlim = block_key(text, "time", "nlim")
        rst_cycle = block_key(text, "output3", "dcycle")   # rst file 1
        if not glob.glob(os.path.join(d, "cont", "rst", "*.00001.rst")):
            print(f"  {d}/cont/rst: no restart file 1 FAIL")
            ok = False
        for sub in ("sf_scalar", "sf_longtrans"):
            ok &= compare(os.path.join(d, "cont", sub), "sfsp", range(nlim + 2))
            ok &= compare(os.path.join(d, "rest", sub), "sfsp",
                          range(rst_cycle + 1, nlim + 2))
    else:
        raise SystemExit(f"unknown kind {kind}")
    print("INVENTORY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
