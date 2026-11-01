#!/usr/bin/env python3
"""Self-test of the .sf/.spec readers on damaged files.

usage: reader_selftest.py FILE.sf FILE.spec

Each good file must read; an empty file, a header cut short and a payload cut short
must raise ValueError (not hang or return short arrays). Prints READER SELFTEST:
PASS or FAIL.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # python3 -I does not add the script directory
sys.path.insert(0, os.path.join(HERE, "..", "spec"))
from sf_reference import read_sf  # noqa: E402
from spec_reference import read_spec  # noqa: E402


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for good, reader in ((sys.argv[1], read_sf), (sys.argv[2], read_spec)):
            raw = open(good, "rb").read()
            ext = os.path.splitext(good)[1]
            cases = {"good": raw, "empty": b"", "header cut": raw[:raw.index(b"\n") + 1],
                     "payload cut": raw[:-8]}
            for name, data in cases.items():
                path = os.path.join(tmp, name.replace(" ", "_") + ext)
                with open(path, "wb") as f:
                    f.write(data)
                try:
                    reader(path)
                    good_read = True
                except ValueError:
                    good_read = False
                passed = good_read == (name == "good")
                ok &= passed
                print(f"{'ok  ' if passed else 'FAIL'} {ext} {name}: "
                      f"{'read' if good_read else 'ValueError'}")
    print("READER SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
