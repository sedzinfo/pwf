# -*- coding: utf-8 -*-
"""
One-off utility script: downloads every NLTK corpus/model. Not a
function module (nothing to import from here) -- guarded so it only
runs when executed directly, not on import.

Renamed from nltk.py: under that name it shadowed the real nltk package for
any script run from this folder (Python puts the script's own folder first
on sys.path), which broke `import nltk` in ml_nlp.py.
"""
if __name__ == "__main__":
    import nltk
    nltk.download("all")
