# -*- coding: utf-8 -*-
"""
One-off utility script: downloads every NLTK corpus/model. Not a
function module (nothing to import from here) -- guarded so it only
runs when executed directly, not on import.

Note: this file is named nltk.py in the same directory that Python
resolves imports from. Any other script in this folder that does
`import nltk` (meaning the real NLTK package) will instead import this
file, since Python's default `sys.path` puts the running script's own
directory first. Consider renaming this file (e.g. download_nltk_data.py)
if anything in this project ever needs the real nltk package.
"""
if __name__ == "__main__":
    import nltk
    nltk.download("all")
