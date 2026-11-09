.PHONY: docs docs-serve test reference

RSCRIPT ?= Rscript

docs:
	python -m pdoc pwf -o docs

docs-serve:
	python -m pdoc pwf

test:
	python -m pytest

# Regenerate rwf reference outputs used by the parity tests (needs R with rwf installed).
reference:
	$(RSCRIPT) tests/reference/make_reference.R
