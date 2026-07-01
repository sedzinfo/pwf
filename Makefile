.PHONY: docs docs-serve

docs:
	python -m pdoc pwf -o docs

docs-serve:
	python -m pdoc pwf
