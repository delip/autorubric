.DEFAULT_GOAL := build

TEX_MAIN := main
LATEXMK := latexmk
LATEXMK_OPTS := -pdf -interaction=nonstopmode -halt-on-error

.PHONY: build clean

build:
	$(LATEXMK) $(LATEXMK_OPTS) $(TEX_MAIN).tex

clean:
	$(LATEXMK) -C $(TEX_MAIN).tex
