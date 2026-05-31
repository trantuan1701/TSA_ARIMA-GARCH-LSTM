.PHONY: all tectonic latexmk fallback package-final-pdf clean

FINAL_PDF ?= final_vnindex_volatility_financial_econometrics_v3.pdf
ROOT_PDF ?= ../paper.pdf

all: tectonic package-final-pdf

tectonic:
	tectonic --keep-logs --keep-intermediates main.tex

latexmk:
	latexmk -pdf main.tex

fallback:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

package-final-pdf:
	cp main.pdf $(FINAL_PDF)
	cp main.pdf $(ROOT_PDF)

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf main.xdv
