.PHONY: all tectonic latexmk fallback clean

all: tectonic

tectonic:
	tectonic --keep-logs --keep-intermediates main.tex

latexmk:
	latexmk -pdf main.tex

fallback:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf main.xdv
