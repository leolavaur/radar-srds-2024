.PHONY: build/main.pdf all clean

all: build/main.pdf

build/main.pdf: src/main-lncs.tex
	latexmk -interaction=nonstopmode -shell-escape -pdf -cd -outdir=../build -jobname=main src/main-lncs.tex

clean:
	rm build/*