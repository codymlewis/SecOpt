all: clean appendix

clean:
	rm *.aux *.bbl *.blg *.log *.pdf

appendix:
	pdflatex extended_appendix.tex && \
	bibtex extended_appendix && \
	pdflatex extended_appendix.tex && \
	pdflatex extended_appendix.tex