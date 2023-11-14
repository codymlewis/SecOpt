all: clean appendix

clean:
	rm *.aux *.bbl *.blg *.log *.pdf

appendix:
	pdflatex extended_appendix.tex && \
	bibtex extended_appendix && \
	pdflatex extended_appendix.tex && \
	pdflatex extended_appendix.tex


gpusetup:
	poetry install && poetry run pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
