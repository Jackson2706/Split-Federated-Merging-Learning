.PHONY: assets validate paper clean
assets:
	MPLCONFIGDIR=/tmp/misfp-matplotlib python3 scripts/plot_assets.py
validate:
	python3 scripts/generate_results.py
paper: assets validate
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
clean:
	rm -f main.aux main.bbl main.blg main.log main.out
