PY = conda run -n footnote python -I

.PHONY: setup data test audit humans numeric-check budget

setup:
	conda env create -f environment.yml

data:
	$(PY) -m footnote.cli data fetch

test:
	$(PY) -m pytest

audit:
	$(PY) -m footnote.cli audit

humans:
	$(PY) -m footnote.cli humans

numeric-check:
	$(PY) -m footnote.cli numeric-check

budget:
	$(PY) -m footnote.cli budget

.PHONY: ingest retrieval-check updates-demo api ui charts secrets

ingest:
	$(PY) -m footnote.cli ingest

retrieval-check:
	$(PY) -m footnote.cli retrieval-check

updates-demo:
	$(PY) -m footnote.cli updates-demo

api:
	conda run -n footnote uvicorn footnote.app.api:app

ui:
	conda run -n footnote streamlit run src/footnote/app/ui.py

charts:
	$(PY) -m footnote.cli charts

secrets:
	$(PY) -m pytest tests/test_no_secrets.py
