PYTHON ?= python

.PHONY: install train test lint format app api benchmark docker-build docker-up

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) -m src.models.train

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	$(PYTHON) -m ruff check . --fix

app:
	streamlit run app.py

api:
	uvicorn src.models.api:app --host 0.0.0.0 --port 8000

benchmark:
	$(PYTHON) -m src.evaluation.performance --train --n-trials 1

docker-build:
	docker build -t healthcare-ml-system .

docker-up:
	docker compose up --build
