SHELL := /bin/bash
COMPOSE := docker compose
SERVICE := lidar-sr

.PHONY: env build rebuild up shell down logs gpu-check test lint format jupyter tensorboard clean

env:
	@test -f .env || (cp .env.example .env && echo "Created .env; edit DATA_ROOT/OUTPUT_ROOT if needed.")

build: env
	$(COMPOSE) build

rebuild: env
	$(COMPOSE) build --no-cache

up: env
	$(COMPOSE) up -d

shell: env
	$(COMPOSE) run --rm $(SERVICE) bash

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f $(SERVICE)

gpu-check: env
	$(COMPOSE) run --rm $(SERVICE) python /usr/local/bin/verify_cuda.py

test: env
	$(COMPOSE) run --rm $(SERVICE) pytest -q

lint: env
	$(COMPOSE) run --rm $(SERVICE) ruff check .

format: env
	$(COMPOSE) run --rm $(SERVICE) ruff format .

jupyter: env
	$(COMPOSE) run --rm --service-ports $(SERVICE) \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token=''

tensorboard: env
	$(COMPOSE) run --rm --service-ports $(SERVICE) \
		tensorboard --logdir=/outputs --host=0.0.0.0 --port=6006

clean:
	$(COMPOSE) down --remove-orphans --volumes
