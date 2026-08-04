# Voice Clone Agent Makefile
.PHONY: help build up down logs ps download models shell backend-shell frontend-shell eval-stt eval-tts eval-e2e clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build docker images
	docker compose build

up: ## Start all services (detached)
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Tail logs
	docker compose logs -f --tail=200

ps: ## Show container status
	docker compose ps

download: ## Download all model weights into ./data/models
	docker compose run --rm --remove-orphans backend python /app/scripts/download_models.py

models: download ## Alias for download

shell: backend-shell ## Drop into backend shell (default)

backend-shell: ## Drop into backend container shell
	docker compose exec backend /bin/bash

frontend-shell: ## Drop into frontend container shell
	docker compose exec frontend /bin/bash

eval-stt: ## Run STT evaluation
	docker compose run --rm --remove-orphans backend python /app/../evaluation/evaluate_stt.py

eval-tts: ## Run TTS evaluation
	docker compose run --rm --remove-orphans backend python /app/../evaluation/evaluate_tts.py

eval-e2e: ## Run end-to-end evaluation
	docker compose run --rm --remove-orphans backend python /app/../evaluation/evaluate_e2e.py

benchmark: ## Run benchmark
	docker compose run --rm --remove-orphans backend python /app/scripts/benchmark.py

clean: ## Remove all data (speakers, calls, models) - DESTRUCTIVE
	@read -p "This deletes ./data/* contents. Continue? [y/N] " ans; [ "$$ans" = "y" ] || exit 1
	rm -rf ./data/speakers/* ./data/calls/* ./data/models/*
	touch ./data/speakers/.gitkeep ./data/calls/.gitkeep ./data/models/.gitkeep

dev-backend: ## Run backend in dev mode without docker (requires local venv)
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend: ## Run frontend in dev mode without docker
	cd frontend && streamlit run streamlit_app/main.py --server.port=8501
