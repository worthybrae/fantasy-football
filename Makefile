# Fantasy draft research tool — common commands.
# One-time setup: make setup && make refresh
# Draft night:    make up   (then open http://localhost:5173)

.PHONY: setup refresh api web up test build

setup: ## create venv, install python + web deps
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cd web && npm install

refresh: ## pull latest stats, depth charts, schedules/odds, ADP into DuckDB
	.venv/bin/python -m pipeline.refresh

api: ## run the FastAPI backend on :8000
	.venv/bin/uvicorn api.main:app --port 8000 --reload

web: ## run the Vite dev server on :5173
	cd web && npm run dev

up: ## run API + web together; Ctrl-C stops both
	@trap 'kill 0' EXIT INT TERM; \
	.venv/bin/uvicorn api.main:app --port 8000 & \
	( cd web && npm run dev ) & \
	wait

test: ## run the python test suite
	.venv/bin/pytest -q

build: ## typecheck + production-build the frontend
	cd web && npm run build
