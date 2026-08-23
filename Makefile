# Fantasy draft research tool — common commands.
# One-time setup: make setup && make refresh
# Draft night:    make up   (then open http://localhost:5173)

.PHONY: setup refresh api web up test build espn-import fit-managers sim mock-backfill farm-mocks corpus-report

setup: ## create venv, install python + web deps
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cd web && npm install

refresh: ## pull latest stats, depth charts, schedules/odds, ADP into DuckDB
	.venv/bin/python -m pipeline.refresh

espn-import: ## import ESPN draft history: make espn-import LEAGUE=<url-or-id>
	.venv/bin/python -m pipeline.import_league "$(LEAGUE)"

draft-corpus: ## fold draft history into the cross-league corpus: make draft-corpus LEAGUE=<id>
	.venv/bin/python -m pipeline.draft_log "$(LEAGUE)"

mock-backfill: ## harvest completed ESPN mock drafts from data/leagues/*.duckdb into the cross-league corpus
	.venv/bin/python -m pipeline.mock_backfill

farm-mocks: ## play live ESPN mock drafts and record them: make farm-mocks N=5 [SEED=0]
	# N is how many COMPLETED drafts to record, not how many rooms to try --
	# a room that fills up, never starts, or ends early does not count against
	# it. Each draft is 20-40 minutes of wall clock, one at a time.
	.venv/bin/python -m pipeline.mock_farm "$(or $(N),1)" --seed=$(or $(SEED),0)

corpus-report: ## describe the mock draft corpus: humans vs ESPN autodraft, by round bucket and position -- read-only
	.venv/bin/python -m pipeline.mock_backfill --report

fit-managers: ## fit per-manager pick models from imported draft history (REDUCED=1 also measures reduced personal models -- slow)
	# filter-out, not a bare $(if): $(if) tests emptiness, so REDUCED=0 would
	# have switched the slow path ON.
	.venv/bin/python -m pipeline.fit_managers $(if $(filter-out 0 no false,$(REDUCED)),--reduced,)

sim: ## run the draft simulator: make sim SLOT=4 [ROLLOUTS=300]
	.venv/bin/python -m pipeline.run_sim "$(SLOT)" $(ROLLOUTS)

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
