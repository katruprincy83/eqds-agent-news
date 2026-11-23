# make deploy        build ./app from the platform runtime + agent/, deploy the bundle, start the app
# make smoke         check /healthz and /info of the deployed app (add Q="question" for one real call)
PLATFORM ?= ../eqds-agents
TARGET ?= demo
PY ?= python3
Q ?=

.PHONY: build deploy smoke eval-setup eval-dataset eval

build:
	$(PY) scripts/build.py --platform $(PLATFORM) --target $(TARGET)

deploy: build
	databricks bundle deploy -t $(TARGET)
	databricks bundle run -t $(TARGET) news

smoke:
	$(PY) scripts/smoke.py $(if $(Q),--invoke "$(Q)")

# MLflow agent evaluation (experiment /Shared/eqds/eval/news-<target>): see eval/README.md
EVALPY ?= .venv/bin/python
eval-setup:
	uv venv .venv --python 3.12 -q || python3 -m venv .venv
	uv pip install -q -r eval-requirements.txt --python .venv/bin/python || .venv/bin/pip install -q -r eval-requirements.txt

eval-dataset:
	$(EVALPY) eval/build_dataset.py

eval: build
	$(EVALPY) eval/run_eval.py $(ARGS)
