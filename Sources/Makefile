# H-SFP / E-HSFP — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python
HSFP_CFG := configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml

.PHONY: help install install-cuda data-check check smoke \
        reproduce-journal reproduce-camera-ready fairness figures clean clean-pyc

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies (assumes torch already installed)
	$(PY) -m pip install -r requirements.txt

install-cuda:  ## Install a CUDA 12.8 torch build, then the rest
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	$(PY) -m pip install -r requirements.txt

data-check:  ## Verify the expected data/ layout exists (see docs/datasets.md)
	@for d in cifar HAM10000 ISIC2018; do \
		if [ -e data/$$d ]; then echo "ok   data/$$d"; else echo "MISS data/$$d (see docs/datasets.md)"; fi; \
	done

check:  ## Lightweight sanity checks (no training)
	$(PY) -m py_compile $$(git ls-files '*.py')
	$(PY) main.py --list >/dev/null && echo "registry OK"
	$(PY) -m camera_ready.partition >/dev/null && echo "partition self-test OK"

smoke:  ## One tiny end-to-end H-SFP run (fast sanity)
	$(PY) main.py --task classification --method h-sfp \
		--cfg configs/camera_ready/smoke/hsfp_smoke.yaml --seed 0

reproduce-journal:  ## Run the full E-HSFP journal suite
	bash scripts/journal_experiments/run_all_journal.sh
	$(PY) scripts/journal_experiments/collect_results.py

reproduce-camera-ready:  ## Run all camera-ready experiments (set SMOKE=1 for a fast pass)
	bash scripts/camera_ready/run_all_camera_ready.sh
	$(PY) scripts/camera_ready/collect_camera_ready.py
	$(PY) scripts/camera_ready/make_tables.py

fairness:  ## Generate the baseline-fairness summary + LaTeX
	$(PY) scripts/camera_ready/gen_fairness.py

figures:  ## Regenerate the communication-overhead figure
	$(PY) tools/plot_communication.py communication_overhead.png

clean-pyc:  ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

clean: clean-pyc  ## Remove caches and local run artifacts (keeps data/ and checkpoints)
	rm -rf logs/* 2>/dev/null || true
