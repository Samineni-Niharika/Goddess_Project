# Implementation Guide

This is a navigation guide, not the full contract. Implementation agents must read each listed per-unit artifact and filter global tables by unit name before changing behavior.

- `streamlit_inference`: read `units/streamlit_inference/{behavior,bindings,unit_decomposition}.yaml`; filter model, preprocessing, taxonomy, and checkpoint rows. Evidence: multi-upload and per-image result/error behavior.
- `dataset_preparation`: read `units/dataset_preparation/{behavior,bindings,unit_decomposition}.yaml`; filter dataset layout/taxonomy rows. Evidence: topology, seed, and split ratios.
- `model_training`: read `units/model_training/{behavior,bindings,unit_decomposition}.yaml`; filter model/preprocessing/taxonomy rows. Evidence: checkpoint compatibility.
- `model_evaluation`: read `units/model_evaluation/{behavior,bindings,unit_decomposition}.yaml`; filter checkpoint/taxonomy rows. Evidence: separate metrics and confusion outputs.
- `calibration_and_ood`: read `units/calibration_and_ood/{behavior,bindings,unit_decomposition}.yaml`; filter calibration/state rows. Evidence: temperatures, grid constraints, and centroid acceptance.
- `cli_inference`: read `units/cli_inference/{behavior,bindings,unit_decomposition}.yaml`; filter checkpoint/preprocessing/taxonomy/calibration rows. Evidence: path validation and batch ordering.

Global contracts: `unit_graph.yaml`, `wire_contracts.yaml`, `shared_modules.yaml`, `cross_unit_state.yaml`, `seams.yaml`, and `migration_boundary.yaml`.
