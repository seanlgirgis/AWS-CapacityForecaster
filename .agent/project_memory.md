# Project Memory

**Current Stage:** EXECUTION
**Current Task:** Debugging AutoGluon Installation
**Last Updated:** 2026-01-31

## Context
We are currently working on getting the AutoGluon AutoML comparator to run correctly in a SageMaker environment. We have encountered `ModuleNotFoundError` and version conflicts with `numpy`, `pandas`, and `scikit-learn`.

## Active Work
- **CONSTITUTION IMPLEMENTATION**: Created `.agent/constitution.md` to establish project laws.
- **Dependency Resolution**: Edited `src/modules/module_04_model_training.py` to add a runtime dependency installer.
- **Pinned Versions**: Specifically pinned the following to resolve conflicts:
    - `pandas==2.1.4` (for numpy compat)
    - `scikit-learn<1.5.0`
    - `numpy<2.0.0`
    - `scipy<1.13.0`
    - `autogluon>=1.1.0`
- **Logic**: Added distinction between Local and SageMaker execution paths in `run_inner_script`.

## Protocol
- **READ THE CONSTITUTION**: Refer to `.agent/constitution.md` for governing rules.

## Next Steps
- Verify the `install_dependencies` wrapper works in the target environment.
- Run the full pipeline to ensure the AutoGluon model trains successfully.
