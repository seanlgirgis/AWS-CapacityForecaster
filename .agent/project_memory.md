# Project Memory

**Current Stage:** EXECUTION (Debugging & Documentation)
**Current Task:** Resolving AutoGluon Dependencies & Documenting Progress
**Last Updated:** 2026-01-31

## 1. Project Status
We have successfully implemented the core modular pipeline (Modules 00-05) and are currently focused on stabilizing the Model Training module (04) against dependency conflicts.

**Key Documents:**
- [Implemented Features & Architecture](file:///C:/pyproj/AWS-CapacityForecaster/progress/0001.Progress_so_Far.md)
- [Current Challenge: AutoGluon Crash](file:///C:/pyproj/AWS-CapacityForecaster/progress/0002.Current_Challenge_AutoGluon_Deps.md)

## 2. Environment & Configuration
- **VS Code**: Configured to use `python.defaultInterpreterPath` in `.vscode/settings.json`.
    - **Policy**: We track this file in git.
    - **Behavior**: Auto-activates `Activate.ps1` on terminal launch.
    - **Strategy**: Keeps local venv outside the project (`c:\py_venv\AWS-CapacityForecaster`) to save OneDrive space, while syncing the path config via git.
- **PowerShell**: Validated that `$PROFILE` is clean; auto-activation is handled purely by VS Code settings.

## 3. Active Technical Challenge (Module 04)
**Issue:** `autogluon.tabular` fails to import due to conflicts with `numpy>=2.0.0` and `pandas>=2.2.0`.
**Solution In-Progress:**
- **Wrapper Script**: `src/modules/module_04_model_training.py` now acts as a dependency guard.
- **Runtime Installation**: The wrapper installs a specific "Goldilocks" dependency set before launching the inner training logic.
- **Pinned Versions**:
    - `numpy<2.0.0`
    - `pandas==2.1.4`
    - `autogluon>=1.1.0`

## 4. Next Steps
1. Run `src/modules/module_04_model_training.py` to test the wrapper.
2. Verify that `module_04_inner.py` runs without `ModuleNotFoundError`.
3. If successful, deploy to SageMaker.
