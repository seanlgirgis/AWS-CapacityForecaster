# Project Environment & Context Rules

## 1. Environment Setup
- **Virtual Environment Path**: `c:\py_venv\AWS-CapacityForecaster`
- **Activation Script**: `C:\pyproj\AWS-CapacityForecaster\env_setter.ps1`
- **Python Interpreter**: Python 3.12.3

## 2. Shell Configuration
- Always use the provided `env_setter.ps1` to configure the shell session.
- **Command**: `. .\env_setter.ps1`
- **Effect**:
  - Activates the virtual environment.
  - Sets `PROJECT_ROOT`.
  - Updates `PYTHONPATH` to include the project root.
  - Sets `KB_INBOX_PATH` (External Documentation).

## 3. Git & Data Persistence
- The `.agent` directory contains persistent context and rules for AI assistants.
- **Git Policy**: The `.agent` directory MUST be tracked by Git. Do not ignore it.

## 4. Path Conventions
- Use absolute paths when possible or relative to `PROJECT_ROOT`.

## 5. Documentation & External Resources
- **External Documentation Path**: `C:\KB\00_Inbox` (Mapped to `$env:KB_INBOX_PATH`)
- Use this path when the user requests "External Documentation" or "Inbox" writes.
