# Project Constitution

This file serves as the supreme authority for the rules, coding standards, and operational protocols of this project. The Agent must abide by the laws and instructions set forth here.

## 1. Prime Directives
1. **Context Persistence**: Always update `.agent/project_memory.md` at the end of a significant session or task to ensure continuity across machines.
2. **Safety First**: Never execute destructive commands (like `rm -rf` or unsupervised deletes) without explicit user confirmation, even if "turbo" mode is active.

## 2. Coding Standards
(Add specific coding styles, linter rules, or framework preferences here)
- Follow PEP 8 for Python code.
- Ensure all new modules have a corresponding test or verification step.

## 3. Operational Protocols
- **Memory Check**: At the start of a session, always read `.agent/project_memory.md` and this Constitution to understand the current state and laws.
- **Memory Updates**: Periodically update `.agent/project_memory.md` to reflect the latest stage of work, especially after completing significant sub-tasks or changing context.
- **Sync Routine**: Periodically ask the user for permission to execute `git pull` (to refresh code) and `git push` (to save state), ensuring seamless continuity across machines.
