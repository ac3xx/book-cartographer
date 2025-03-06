# CLAUDE.md - Epub Dict Gen

## Build & Development Commands
- Install dependencies: `poetry install`
- Run tests: `poetry run pytest`
- Run single test: `poetry run pytest tests/path_to_test.py::test_name`
- Format code: `poetry run black .`
- Type check: `poetry run mypy .`
- Lint: `poetry run ruff check .`
- Process EPUB: `poetry run python -m epub_character_graph process path/to/file.epub`
- Additional processing flags: `--all-relationships` `--use-llm-for-nlp`

## Code Style Guidelines
- **Python Version**: >=3.12
- **Formatting**: Use Black with default settings
- **Imports**: Group standard lib, third-party, local imports with a blank line between
- **Types**: Use type annotations for all function parameters and return values
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error Handling**: Use explicit exception types with meaningful error messages
- **Testing**: Write pytest tests for all functionality
- **Docstrings**: Google style for all public functions and classes
- **Async**: Use async/await for I/O-bound operations, especially LLM calls

## Project Documentation
- **Improvements**: See `IMPROVEMENTS.md` for recent feature additions
- **Centrality Fix**: See `CENTRALITY_FIX.md` for details on the centrality calculation improvements
- **Pending Issues**: See `PENDING_ISSUES.md` for known issues and future work

## Tools & Utilities
- **Recalculate Centrality**: `poetry run python recalculate_centrality.py <graph_file_path>`
- **Test Helpers**: Scripts for testing centrality calculations are in project root

## Recent Fixes
- Fixed character interaction count tracking during extraction
- Added co-occurrence detection for more accurate relationship metrics
- Fixed Pydantic deprecation warnings by updating to model_dump()
- See `PENDING_ISSUES.md` for details on completed fixes

## Known Issues
- Entity extractor test is failing due to async function handling
- See `PENDING_ISSUES.md` for remaining issues and details