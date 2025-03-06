# Contributing to BookCartographer

Thank you for considering contributing to BookCartographer! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug in the project:

1. Check if the bug has already been reported in the [GitHub Issues](https://github.com/ac3xx/book-cartographer/issues).
2. If not, create a new issue with a clear description of the bug, including:
   - A clear and descriptive title
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Any relevant logs or screenshots

### Suggesting Enhancements

If you have an idea for an enhancement:

1. Check if a similar enhancement has already been suggested in the [GitHub Issues](https://github.com/ac3xx/book-cartographer/issues).
2. If not, create a new issue with a clear description of your enhancement idea, including:
   - A clear and descriptive title
   - A detailed description of the proposed enhancement
   - Any potential implementation approaches (if applicable)

### Pull Requests

We welcome pull requests! Here's how to submit one:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Run tests to ensure your changes don't break existing functionality.
5. Submit a pull request with a clear description of the changes.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Poetry dependency manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ac3xx/book-cartographer.git
cd book-cartographer
```

2. Install dependencies:
```bash
poetry install
```

3. Install the pre-commit hooks:
```bash
pre-commit install
```

### Development Workflow

1. Make sure your code follows the project's style guidelines:
   - Use Black for code formatting
   - Use isort for import sorting
   - Follow type hints and docstring conventions

2. Run tests before submitting a pull request:
```bash
poetry run pytest
```

3. Run linters to ensure code quality:
```bash
poetry run ruff check .
poetry run mypy .
```

## Project Structure

- `src/book_cartographer/` - Main package
- `tests/` - Unit tests
- `docs/` - Documentation
- `scripts/` - Utility scripts
- `config/` - Configuration files

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).