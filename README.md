# EPUB Entity Extraction and Character Graph Generator

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)

A command-line application that processes EPUB files to extract named entities, build a character graph, and generate specialized outputs for various use cases.

## Features

- Extract named entities from EPUB content (characters, locations, organizations)
- Build character graphs representing relationships between characters
- Generate AI image prompts based on character descriptions
- Create KOReader-compatible dictionaries for in-reader character lookup
- Hybrid NLP approach using spaCy and LLMs

## Installation

### Prerequisites

- Python 3.12 or higher
- Poetry dependency manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/epub-dict-gen.git
cd epub-dict-gen
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Download spaCy model (if you don't already have it):
```bash
poetry run python -m spacy download en_core_web_lg
```

4. Set up API keys (required for LLM processing):
   * For OpenAI: `export OPENAI_API_KEY=your_api_key_here`
   * For Anthropic: `export ANTHROPIC_API_KEY=your_api_key_here`

## Usage

### Basic Usage

```bash
# Process an EPUB file
poetry run python -m epub_character_graph process your_book.epub --mode=image-prompts

# Create a configuration file
poetry run python -m epub_character_graph init-config
```

Processing an EPUB file will:
1. Extract character information from the EPUB
2. Create a character graph
3. Generate AI image prompts for the characters
4. Save the output to `your_book_characters.txt`

### Command-line Reference

```
Usage: python -m epub_character_graph [OPTIONS] COMMAND [ARGS]...

Commands:
  process      Process EPUB files to extract character data and generate outputs
  init-config  Initialize a new configuration file with default settings

Options:
  --help      Show this message and exit
```

#### Process Command

```
Usage: python -m epub_character_graph process [OPTIONS] EPUB_FILE

  Process EPUB files to extract character data and generate outputs.

  EPUB_FILE is the path to the EPUB file to process.

Options:
  -c, --config PATH             Path to configuration file
  -o, --output PATH             Path to output file
  -m, --mode [image-prompts|kodict]
                                Output mode
  --llm-provider TEXT           LLM provider to use
  --llm-model TEXT              LLM model to use
  --api-key TEXT                API key for LLM provider
  -v, --verbose                 Enable verbose logging
  --help                        Show this message and exit
```

#### Init-Config Command

```
Usage: python -m epub_character_graph init-config [OPTIONS]

  Initialize a new configuration file with default settings.

  This command creates a new configuration file with all default settings.
  It can be used as a starting point for customizing the configuration.

Options:
  -o, --output PATH  Path to output configuration file  [default: ./config.toml]
  --overwrite        Overwrite existing config file if it exists
  --help             Show this message and exit
```

### Output Modes

#### Image Prompts

Generates character descriptions formatted for AI image generation:

```bash
poetry run python -m epub_character_graph your_book.epub --mode=image-prompts
```

Output example:
```
# AI Image Generation Prompts for "Book Title"

## Character Name

A tall, elegant woman in her mid-thirties with fiery red hair and piercing green eyes. She wears a tailored emerald dress with gold embroidery and carries herself with regal bearing.

**Additional details:**
- **Physical features:** Pale skin, high cheekbones, slender build
- **Age:** Mid-thirties
- **Clothing:** Formal Victorian-era gowns, often in green or black
- **Personality:** Determined, intelligent, secretive
```

#### KOReader Dictionary

Generates a character encyclopedia compatible with KOReader:

```bash
poetry run python -m epub_character_graph your_book.epub --mode=kodict
```

The output file can be imported into KOReader as a dictionary, allowing readers to look up character information while reading.

## Configuration

You can customize the behavior by creating a configuration file and passing it with the `--config` option.

### Creating a Configuration File

The application supports configuration files in TOML or JSON format. To create a default configuration file:

```bash
poetry run python -m epub_character_graph init-config --output config.toml
```

This will create a configuration file with default settings. You can then edit this file to customize the behavior.

### Configuration Options

The configuration file is divided into sections:

#### General Settings

```toml
log_level = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
save_interim_data = false  # Whether to save interim processing data
interim_data_dir = "./interim_data"  # Directory for interim data
```

#### LLM Settings

```toml
[llm]
provider = "openai"  # LLM provider (openai, anthropic, etc.)
model = "gpt-4"  # Model to use
api_key = ""  # API key (if not set, will look for environment variable)
max_tokens = 4096  # Maximum tokens for completions
temperature = 0.7  # Temperature for completions (0.0-1.0)
fallback_providers = ["anthropic", "google"]  # Providers to try if primary fails
fallback_models = { "anthropic" = "claude-2", "google" = "gemini-pro" }  # Models for fallback providers
retry_attempts = 3  # Number of retry attempts
timeout = 60  # Timeout in seconds for LLM requests
```

#### Processing Settings

```toml
[processing]
spacy_model = "en_core_web_lg"  # spaCy model to use
batch_size = 10  # Batch size for processing
max_chunk_size = 1000  # Maximum chunk size for text processing
min_entity_occurrences = 3  # Minimum occurrences for entity to be considered
cache_dir = "~/.cache/epub-character-graph"  # Cache directory for LLM responses
use_cache = true  # Whether to use caching
parallel_requests = 5  # Number of parallel LLM requests
extract_relationships = true  # Whether to extract character relationships
extract_groups = true  # Whether to extract character groups/factions
```

#### Output Settings

```toml
[output]
mode = "image-prompts"  # Output mode (image-prompts or kodict)
output_file = "characters.txt"  # Output file path (optional)
template_dir = "./templates"  # Directory containing Jinja2 templates
include_minor_characters = false  # Whether to include minor characters in output
image_size = "1024x1024"  # Size for image prompts
image_style = "realistic"  # Style for image prompts
```

### Configuration Precedence

The application uses the following order of precedence for configuration:

1. Command-line arguments (highest priority)
2. Configuration file specified with `--config`
3. Default configuration files in standard locations:
   - `./config.toml` or `./config.json` (current directory)
   - `~/.config/epub-character-graph/config.toml` or `~/.config/epub-character-graph/config.json`
4. Default values built into the application (lowest priority)

## How It Works

1. **EPUB Parsing**: Extracts text content from EPUB files
2. **Entity Extraction**: Uses spaCy to identify potential character names and other entities
3. **LLM Processing**: Uses LLMs to verify characters and extract detailed information
4. **Graph Building**: Creates a network graph of characters and their relationships
5. **Output Generation**: Produces formatted output based on the selected mode

## Technical Details

- Uses a hybrid approach combining rule-based NLP and LLMs
- Implements efficient chunking for processing large texts
- Provides caching to minimize redundant LLM calls
- Supports multiple LLM providers with fallbacks
- Built with modern Python practices including typing and async processing

## License

MIT