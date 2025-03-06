# BookCartographer

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)

A comprehensive command-line tool that maps and analyzes the entire world of EPUB books - extracting characters, locations, organizations, and other entities to build relationship graphs and generate specialized outputs for readers, writers, and researchers.

## Features

- Extract named entities from EPUB content (characters, locations, organizations, objects, and more)
- Build comprehensive entity graphs representing relationships between all story elements
- Generate AI image prompts based on entity descriptions
- Create KOReader-compatible dictionaries for in-reader entity lookup
- Track entity evolution across book series
- Map the complete narrative landscape with centrality and importance metrics
- Hybrid NLP approach combining spaCy and LLMs for nuanced understanding

## Installation

### Prerequisites

- Python 3.9 or higher
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
poetry run book-cartographer process your_book.epub --mode=image-prompts

# Create a configuration file
poetry run book-cartographer init-config
```

Processing an EPUB file will:
1. Extract entity information from the EPUB (characters, locations, etc.)
2. Create a comprehensive entity relationship graph
3. Generate the requested output format (image prompts, dictionary, etc.)
4. Save the output and graph data for further analysis

### Command-line Reference

```
Usage: book-cartographer [OPTIONS] COMMAND [ARGS]...

Commands:
  process      Process EPUB files to extract entity data and generate outputs
  init-config  Initialize a new configuration file with default settings

Options:
  --help      Show this message and exit
```

#### Process Command

```
Usage: book-cartographer process [OPTIONS] EPUB_FILE

  Process EPUB files to extract entity data and generate outputs.

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
Usage: book-cartographer init-config [OPTIONS]

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
poetry run book-cartographer process your_book.epub --mode=image-prompts
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
poetry run book-cartographer process your_book.epub --mode=kodict
```

The output file can be imported into KOReader as a dictionary, allowing readers to look up character information while reading.

## Configuration

You can customize the behavior by creating a configuration file and passing it with the `--config` option.

### Creating a Configuration File

The application supports configuration files in TOML or JSON format. To create a default configuration file:

```bash
poetry run book-cartographer init-config --output config.toml
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

#### Series Settings

```toml
[series]
enable_series_tracking = false  # Enable series tracking features
series_name = ""  # Series name (overrides auto-detection)
character_similarity_threshold = 0.75  # Threshold for matching characters across books (0.0-1.0)
track_character_evolution = true  # Track character evolution across books in the series
cross_book_references = true  # Include cross-references between books
series_metadata_file = ""  # Path to series metadata file
auto_detect_series = true  # Attempt to auto-detect series information from book metadata
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
use_llm_for_nlp = false  # Whether to use LLM for initial NLP entity extraction
all_character_relationships = false  # Extract relationships between all characters, not just major ones
track_metrics = true  # Track additional metrics like mention counts and centrality scores
store_raw_metrics = true  # Store raw metrics in output JSON
track_character_evolution = true  # Track character evolution throughout the narrative
comprehensive_analysis = true  # Perform comprehensive character analysis
store_intermediate_outputs = false  # Store intermediate processing outputs

# Excerpt limits for different character types
[processing.excerpt_limits]
major_character = 20  # Excerpt limit for major characters
supporting_character = 10  # Excerpt limit for supporting characters
minor_character = 5  # Excerpt limit for minor characters
relationship = 15  # Excerpt limit for character relationships
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
2. **Entity Extraction**: Uses spaCy to identify all types of entities (characters, locations, items, etc.)
3. **LLM Processing**: Uses LLMs to verify entities and extract detailed information and relationships
4. **Graph Building**: Creates a comprehensive network graph of all narrative elements
5. **Analysis**: Performs centrality and importance analysis for all entities
6. **Series Tracking**: Identifies and tracks entities across book series when applicable
7. **Output Generation**: Produces formatted output based on the selected mode

## Technical Details

- Uses a hybrid approach combining rule-based NLP and LLMs
- Implements efficient chunking for processing large texts
- Provides caching to minimize redundant LLM calls
- Supports multiple LLM providers with fallbacks
- Built with modern Python practices including typing and async processing

## License

MIT