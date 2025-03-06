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
git clone https://github.com/ac3xx/book-cartographer.git
cd book-cartographer
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

# Process with advanced options
poetry run book-cartographer process your_book.epub --use-llm-for-nlp --all-relationships --track-metrics

# Create a configuration file
poetry run book-cartographer init-config --output my_config.toml

# Process with a custom configuration file
poetry run book-cartographer process your_book.epub --config my_config.toml

# Process a book in a series
poetry run book-cartographer process book2.epub --series-name "My Series" --book-number 2 --add-to-series

# Initialize a new series
poetry run book-cartographer series-init "My Series" --author "Author Name"

# Analyze a series
poetry run book-cartographer series-analyze my_series_series.json --output series_analysis.txt
```

Processing an EPUB file will:
1. Extract entity information from the EPUB (characters, locations, etc.)
2. Create a comprehensive entity relationship graph
3. Generate the requested output format (image prompts, dictionary, etc.)
4. Save the output and graph data for further analysis

### Command-line Reference

```
Usage: book-cartographer [OPTIONS] COMMAND [ARGS]...

  BookCartographer: Map and analyze the worlds within EPUB files.

  Extract entities, relationships, and narrative elements from EPUB books.
  Generate specialized outputs for readers, writers, and researchers.

Options:
  --help  Show this message and exit.

Commands:
  init-config     Initialize a new configuration file with default settings.
  process         Process EPUB files to extract entity data and generate...
  series-analyze  Analyze character evolution across a book series.
  series-init     Initialize a new series tracking file.
```

#### Process Command

```
Usage: book-cartographer process [OPTIONS] EPUB_FILE

  Process EPUB files to extract entity data and generate outputs.

  EPUB_FILE is the path to the EPUB file to process.

  This command extracts characters, locations, organizations, and other
  entities from an EPUB file, builds relationship graphs, and generates
  outputs in the specified format.

Options:
  -c, --config PATH               Path to configuration file
  -o, --output PATH               Path to output file
  -m, --mode [image-prompts|kodict]
                                  Output mode
  --llm-provider TEXT             LLM provider to use
  --llm-model TEXT                LLM model to use
  --api-key TEXT                  API key for LLM provider
  -v, --verbose                   Enable verbose logging
  --all-relationships             Extract relationships between all
                                  characters, not just major ones
  --use-llm-for-nlp               Use LLM for initial NLP entity extraction
  --track-metrics / --no-track-metrics
                                  Track additional metrics like mention counts
                                  and centrality scores
  --store-raw-metrics / --no-store-raw-metrics
                                  Store raw metrics in output JSON
  --track-evolution / --no-track-evolution
                                  Track character evolution throughout the
                                  narrative
  --comprehensive-analysis / --no-comprehensive-analysis
                                  Perform comprehensive character analysis
                                  including dialogue, motivations, and
                                  emotions
  --store-intermediate / --no-store-intermediate
                                  Store intermediate processing outputs like
                                  character excerpts and LLM responses
  --major-excerpts INTEGER        Number of excerpts to include for major
                                  characters
  --supporting-excerpts INTEGER   Number of excerpts to include for supporting
                                  characters
  --minor-excerpts INTEGER        Number of excerpts to include for minor
                                  characters
  --relationship-excerpts INTEGER Number of excerpts to include for
                                  relationships
  --series-name TEXT              Name of the series this book belongs to
  --book-number INTEGER           Book number in the series (e.g., 1 for first
                                  book)
  --add-to-series                 Add this book to an existing series tracking
                                  file
  --help                          Show this message and exit.
```

#### Init-Config Command

```
Usage: book-cartographer init-config [OPTIONS]

  Initialize a new configuration file with default settings.

  This command creates a new configuration file with all default settings. It
  can be used as a starting point for customizing BookCartographer's behavior.

Options:
  -o, --output PATH  Path to output configuration file
  --overwrite        Overwrite existing config file if it exists
  --help             Show this message and exit.
```

#### Series-Init Command

```
Usage: book-cartographer series-init [OPTIONS] SERIES_NAME

  Initialize a new series tracking file.

  SERIES_NAME is the name of the book series to initialize.

Options:
  -a, --author TEXT  Author of the series
  -o, --output PATH  Path to output series file
  --help             Show this message and exit.
```

#### Series-Analyze Command

```
Usage: book-cartographer series-analyze [OPTIONS] SERIES_FILE

  Analyze character evolution across a book series.

  SERIES_FILE is the path to the series tracking file.

Options:
  -o, --output PATH  Path to output analysis file
  --help             Show this message and exit.
```

### Output Modes

#### Image Prompts

Generates character descriptions formatted for AI image generation:

```bash
poetry run book-cartographer process your_book.epub --mode=image-prompts --output character_images.md
```

Output example:
```markdown
# AI Image Generation Prompts for "Book Title"

## Character Name

A tall, elegant woman in her mid-thirties with fiery red hair and piercing green eyes. She wears a tailored emerald dress with gold embroidery and carries herself with regal bearing.

**Additional details:**
- **Physical features:** Pale skin, high cheekbones, slender build
- **Age:** Mid-thirties
- **Clothing:** Formal Victorian-era gowns, often in green or black
- **Personality:** Determined, intelligent, secretive

## Location: Ancient Temple

A crumbling stone temple deep in the jungle, with moss-covered walls and intricate carvings of forgotten deities. Shafts of golden sunlight pierce through holes in the ceiling, illuminating dust particles dancing in the air.

**Additional details:**
- **Architectural style:** Ancient Mesoamerican
- **Condition:** Partially ruined, overgrown
- **Notable features:** Central altar, hieroglyphic wall panels, hidden chambers
- **Atmosphere:** Mysterious, foreboding, sacred
```

#### KOReader Dictionary

Generates a character encyclopedia compatible with KOReader:

```bash
poetry run book-cartographer process your_book.epub --mode=kodict --output book_dictionary.dict
```

The output file can be imported into KOReader as a dictionary, allowing readers to look up character information while reading.

Example dictionary entry:
```
Elizabeth Bennet

Also known as: Lizzy, Eliza

Role: Protagonist

#### Appearance
Age: 20
Physical features: Intelligent dark eyes, light and pleasing figure, handsome face
Clothing: Simple, elegant Regency-era dresses

#### Personality
Intelligent, witty, playful, independent, strong-willed, honest, judgmental

#### Background
Second daughter of the Bennet family from Longbourn estate in Hertfordshire. 
Received limited formal education but is well-read and accomplished in music.

#### Relationships
- Mr. Darcy: Initially mutual dislike due to pride and prejudice, evolves into deep respect and love
- Jane Bennet: Close confidante and beloved elder sister
- Charlotte Lucas: Longtime friend who chooses security over love
- Mr. Wickham: Initially charmed by him before discovering his true character

#### Character Arc
Begins as a quick to judge, prideful young woman who learns to question her first impressions and overcome her prejudices.
```

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
provider = "gemini"  # LLM provider (gemini, anthropic, openai, etc.)
model = "gemini-2.0-flash"  # Model to use
api_key = ""  # API key (if not set, will look for environment variable based on provider)
max_tokens = 4096  # Maximum tokens for completions
temperature = 0.7  # Temperature for completions (0.0-1.0)
fallback_providers = ["anthropic", "openai"]  # Providers to try if primary fails
fallback_models = { "anthropic" = "claude-3-5-haiku-latest", "openai" = "o3-mini" }  # Models for fallback providers
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
cache_dir = "~/.cache/book-cartographer"  # Cache directory for LLM responses
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
   - `~/.config/book-cartographer/config.toml` or `~/.config/book-cartographer/config.json`
4. Default values built into the application (lowest priority)

You can generate a configuration file with all available options using:

```bash
# Create default config
poetry run book-cartographer init-config

# Create config in specific location
poetry run book-cartographer init-config --output ~/.config/book-cartographer/config.toml
```

## How It Works

1. **EPUB Parsing**: Extracts text content from EPUB files
2. **Entity Extraction**: Uses spaCy or LLMs to identify all types of entities (characters, locations, items, etc.)
3. **Excerpt Collection**: Gathers relevant text passages for each identified entity
4. **Entity Verification**: Uses LLMs to verify entities and filter false positives
5. **Relationship Analysis**: Identifies connections and relationships between entities
6. **Graph Building**: Creates a comprehensive network graph of all narrative elements
7. **Metric Calculation**: Performs centrality and importance analysis for all entities
8. **Series Tracking**: Identifies and tracks entities across book series when applicable
9. **Output Generation**: Produces formatted output based on the selected mode

## Technical Details

- Uses a hybrid approach combining rule-based NLP and LLMs for robust entity extraction
- Implements efficient chunking for processing large texts (works with books of any length)
- Provides caching to minimize redundant LLM calls and reduce API costs
- Supports multiple LLM providers (Gemini, Claude, OpenAI) with automatic fallbacks
- Handles relationship extraction with detailed contextual understanding
- Uses graph theory for calculating entity importance and narrative centrality
- Implements asynchronous processing for improved performance
- Supports series analysis across multiple books
- Built with modern Python practices including strong typing and async/await patterns
- Modular architecture with pluggable output generators for extensibility

## License

MIT