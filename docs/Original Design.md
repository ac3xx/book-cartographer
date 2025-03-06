# EPUB Entity Extraction and Character Graph Generator

Create a command-line application that processes EPUB files to extract named entities, build a character graph, and generate specialized outputs based on configurable use cases.

## Core Requirements

### 1. Input Processing

- Create a CLI that accepts one or more EPUB files as input
- Implement robust EPUB parsing to extract readable text content
- **Suggested libraries**: `ebooklib` for EPUB parsing

### 2. Hybrid NLP Processing

- Implement a two-stage NLP pipeline:
  - Stage 1: Use `spaCy` for initial entity extraction and text processing
  - Stage 2: Use LLMs with `litellm` for enhanced contextual understanding
- Extract named entities from the EPUB content:
  - Character names (spaCy)
  - Locations (spaCy)
  - Organizations (spaCy)
  - Temporal references (spaCy)
- For each identified character, use LLMs to extract and categorize:
  - Visual descriptions (appearance, clothing)
  - Personality traits
  - Character relationships
  - Significant actions or plot events
- **Suggested libraries**:
  - `spaCy` for initial NLP and entity recognition
  - `litellm` for interfacing with multiple LLM providers (OpenAI, Anthropic, etc.)
  - `asyncio` for parallel LLM processing

### 3. Graph Structure

- Implement a graph data structure where:
  - Nodes = characters with their attributes
  - Edges = relationships between characters
  - Edge weights = relationship importance/frequency
- Store character metadata in each node
- Track relationship types and strength in edges
- **Suggested libraries**: `networkx` for graph implementation and analysis

### 4. Output Modes

Implement two specialized output modes:

1. **AI Image Generation Prompts**:

   - Select characters from the graph
   - Compile visual descriptors into structured prompts
   - Format output for AI image generation systems

1. **KOReader Dictionary Format**:

   - Generate character encyclopedia entries
   - Format content to be compatible with KOReader dictionary import

- **Suggested libraries**: `jinja2` for templating output formats

### 5. CLI Interface

- Accept arguments for:
  - Input file path(s)
  - Output mode selection
  - Output file path
  - LLM provider selection
  - Optional processing flags
- Provide meaningful progress feedback and error handling
- **Suggested libraries**: `click` for CLI interface, `rich` for terminal formatting and progress display

## Technical Specifications

- Structure the project using modern Python practices
- Implement proper logging with the `logging` module
- Use `pydantic` for data validation and settings management
- Use `tqdm` for progress bars
- Implement LLM provider fallback with `litellm`
- Cache LLM responses for efficiency with `diskcache`
- Add configuration for API keys and model preferences

## LLM Integration Strategy

- Use spaCy for initial entity extraction (efficient, deterministic)
- Use LLMs for:
  - Resolving ambiguous character references
  - Extracting deeper character traits and relationships
  - Generating final structured outputs
- Configure with sensible chunking strategy to handle context window limitations
- Implement batching and rate limiting for API calls
- Include local embeddings for semantic similarity backup when needed

## Example Usage

```
python -m epub_character_graph input.epub --mode=image-prompts --output=prompts.txt --llm-provider=openai
python -m epub_character_graph input.epub --mode=kodict --output=characters.dict --llm-provider=anthropic
```

## Project Structure

```
epub_character_graph/
├── __init__.py
├── __main__.py
├── cli.py               # Command-line interface using Click
├── config.py            # Configuration handling (API keys, etc.)
├── epub_parser.py       # EPUB file handling
├── nlp/
│   ├── __init__.py
│   ├── entity_extractor.py    # spaCy-based entity extraction
│   ├── llm_processor.py       # LLM integration via litellm
│   └── prompts.py             # LLM prompt templates
├── character_graph.py   # Graph data structure
├── output_generators/
│   ├── __init__.py
│   ├── base.py          # Base generator class
│   ├── image_prompt.py  # AI image prompt generator
│   └── kodict.py        # KOReader dictionary generator
└── utils.py             # Helper functions
```

## Deliverables

1. Source code with documentation
1. Installation instructions including API key setup
1. Usage examples with sample outputs
1. Basic test cases
