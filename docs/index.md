# EPUB Character Graph Documentation

## Project Overview

The EPUB Character Graph project extracts character information from EPUB files, builds relationship networks, and generates specialized outputs such as image prompts or dictionaries.

## Core Features

- Extract characters and relationships from EPUB files
- Build detailed character profiles with personality, background, appearance, etc.
- Create network graphs of character relationships
- Generate AI image prompts based on character descriptions
- Create dictionaries compatible with e-readers
- Support series tracking across multiple books

## Process Flow

The system follows this general processing pipeline:

1. Parse EPUB file to extract text and metadata
2. Identify character names using NLP (either spaCy or LLM-based)
3. Verify characters and extract character details
4. Identify relationships between characters
5. Build a character graph representing the book's character network
6. Generate desired output formats

## Pipeline Diagrams

- [Core Processing Pipeline](./images/epub_character_graph_pipeline.png) - The main processing flow
- [Series Support Pipeline](./series_pipeline.mmd) - The series processing flow

## Documentation

- [Original Design](./Original%20Design.md) - The original design document
- [Improvements](./IMPROVEMENTS.md) - Recent improvements to the project
- [Centrality Fix](./CENTRALITY_FIX.md) - Details on centrality calculation fixes
- [Interaction Counts Fix](./INTERACTION_COUNTS_FIX.md) - Details on interaction count tracking
- [Pending Issues](./PENDING_ISSUES.md) - Known issues and planned work
- [Series Support](./SERIES_SUPPORT.md) - Implementation plan for series support

## API Reference

The main components of the system are:

- `EPUBParser` - Extracts text and metadata from EPUB files
- `EntityExtractor` - Identifies character names using NLP techniques
- `LLMProcessor` - Uses LLMs to extract character details and relationships
- `CharacterGraph` - Represents the network of characters and relationships
- `SeriesGraph` - Manages character data across multiple books in a series
- Output generators for different formats (image prompts, dictionary, etc.)

## Usage

See the [README.md](../README.md) file in the project root for usage instructions and command-line reference.