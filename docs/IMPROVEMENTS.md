# EPUB Character Graph Generator Improvements

## Overview of Enhancements

We have significantly improved the EPUB character graph generator to provide more comprehensive and detailed character information. These improvements focus on extracting more meaningful character data, tracking metrics, and enhancing relationship analysis.

## Key Improvements

### 1. Increased Excerpt Limits

- Added configurable excerpt limits based on character type:
  - Major characters: 20 excerpts (default)
  - Supporting characters: 10 excerpts (default)
  - Minor characters: 5 excerpts (default)
  - Relationships: 15 excerpts (default)
- Can be configured in `config.toml` or via command line flags

### 2. Tracking of Character Metrics

- Added mention count tracking for all characters
- Added dialogue count tracking to identify speaking patterns
- Added interaction count tracking between characters
- Store these metrics in the output JSON for analysis

### 3. Enhanced Character Classification

- Added three-tier character type system: major, supporting, minor
- Identify character archetypes (hero, mentor, ally, trickster, etc.)
- Calculate importance scores based on centrality metrics
- Store raw centrality scores in output for analysis

### 4. Comprehensive Character Evolution Tracking

- Track character evolution throughout the narrative:
  - Early presentation
  - Middle development
  - Final state
- Track changes in physical appearance over time
- Record key moments that define character development

### 5. Detailed Relationship Analysis

- Enhanced relationship tracking:
  - Relationship dynamics
  - Power balance
  - Emotional tone
  - Evolution over the course of the story
  - Key pivotal moments
  - Conflicts within relationships

### 6. Emotional and Psychological Dimensions

- Added tracking of character emotions at different points in the story
- Record character motivations and goals
- Identify internal conflicts and struggles
- Analyze dialogue style and patterns

### 7. Environmental Context

- Track locations and settings associated with characters
- Establish character's relationship to their environment

## Command Line Options

New command-line options have been added to control these features:

```bash
# Enable/disable metrics tracking
--track-metrics / --no-track-metrics

# Store raw metrics in JSON output
--store-raw-metrics / --no-store-raw-metrics

# Track character evolution
--track-evolution / --no-track-evolution

# Enable comprehensive analysis
--comprehensive-analysis / --no-comprehensive-analysis

# Store intermediate outputs to JSON files
--store-intermediate / --no-store-intermediate

# Set excerpt limits
--major-excerpts NUM
--supporting-excerpts NUM
--minor-excerpts NUM
--relationship-excerpts NUM
```

## Output Format

The JSON output now includes additional fields for each character:

- `character_type`: "major", "supporting", or "minor"
- `archetype`: Character archetype classification
- `physical_appearance_evolution`: Changes in appearance over time
- `dialogue_style`: Character's speaking patterns
- `motivations`: Character's goals and drives
- `emotional_state`: Emotions at different points in the story
- `internal_conflicts`: Character's inner struggles
- `locations`: Settings associated with the character
- `metrics`: Numerical metrics about the character
- `evolution`: Information about character's narrative evolution

Relationships now include:

- `dynamics`: Description of relationship dynamics
- `conflicts`: Points of conflict in the relationship
- `balance`: Power balance description
- `emotional_tone`: Emotional quality of the relationship
- `pivotal_moments`: Key moments defining the relationship
- `evolution`: How the relationship changes over time
- `metrics`: Numerical metrics about the relationship

## Example Usage

```bash
# Process with comprehensive analysis and increased excerpt limits
poetry run python -m epub_character_graph process my_book.epub --track-metrics --store-raw-metrics --track-evolution --comprehensive-analysis --major-excerpts 30 --supporting-excerpts 15 --minor-excerpts 7

# Process with LLM-based extraction and relationship analysis
poetry run python -m epub_character_graph process my_book.epub --use-llm-for-nlp --all-relationships --track-metrics
```
