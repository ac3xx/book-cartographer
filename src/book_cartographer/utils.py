"""Utility functions for the EPUB character graph generator."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_character_excerpts(
    text: str, characters: Set[str], context_size: int = 150, track_metrics: bool = True
) -> Dict[str, List[str]]:
    """Extract text excerpts containing character mentions and track metrics.

    Args:
        text: Full text content
        characters: Set of character names to extract excerpts for
        context_size: Number of characters before and after the match to include
        track_metrics: Whether to track additional metrics like mention counts

    Returns:
        Dictionary mapping character names to lists of excerpts
    """
    # Sort character names by length (longest first) to avoid partial matches
    sorted_characters = sorted(characters, key=len, reverse=True)
    
    # Dictionary to store excerpts for each character
    excerpts: Dict[str, List[str]] = {char: [] for char in characters}
    
    # Track metrics if requested
    metrics = {}
    if track_metrics:
        metrics = {
            "mention_counts": {char: 0 for char in characters},
            "interactions": {char: {other: 0 for other in characters if other != char} for char in characters},
            "dialogue_counts": {char: 0 for char in characters}
        }
    
    # Dictionary to store character pairs that appear in the same excerpt
    character_interactions = {}
    
    for character in sorted_characters:
        # Create a pattern that matches whole words only
        pattern = r'\b' + re.escape(character) + r'\b'
        
        # Find all matches in the text
        for match in re.finditer(pattern, text):
            # Track mention count
            if track_metrics:
                metrics["mention_counts"][character] += 1
            
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            
            # Extract excerpt with context
            excerpt = text[start:end]
            
            # Clean up excerpt (remove extra whitespace, normalize newlines)
            excerpt = re.sub(r'\s+', ' ', excerpt).strip()
            
            # Skip if the excerpt is empty or duplicated
            if not excerpt or excerpt in excerpts[character]:
                continue
                
            # Add to excerpts for this character
            excerpts[character].append(excerpt)
            
            # Check for dialogue
            if track_metrics and re.search(r'["\']\s*\w+', excerpt):
                # Very simplistic dialogue detection - look for quotes followed by text
                metrics["dialogue_counts"][character] += 1
            
            # Check for interactions with other characters
            if track_metrics:
                for other in sorted_characters:
                    if other != character and re.search(r'\b' + re.escape(other) + r'\b', excerpt):
                        # These characters appear together
                        pair = tuple(sorted([character, other]))
                        if pair not in character_interactions:
                            character_interactions[pair] = 0
                        character_interactions[pair] += 1
                        
                        # Update interaction count for both characters
                        metrics["interactions"][character][other] += 1
                        metrics["interactions"][other][character] += 1
    
    # Log statistics
    total_excerpts = sum(len(e) for e in excerpts.values())
    logger.info(f"Extracted {total_excerpts} excerpts for {len(characters)} characters")
    
    if track_metrics:
        # Log metrics
        total_mentions = sum(metrics["mention_counts"].values())
        total_interactions = sum(character_interactions.values())
        logger.info(f"Tracked {total_mentions} total character mentions")
        logger.info(f"Detected {total_interactions} character interactions")
        
        # Attach metrics to the excerpts dictionary
        excerpts["_metrics"] = metrics
    
    return excerpts


def validate_epub_path(file_path: str) -> Path:
    """Validate and convert an EPUB file path.

    Args:
        file_path: Path to the EPUB file

    Returns:
        Validated Path object

    Raises:
        ValueError: If the file doesn't exist or isn't an EPUB
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    if path.suffix.lower() != ".epub":
        raise ValueError(f"Not an EPUB file: {file_path}")
    
    return path


def sanitize_filename(text: str) -> str:
    """Sanitize text for use as a filename.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text suitable for a filename
    """
    # Replace spaces with underscores
    text = text.replace(" ", "_")
    
    # Remove characters that are problematic in filenames
    text = re.sub(r'[^\w\-\.]', '', text)
    
    # Ensure the filename isn't too long
    if len(text) > 100:
        text = text[:100]
    
    return text