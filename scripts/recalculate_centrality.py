#!/usr/bin/env python
"""Script to recalculate centrality scores for an existing character graph."""

import json
import logging
from pathlib import Path

from epub_character_graph.character_graph import CharacterGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recalculate_centrality(graph_file: Path, output_file: Path = None) -> None:
    """Recalculate centrality scores for an existing character graph.
    
    Args:
        graph_file: Path to the JSON file containing the character graph
        output_file: Path to save the updated graph (defaults to overwriting original)
    """
    if output_file is None:
        output_file = graph_file
    
    logger.info(f"Loading character graph from {graph_file}")
    graph = CharacterGraph.load_from_file(graph_file)
    
    logger.info("Calculating centrality metrics")
    centrality = graph.calculate_centrality_metrics(update_nodes=True)
    
    # Get top 10 characters by centrality and importance
    characters = graph.get_all_characters()
    
    print("\nCentrality Scores (Top 10):")
    for char in sorted(characters, key=lambda x: x['metrics'].get('centrality_score', 0), reverse=True)[:10]:
        print(f"{char['name']}: {char['metrics'].get('centrality_score', 0):.6f}")
    
    print("\nImportance Scores (Top 10):")
    for char in sorted(characters, key=lambda x: x['metrics'].get('importance_score', 0), reverse=True)[:10]:
        print(f"{char['name']}: {char['metrics'].get('importance_score', 0):.6f}")
    
    logger.info(f"Saving updated character graph to {output_file}")
    graph.save_to_file(output_file)
    
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recalculate centrality scores for a character graph")
    parser.add_argument("graph_file", type=Path, help="Path to the JSON file containing the character graph")
    parser.add_argument("--output", "-o", type=Path, help="Path to save the updated graph (defaults to overwriting original)")
    
    args = parser.parse_args()
    recalculate_centrality(args.graph_file, args.output)