#!/usr/bin/env python
"""Script to update interaction counts for testing centrality calculations."""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_interaction_counts(graph_file: Path, output_file: Path) -> None:
    """Update interaction counts in an existing character graph for testing.
    
    Args:
        graph_file: Path to the JSON file containing the character graph
        output_file: Path to save the updated graph
    """
    logger.info(f"Loading character graph from {graph_file}")
    with open(graph_file, 'r') as f:
        data = json.load(f)
    
    logger.info("Updating interaction counts")
    
    # Update character mention counts to reflect importance
    character_mentions = {
        "Carl": 250,        # Increased from 120
        "Donut": 200,       # Increased from 100
        "Bea": 80,          # Increased from 45
        "Mongo": 50,        # Increased from 25
        "Mordecai": 40,     # Increased from 20
        "Samantha": 35,     # Increased from 18
        "Zorathis": 30,     # Increased from 15
        "The Maestro": 25,  # Increased from 12
        "Lorelai": 20,      # Increased from 10
        "Agatha": 15        # Increased from 8
    }
    
    # Update relationship interaction counts
    relationship_interactions = [
        {"pair": ["Carl", "Donut"], "interactions": 180, "mentions": 150},  # Increased
        {"pair": ["Carl", "Bea"], "interactions": 60, "mentions": 45},      # Increased
        {"pair": ["Donut", "Bea"], "interactions": 45, "mentions": 35},     # Increased
        {"pair": ["Carl", "Mongo"], "interactions": 40, "mentions": 30},    # Increased
        {"pair": ["Carl", "Mordecai"], "interactions": 35, "mentions": 25}, # Increased
        {"pair": ["Donut", "Mongo"], "interactions": 30, "mentions": 20},   # Increased
        {"pair": ["Carl", "Samantha"], "interactions": 25, "mentions": 20}, # Increased
        {"pair": ["Donut", "Samantha"], "interactions": 20, "mentions": 15}, # New relationship
        {"pair": ["Donut", "Mordecai"], "interactions": 15, "mentions": 10}, # New relationship
    ]
    
    # Update character mention counts
    for char in data["characters"]:
        if char["name"] in character_mentions:
            if "metrics" not in char:
                char["metrics"] = {}
            char["metrics"]["mention_count"] = character_mentions[char["name"]]
    
    # Create a map of existing relationships
    existing_relationships = {}
    for rel in data["relationships"]:
        pair = frozenset([rel["character1"], rel["character2"]])
        existing_relationships[pair] = rel
    
    # Update existing relationships and add new ones
    for rel_info in relationship_interactions:
        pair = frozenset(rel_info["pair"])
        
        if pair in existing_relationships:
            # Update existing relationship
            rel = existing_relationships[pair]
            if "metrics" not in rel:
                rel["metrics"] = {}
            rel["metrics"]["interaction_count"] = rel_info["interactions"]
            rel["metrics"]["mention_together_count"] = rel_info["mentions"]
        else:
            # Create new relationship if both characters exist
            char1, char2 = rel_info["pair"]
            
            # Check if both characters exist
            if any(c["name"] == char1 for c in data["characters"]) and \
               any(c["name"] == char2 for c in data["characters"]):
                # Add new relationship
                new_rel = {
                    "character1": char1,
                    "character2": char2,
                    "relationship_type": "relationship",
                    "description": f"Relationship between {char1} and {char2}",
                    "strength": 5,
                    "metrics": {
                        "interaction_count": rel_info["interactions"],
                        "mention_together_count": rel_info["mentions"],
                        "dialogue_count": 0,
                        "scene_count": 0
                    },
                    "dynamics": "",
                    "conflicts": [],
                    "balance": "",
                    "emotional_tone": "",
                    "pivotal_moments": [],
                    "evolution": {
                        "arc_points": [],
                        "early_state": "",
                        "middle_development": "",
                        "final_state": ""
                    }
                }
                data["relationships"].append(new_rel)
                logger.info(f"Added new relationship between {char1} and {char2}")
    
    logger.info(f"Number of relationships: {len(data['relationships'])}")
    
    # Save updated graph
    logger.info(f"Saving updated character graph to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update interaction counts for testing centrality calculations")
    parser.add_argument("graph_file", type=Path, help="Path to the JSON file containing the character graph")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to save the updated graph")
    
    args = parser.parse_args()
    update_interaction_counts(args.graph_file, args.output)