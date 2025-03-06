"""Character graph implementation for representing character relationships."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SeriesMetadata(BaseModel):
    """Metadata for a book series."""
    
    series_name: str = ""
    book_number: int = 0
    total_books: int = 0
    book_title: str = ""
    series_genre: str = ""
    reading_order: int = 0  # May differ from book_number for prequels, etc.
    
    def is_in_series(self) -> bool:
        """Check if this book is part of a series."""
        return bool(self.series_name)


class CharacterMetrics(BaseModel):
    """Metrics and statistics for a character."""
    
    mention_count: int = 0
    dialogue_count: int = 0
    scene_count: int = 0
    interaction_counts: Dict[str, int] = {}
    centrality_score: float = 0.0
    importance_score: float = 0.0
    

class CharacterEvolution(BaseModel):
    """Tracking of a character's evolution throughout the narrative."""
    
    arc_points: List[Dict[str, str]] = []
    key_moments: List[Dict[str, str]] = []
    early_presentation: str = ""
    middle_development: str = ""
    final_state: str = ""


class CharacterNode(BaseModel):
    """Model for character node in the graph."""

    name: str
    alternative_names: List[str] = []
    character_type: str = "minor"
    is_major_character: bool = False  # Legacy field, maintained for compatibility
    physical_appearance: Dict[str, str] = {}
    physical_appearance_evolution: List[Dict[str, str]] = []
    personality: List[str] = []
    background: str = ""
    story_role: str = ""
    archetype: str = ""
    key_actions: List[str] = []
    character_arc: str = ""
    dialogue_style: str = ""
    motivations: List[str] = []
    emotional_state: Dict[str, List[str]] = {}
    internal_conflicts: List[str] = []
    locations: List[str] = []
    image_generation_prompt: str = ""
    metrics: CharacterMetrics = CharacterMetrics()
    evolution: CharacterEvolution = CharacterEvolution()
    
    # Series-specific fields
    canonical_name: str = ""  # Standardized name across series
    first_appearance: Optional[str] = None  # Book where character first appeared
    last_appearance: Optional[str] = None  # Book where character last appeared
    role_changes: List[Dict[str, Any]] = []  # How role changes across books


class RelationshipMetrics(BaseModel):
    """Metrics for character relationships."""
    
    interaction_count: int = 0
    scene_count: int = 0
    dialogue_count: int = 0
    mention_together_count: int = 0
    

class RelationshipEvolution(BaseModel):
    """Tracking of relationship evolution throughout the narrative."""
    
    arc_points: List[Dict[str, str]] = []
    early_state: str = ""
    middle_development: str = ""
    final_state: str = ""


class RelationshipEdge(BaseModel):
    """Model for relationship edge in the graph."""

    character1: str
    character2: str
    relationship_type: str
    description: str
    strength: int
    evidence: str = ""
    dynamics: str = ""
    conflicts: List[str] = []
    balance: str = ""
    emotional_tone: str = ""
    pivotal_moments: List[Dict[str, str]] = []
    evolution: RelationshipEvolution = RelationshipEvolution()
    metrics: RelationshipMetrics = RelationshipMetrics()
    
    # Series-specific fields
    first_interaction: Optional[str] = None  # Book where relationship first appears
    significant_developments: List[Dict[str, Any]] = []  # Major relationship changes
    relationship_trajectory: str = ""  # Overall arc across series


class CharacterGraph:
    """Graph representation of characters and their relationships."""

    def __init__(self, title: str, author: str, series_metadata: Optional[SeriesMetadata] = None):
        """Initialize the character graph.

        Args:
            title: Book title
            author: Book author
            series_metadata: Optional metadata for book series
        """
        self.title = title
        self.author = author
        self.series_metadata = series_metadata or SeriesMetadata(book_title=title)
        self.graph = nx.Graph()
        self.graph.graph["title"] = title
        self.graph.graph["author"] = author
        
        # Add series metadata to graph attributes
        if self.series_metadata.is_in_series():
            self.graph.graph["series_name"] = self.series_metadata.series_name
            self.graph.graph["book_number"] = self.series_metadata.book_number
            self.graph.graph["total_books"] = self.series_metadata.total_books
            self.graph.graph["reading_order"] = self.series_metadata.reading_order
            self.graph.graph["series_genre"] = self.series_metadata.series_genre

    def add_character(self, character_data: Dict) -> None:
        """Add a character node to the graph.

        Args:
            character_data: Character data dictionary
        """
        # Set default values for new fields if they're not present
        for field in [
            "character_type", "physical_appearance_evolution", "archetype",
            "dialogue_style", "motivations", "emotional_state", "internal_conflicts",
            "locations", "metrics", "evolution"
        ]:
            if field not in character_data:
                if field in ["metrics", "evolution"]:
                    character_data[field] = {}  # Empty dict for complex types
                elif field in ["motivations", "internal_conflicts", "locations", "physical_appearance_evolution"]:
                    character_data[field] = []  # Empty list for list types
                elif field == "emotional_state":
                    character_data[field] = {}  # Empty dict for emotional state
                else:
                    character_data[field] = ""  # Empty string for string fields
        
        # Handle backward compatibility for character type
        if "is_major_character" in character_data and character_data["is_major_character"]:
            character_data["character_type"] = "major"
        elif "character_type" not in character_data:
            character_data["character_type"] = "minor"
        
        character = CharacterNode(**character_data)
        
        # Add node with all character attributes
        node_attrs = {
            "alternative_names": character.alternative_names,
            "character_type": character.character_type,
            "is_major_character": character.is_major_character,
            "physical_appearance": character.physical_appearance,
            "physical_appearance_evolution": character.physical_appearance_evolution,
            "personality": character.personality,
            "background": character.background,
            "story_role": character.story_role,
            "archetype": character.archetype,
            "key_actions": character.key_actions,
            "character_arc": character.character_arc,
            "dialogue_style": character.dialogue_style,
            "motivations": character.motivations,
            "emotional_state": character.emotional_state,
            "internal_conflicts": character.internal_conflicts,
            "locations": character.locations,
            "image_generation_prompt": character.image_generation_prompt,
            "metrics": character.metrics.model_dump(),
            "evolution": character.evolution.model_dump()
        }
        
        self.graph.add_node(character.name, **node_attrs)
        
        logger.debug(f"Added character node: {character.name}")

    def add_relationship(self, relationship_data: Dict) -> None:
        """Add a relationship edge to the graph.

        Args:
            relationship_data: Relationship data dictionary
        """
        relationship = RelationshipEdge(**relationship_data)
        
        # Check if both characters exist in the graph
        if not self.graph.has_node(relationship.character1):
            logger.warning(
                f"Cannot add relationship: Character '{relationship.character1}' not in graph"
            )
            return
            
        if not self.graph.has_node(relationship.character2):
            logger.warning(
                f"Cannot add relationship: Character '{relationship.character2}' not in graph"
            )
            return
        
        # Set default values for new fields if they're not present
        # Add edge with all relationship attributes
        edge_attrs = {
            "relationship_type": relationship.relationship_type,
            "description": relationship.description,
            "strength": relationship.strength,
            "evidence": relationship.evidence,
            "dynamics": relationship.dynamics,
            "conflicts": relationship.conflicts,
            "balance": relationship.balance,
            "emotional_tone": relationship.emotional_tone,
            "pivotal_moments": relationship.pivotal_moments,
            "evolution": relationship.evolution.model_dump(),
            "metrics": relationship.metrics.model_dump()
        }
        
        self.graph.add_edge(
            relationship.character1,
            relationship.character2,
            **edge_attrs
        )
        
        # Update interaction counts in character metrics if tracking metrics
        if relationship.metrics.interaction_count > 0:
            # Update character1's interaction count with character2
            char1_metrics = self.graph.nodes[relationship.character1]["metrics"]
            if "interaction_counts" not in char1_metrics:
                char1_metrics["interaction_counts"] = {}
            if relationship.character2 not in char1_metrics["interaction_counts"]:
                char1_metrics["interaction_counts"][relationship.character2] = 0
            char1_metrics["interaction_counts"][relationship.character2] += relationship.metrics.interaction_count
            
            # Update character2's interaction count with character1
            char2_metrics = self.graph.nodes[relationship.character2]["metrics"]
            if "interaction_counts" not in char2_metrics:
                char2_metrics["interaction_counts"] = {}
            if relationship.character1 not in char2_metrics["interaction_counts"]:
                char2_metrics["interaction_counts"][relationship.character1] = 0
            char2_metrics["interaction_counts"][relationship.character1] += relationship.metrics.interaction_count
        
        logger.debug(
            f"Added relationship edge: {relationship.character1} - {relationship.character2}"
        )

    def build_from_data(self, data: Dict) -> None:
        """Build the graph from character and relationship data.

        Args:
            data: Dictionary containing character and relationship data
        """
        # Add all characters first
        for character_data in data.get("characters", []):
            self.add_character(character_data)
        
        # Then add relationships
        for relationship_data in data.get("relationships", []):
            self.add_relationship(relationship_data)
        
        # Calculate centrality and update character metrics
        self.calculate_centrality_metrics(update_nodes=True)
        
        # Calculate mention/interaction counts if available
        self._update_mention_counts()
        
        logger.info(
            f"Built character graph with {self.graph.number_of_nodes()} characters and "
            f"{self.graph.number_of_edges()} relationships"
        )
        
    def _update_mention_counts(self):
        """Update mention counts from relationship data if available."""
        # First aggregate all relationship mentions
        for u, v, data in self.graph.edges(data=True):
            metrics = data.get("metrics", {})
            interaction_count = metrics.get("interaction_count", 0)
            mention_count = metrics.get("mention_together_count", 0)
            
            # Update node metrics
            if interaction_count > 0 or mention_count > 0:
                # Update u's metrics
                u_metrics = self.graph.nodes[u].get("metrics", {})
                if "mention_count" not in u_metrics:
                    u_metrics["mention_count"] = 0
                # Each relationship mention adds to the character's mention count
                u_metrics["mention_count"] += mention_count
                self.graph.nodes[u]["metrics"] = u_metrics
                
                # Update v's metrics
                v_metrics = self.graph.nodes[v].get("metrics", {})
                if "mention_count" not in v_metrics:
                    v_metrics["mention_count"] = 0
                v_metrics["mention_count"] += mention_count
                self.graph.nodes[v]["metrics"] = v_metrics

    def get_character(self, name: str) -> Optional[Dict]:
        """Get character data by name.

        Args:
            name: Character name

        Returns:
            Character data dictionary or None if not found
        """
        if not self.graph.has_node(name):
            # Check alternative names
            for node in self.graph.nodes:
                if name in self.graph.nodes[node].get("alternative_names", []):
                    return dict(name=node, **self.graph.nodes[node])
            return None
        
        return dict(name=name, **self.graph.nodes[name])

    def get_relationships(self, character_name: str) -> List[Dict]:
        """Get all relationships for a character.

        Args:
            character_name: Character name

        Returns:
            List of relationship dictionaries
        """
        if not self.graph.has_node(character_name):
            logger.warning(f"Character '{character_name}' not found in graph")
            return []
        
        relationships = []
        
        for neighbor in self.graph.neighbors(character_name):
            edge_data = self.graph.get_edge_data(character_name, neighbor)
            relationships.append({
                "character1": character_name,
                "character2": neighbor,
                **edge_data
            })
        
        return relationships

    def get_all_characters(self) -> List[Dict]:
        """Get all characters in the graph.

        Returns:
            List of character dictionaries
        """
        characters = []
        
        for node in self.graph.nodes:
            characters.append(dict(name=node, **self.graph.nodes[node]))
        
        return characters

    def get_major_characters(self) -> List[Dict]:
        """Get major characters in the graph.

        Returns:
            List of major character dictionaries
        """
        major_characters = []
        
        for node in self.graph.nodes:
            if self.graph.nodes[node].get("is_major_character", False):
                major_characters.append(dict(name=node, **self.graph.nodes[node]))
        
        return major_characters

    def get_all_relationships(self) -> List[Dict]:
        """Get all relationships in the graph.

        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        for u, v, data in self.graph.edges(data=True):
            relationships.append({
                "character1": u,
                "character2": v,
                **data
            })
        
        return relationships

    def save_to_file(self, file_path: Path) -> None:
        """Save the graph to a JSON file.

        Args:
            file_path: Path to save the graph to
        """
        # Convert series metadata to dictionary if present
        series_data = {}
        if self.series_metadata and self.series_metadata.is_in_series():
            series_data = self.series_metadata.model_dump()
        
        data = {
            "title": self.title,
            "author": self.author,
            "series_metadata": series_data,
            "characters": self.get_all_characters(),
            "relationships": self.get_all_relationships(),
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved character graph to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Path) -> "CharacterGraph":
        """Load the graph from a JSON file.

        Args:
            file_path: Path to load the graph from

        Returns:
            CharacterGraph instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract series metadata if present
        series_data = data.get("series_metadata", {})
        series_metadata = None
        if series_data:
            series_metadata = SeriesMetadata(**series_data)
            
        # Create the graph with series metadata if available
        graph = cls(
            data.get("title", "Unknown"), 
            data.get("author", "Unknown"),
            series_metadata
        )
        
        graph.build_from_data(data)
        
        logger.info(f"Loaded character graph from {file_path}")
        return graph

    def get_community_structure(self) -> Dict[int, List[str]]:
        """Detect communities in the character graph.

        Returns:
            Dictionary mapping community IDs to lists of character names
        """
        if self.graph.number_of_nodes() < 3:
            # Not enough nodes for community detection
            return {0: list(self.graph.nodes)}
        
        try:
            # Use Louvain method for community detection
            communities = nx.community.louvain_communities(self.graph)
            
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[i] = list(community)
            
            return community_dict
        except Exception as e:
            logger.warning(f"Error detecting communities: {str(e)}")
            # Return all characters as a single community
            return {0: list(self.graph.nodes)}

    def calculate_centrality_metrics(self, update_nodes: bool = True) -> Dict[str, float]:
        """Calculate centrality metrics for all characters and optionally update node attributes.
        
        Args:
            update_nodes: Whether to update node metrics attributes with calculated values
            
        Returns:
            Dictionary mapping character names to centrality scores
        """
        if self.graph.number_of_nodes() == 0:
            return {}
            
        try:
            # Create a weighted copy of the graph using interaction counts
            weighted_graph = nx.Graph()
            # First add all nodes
            for node in self.graph.nodes():
                weighted_graph.add_node(node)
                
            # Then add edges with weights based on interaction counts
            for u, v, data in self.graph.edges(data=True):
                # Extract interaction count from relationship metrics
                # Default to 1 if no interaction count is available
                interaction_count = data.get('metrics', {}).get('interaction_count', 1)
                # Add mention count if available to increase weight
                mention_count = data.get('metrics', {}).get('mention_together_count', 0)
                # Combine both metrics for a more meaningful weight
                weight = interaction_count + mention_count
                # Ensure minimum weight of 1
                weight = max(1, weight)
                weighted_graph.add_edge(u, v, weight=weight)
            
            # Try to calculate eigenvector centrality using weights with more iterations
            try:
                # First try with eigenvector_centrality_numpy which is more stable
                if nx.is_directed(weighted_graph):
                    centrality = nx.eigenvector_centrality(weighted_graph, weight='weight', max_iter=1000, tol=1e-6)
                else:
                    try:
                        import numpy as np
                        centrality = nx.eigenvector_centrality_numpy(weighted_graph, weight='weight')
                    except ImportError:
                        centrality = nx.eigenvector_centrality(weighted_graph, weight='weight', max_iter=1000, tol=1e-6)
            except Exception as e:
                logger.warning(f"Eigenvector centrality calculation failed: {str(e)}")
                # Fall back to pagerank which is more robust
                try:
                    centrality = nx.pagerank(weighted_graph, weight='weight')
                except Exception as e:
                    logger.warning(f"PageRank calculation failed: {str(e)}")
                    # Last resort: use degree centrality which always works
                    centrality = nx.degree_centrality(weighted_graph)
                    
            # Calculate other useful centrality metrics with weights where supported
            try:
                betweenness = nx.betweenness_centrality(weighted_graph, weight='weight')
            except Exception:
                # Fall back to unweighted betweenness
                betweenness = nx.betweenness_centrality(weighted_graph)
                
            degree = nx.degree_centrality(weighted_graph)
            
            # Also consider mention counts in importance calculation
            mention_factor = {}
            for node in self.graph.nodes:
                # Get mention count from node metrics, default to 0
                mention_count = self.graph.nodes[node].get('metrics', {}).get('mention_count', 0)
                # Normalize mention count relative to highest mention count in graph
                max_mentions = max([self.graph.nodes[n].get('metrics', {}).get('mention_count', 0) 
                                   for n in self.graph.nodes], default=1)
                if max_mentions > 0:
                    mention_factor[node] = mention_count / max_mentions
                else:
                    mention_factor[node] = 0
            
            # Create combined metrics
            combined_metrics = {}
            for node in self.graph.nodes:
                # Normalize to 0-10 scale for importance score
                eigen_score = centrality.get(node, 0) * 10
                between_score = betweenness.get(node, 0) * 10
                degree_score = degree.get(node, 0) * 10
                mention_score = mention_factor.get(node, 0) * 10
                
                # Calculate weighted importance score with all factors
                # Eigenvector still has most weight, but mention count now included
                importance = (eigen_score * 0.4) + (between_score * 0.2) + (degree_score * 0.2) + (mention_score * 0.2)
                
                combined_metrics[node] = {
                    "centrality_score": centrality.get(node, 0),
                    "betweenness_score": betweenness.get(node, 0),
                    "degree_score": degree.get(node, 0),
                    "mention_factor": mention_factor.get(node, 0),
                    "importance_score": importance
                }
            
            # Update node attributes if requested
            if update_nodes:
                for node, metrics in combined_metrics.items():
                    # Update the metrics in the node attributes
                    node_metrics = self.graph.nodes[node].get("metrics", {})
                    node_metrics["centrality_score"] = metrics["centrality_score"]
                    node_metrics["betweenness_score"] = metrics["betweenness_score"]
                    node_metrics["degree_score"] = metrics["degree_score"]
                    node_metrics["mention_factor"] = metrics["mention_factor"]
                    node_metrics["importance_score"] = metrics["importance_score"]
                    self.graph.nodes[node]["metrics"] = node_metrics
                    
                    # Also update character_type based on importance score
                    importance = metrics["importance_score"]
                    if importance > 6.5:  # Major character threshold
                        self.graph.nodes[node]["character_type"] = "major"
                        self.graph.nodes[node]["is_major_character"] = True
                    elif importance > 4.0:  # Supporting character threshold
                        self.graph.nodes[node]["character_type"] = "supporting"
                        self.graph.nodes[node]["is_major_character"] = False
                    else:  # Minor character
                        self.graph.nodes[node]["character_type"] = "minor"
                        self.graph.nodes[node]["is_major_character"] = False
            
            return centrality
            
        except Exception as e:
            logger.warning(f"Error calculating centrality: {str(e)}")
            # Fall back to degree centrality if eigenvector fails
            try:
                return nx.degree_centrality(self.graph)
            except Exception:
                return {node: 0.0 for node in self.graph.nodes}
    
    def get_central_characters(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the most central characters based on eigenvector centrality.

        Args:
            n: Number of central characters to return

        Returns:
            List of tuples (character_name, centrality_score)
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        # Calculate centrality but don't update nodes (we just want the scores)
        centrality = self.calculate_centrality_metrics(update_nodes=False)
        
        # Sort by centrality score
        sorted_centrality = sorted(
            centrality.items(), key=lambda x: x[1], reverse=True
        )
        
        # Return top N
        return sorted_centrality[:n]