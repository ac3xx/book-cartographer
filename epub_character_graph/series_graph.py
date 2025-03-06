"""Series graph implementation for managing character data across multiple books."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

import networkx as nx
from pydantic import BaseModel, Field

from epub_character_graph.character_graph import (
    CharacterGraph, 
    CharacterNode, 
    RelationshipEdge,
    SeriesMetadata
)

logger = logging.getLogger(__name__)


class BookReference(BaseModel):
    """Reference to a book within a series."""
    
    title: str
    book_number: int = 0
    reading_order: int = 0
    graph_file: Optional[Path] = None
    publication_date: Optional[str] = None


class CharacterReference(BaseModel):
    """Reference to a character across multiple books."""
    
    canonical_name: str
    aliases: Dict[str, List[str]] = {}  # Book title -> list of names used in that book
    appearances: List[str] = []  # List of book titles where character appears
    first_appearance: Optional[str] = None
    last_appearance: Optional[str] = None
    importance_by_book: Dict[str, float] = {}  # Book title -> importance score
    character_type_by_book: Dict[str, str] = {}  # Book title -> character type
    evolution: List[Dict[str, Any]] = []  # List of evolution points across books
    
    
class RelationshipReference(BaseModel):
    """Reference to a character relationship across multiple books."""
    
    character1: str  # Canonical name
    character2: str  # Canonical name
    appearances: List[str] = []  # List of book titles where relationship appears
    first_appearance: Optional[str] = None
    relationship_type_by_book: Dict[str, str] = {}  # Book title -> relationship type
    strength_by_book: Dict[str, int] = {}  # Book title -> relationship strength
    evolution: List[Dict[str, Any]] = []  # List of evolution points across books


class SeriesGraph:
    """Container for multiple character graphs from a book series."""
    
    def __init__(self, series_name: str, author: str):
        """Initialize the series graph.
        
        Args:
            series_name: Name of the book series
            author: Author of the series
        """
        self.series_name = series_name
        self.author = author
        self.books: Dict[str, BookReference] = {}
        self.characters: Dict[str, CharacterReference] = {}
        self.relationships: List[RelationshipReference] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def add_book(self, graph: CharacterGraph) -> None:
        """Add a book's character graph to the series.
        
        Args:
            graph: Character graph for a book in the series
        """
        # Extract series metadata
        series_meta = graph.series_metadata
        book_title = graph.title
        
        # Skip if book is already added
        if book_title in self.books:
            logger.warning(f"Book '{book_title}' is already in the series graph")
            return
            
        # Add book reference
        book_ref = BookReference(
            title=book_title,
            book_number=series_meta.book_number,
            reading_order=series_meta.reading_order
        )
        self.books[book_title] = book_ref
        
        # Process characters from this book
        for character in graph.get_all_characters():
            self._process_character(character, book_title)
            
        # Process relationships from this book
        for relationship in graph.get_all_relationships():
            self._process_relationship(relationship, book_title)
            
        # Update timestamp
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Added book '{book_title}' to series '{self.series_name}'")
    
    def _process_character(self, character: Dict, book_title: str) -> None:
        """Process a character from a book and update series data.
        
        Args:
            character: Character data dictionary
            book_title: Title of the book this character is from
        """
        name = character.get("name", "")
        if not name:
            return
            
        # Determine canonical name (for now, use the current name)
        # In a real implementation, this would use a more sophisticated matching algorithm
        canonical_name = self._get_canonical_name(name, character, book_title)
        
        # Get or create character reference
        if canonical_name not in self.characters:
            self.characters[canonical_name] = CharacterReference(
                canonical_name=canonical_name,
                first_appearance=book_title,
                last_appearance=book_title,
                appearances=[book_title]
            )
        else:
            char_ref = self.characters[canonical_name]
            if book_title not in char_ref.appearances:
                char_ref.appearances.append(book_title)
                # Update last appearance
                char_ref.last_appearance = book_title
        
        # Update aliases
        alt_names = character.get("alternative_names", [])
        self.characters[canonical_name].aliases[book_title] = [name] + alt_names
        
        # Update importance and character type
        importance = character.get("metrics", {}).get("importance_score", 0.0)
        char_type = character.get("character_type", "minor")
        
        self.characters[canonical_name].importance_by_book[book_title] = importance
        self.characters[canonical_name].character_type_by_book[book_title] = char_type
        
        # Add evolution point if available
        character_arc = character.get("character_arc", "")
        if character_arc:
            evolution_point = {
                "book": book_title,
                "character_arc": character_arc,
                "story_role": character.get("story_role", ""),
                "archetype": character.get("archetype", "")
            }
            self.characters[canonical_name].evolution.append(evolution_point)
    
    def _process_relationship(self, relationship: Dict, book_title: str) -> None:
        """Process a relationship from a book and update series data.
        
        Args:
            relationship: Relationship data dictionary
            book_title: Title of the book this relationship is from
        """
        char1 = relationship.get("character1", "")
        char2 = relationship.get("character2", "")
        if not char1 or not char2:
            return
            
        # Get canonical names for both characters
        canon1 = self._get_canonical_name_from_book_name(char1, book_title)
        canon2 = self._get_canonical_name_from_book_name(char2, book_title)
        
        if not canon1 or not canon2:
            logger.warning(f"Could not find canonical names for relationship: {char1} - {char2}")
            return
            
        # Ensure consistent ordering of characters
        if canon1 > canon2:
            canon1, canon2 = canon2, canon1
            
        # Find existing relationship or create new one
        rel_ref = None
        for rel in self.relationships:
            if rel.character1 == canon1 and rel.character2 == canon2:
                rel_ref = rel
                break
                
        if not rel_ref:
            rel_ref = RelationshipReference(
                character1=canon1,
                character2=canon2,
                first_appearance=book_title,
                appearances=[book_title]
            )
            self.relationships.append(rel_ref)
        elif book_title not in rel_ref.appearances:
            rel_ref.appearances.append(book_title)
            
        # Update relationship type and strength
        rel_type = relationship.get("relationship_type", "unknown")
        strength = relationship.get("strength", 0)
        
        rel_ref.relationship_type_by_book[book_title] = rel_type
        rel_ref.strength_by_book[book_title] = strength
        
        # Add evolution point if available
        dynamics = relationship.get("dynamics", "")
        if dynamics:
            evolution_point = {
                "book": book_title,
                "dynamics": dynamics,
                "emotional_tone": relationship.get("emotional_tone", ""),
                "conflicts": relationship.get("conflicts", [])
            }
            rel_ref.evolution.append(evolution_point)
    
    def _get_canonical_name(self, name: str, character: Dict, book_title: str) -> str:
        """Determine the canonical name for a character.
        
        Args:
            name: Current character name
            character: Character data dictionary
            book_title: Title of the book this character is from
            
        Returns:
            Canonical name for this character
        """
        # Check if this character already has a canonical name in the current book
        if "canonical_name" in character and character["canonical_name"]:
            return character["canonical_name"]
            
        # Check if this name or any alternative names match existing characters
        alt_names = character.get("alternative_names", [])
        all_names = [name] + alt_names
        
        for canon_name, char_ref in self.characters.items():
            # Check each book's aliases for this character
            for book, aliases in char_ref.aliases.items():
                for alias in aliases:
                    if alias in all_names:
                        return canon_name
                        
        # No match found, use current name as canonical
        return name
    
    def _get_canonical_name_from_book_name(self, book_name: str, book_title: str) -> Optional[str]:
        """Find canonical name for a character based on their name in a specific book.
        
        Args:
            book_name: Character name used in the book
            book_title: Title of the book
            
        Returns:
            Canonical name if found, None otherwise
        """
        for canon_name, char_ref in self.characters.items():
            if book_title in char_ref.aliases:
                if book_name in char_ref.aliases[book_title]:
                    return canon_name
        return None
    
    def get_character_by_canonical_name(self, canonical_name: str) -> Optional[CharacterReference]:
        """Get a character by their canonical name.
        
        Args:
            canonical_name: Canonical name of the character
            
        Returns:
            CharacterReference if found, None otherwise
        """
        return self.characters.get(canonical_name)
    
    def get_character_by_book_name(self, book_name: str, book_title: str) -> Optional[CharacterReference]:
        """Get a character by their name in a specific book.
        
        Args:
            book_name: Name used in the book
            book_title: Title of the book
            
        Returns:
            CharacterReference if found, None otherwise
        """
        canon_name = self._get_canonical_name_from_book_name(book_name, book_title)
        if canon_name:
            return self.characters.get(canon_name)
        return None
    
    def get_relationships_for_character(self, canonical_name: str) -> List[RelationshipReference]:
        """Get all relationships for a character.
        
        Args:
            canonical_name: Canonical name of the character
            
        Returns:
            List of relationships the character is involved in
        """
        return [
            rel for rel in self.relationships 
            if rel.character1 == canonical_name or rel.character2 == canonical_name
        ]
    
    def get_books_by_reading_order(self) -> List[BookReference]:
        """Get books sorted by reading order.
        
        Returns:
            List of books sorted by reading order
        """
        return sorted(
            self.books.values(), 
            key=lambda book: (book.reading_order, book.book_number)
        )
    
    def get_recurring_characters(self, min_books: int = 2) -> List[CharacterReference]:
        """Get characters that appear in multiple books.
        
        Args:
            min_books: Minimum number of books a character must appear in
            
        Returns:
            List of recurring characters
        """
        return [
            char for char in self.characters.values() 
            if len(char.appearances) >= min_books
        ]
    
    def get_evolving_relationships(self) -> List[RelationshipReference]:
        """Get relationships that evolve across multiple books.
        
        Returns:
            List of evolving relationships
        """
        return [
            rel for rel in self.relationships 
            if len(rel.appearances) > 1 and len(rel.evolution) > 1
        ]
    
    def save_to_file(self, file_path: Path) -> None:
        """Save the series graph to a JSON file.
        
        Args:
            file_path: Path to save the series graph to
        """
        # Convert book references to serializable format
        serializable_books = {}
        for title, book in self.books.items():
            book_dict = book.model_dump()
            # Convert Path objects to strings
            if book_dict.get("graph_file"):
                book_dict["graph_file"] = str(book_dict["graph_file"])
            serializable_books[title] = book_dict
            
        data = {
            "series_name": self.series_name,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "books": serializable_books,
            "characters": {name: char.model_dump() for name, char in self.characters.items()},
            "relationships": [rel.model_dump() for rel in self.relationships]
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved series graph to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> "SeriesGraph":
        """Load a series graph from a JSON file.
        
        Args:
            file_path: Path to load the series graph from
            
        Returns:
            SeriesGraph instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        series_graph = cls(
            data.get("series_name", "Unknown Series"),
            data.get("author", "Unknown Author")
        )
        
        # Restore timestamps
        series_graph.created_at = data.get("created_at", series_graph.created_at)
        series_graph.updated_at = data.get("updated_at", series_graph.updated_at)
        
        # Restore books
        for title, book_data in data.get("books", {}).items():
            # Convert string paths back to Path objects
            if "graph_file" in book_data and book_data["graph_file"]:
                book_data["graph_file"] = Path(book_data["graph_file"])
            series_graph.books[title] = BookReference(**book_data)
            
        # Restore characters
        for name, char_data in data.get("characters", {}).items():
            series_graph.characters[name] = CharacterReference(**char_data)
            
        # Restore relationships
        for rel_data in data.get("relationships", []):
            series_graph.relationships.append(RelationshipReference(**rel_data))
            
        logger.info(f"Loaded series graph from {file_path}")
        return series_graph