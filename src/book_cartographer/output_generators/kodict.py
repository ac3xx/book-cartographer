"""KOReader dictionary format generator for character information."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import jinja2

from src.book_cartographer.character_graph import CharacterGraph
from src.book_cartographer.output_generators.base import (
    BaseOutputGenerator,
    OutputGeneratorConfig,
)

logger = logging.getLogger(__name__)

# Default template for KOReader dictionary entries
DEFAULT_TEMPLATE = """{{ character.name }}

{% if character.alternative_names %}
Also known as: {{ ", ".join(character.alternative_names) }}
{% endif %}

{% if character.story_role %}
Role: {{ character.story_role }}
{% endif %}

{% if character.physical_appearance %}
#### Appearance
{% if character.physical_appearance.age %}
Age: {{ character.physical_appearance.age }}
{% endif %}
{% if character.physical_appearance.physical_features %}
Physical features: {{ character.physical_appearance.physical_features }}
{% endif %}
{% if character.physical_appearance.clothing %}
Clothing: {{ character.physical_appearance.clothing }}
{% endif %}
{% endif %}

{% if character.personality %}
#### Personality
{{ ", ".join(character.personality) }}
{% endif %}

{% if character.background %}
#### Background
{{ character.background }}
{% endif %}

{% if relationships %}
#### Relationships
{% for rel in relationships %}
- {{ rel.character2 }}: {{ rel.description }}
{% endfor %}
{% endif %}

{% if character.key_actions %}
#### Key Actions
{% for action in character.key_actions %}
- {{ action }}
{% endfor %}
{% endif %}

{% if character.character_arc %}
#### Character Arc
{{ character.character_arc }}
{% endif %}
"""


class KODictGenerator(BaseOutputGenerator):
    """Generator for KOReader dictionary format."""

    def __init__(
        self, 
        config: OutputGeneratorConfig, 
        graph: CharacterGraph,
        template: Optional[str] = None,
    ):
        """Initialize the KOReader dictionary generator.

        Args:
            config: Output generator configuration
            graph: Character graph
            template: Optional custom template string
        """
        super().__init__(config, graph)
        self.template = template or DEFAULT_TEMPLATE
        self._setup_template_engine()

    def _setup_template_engine(self) -> None:
        """Set up the Jinja2 template engine."""
        if self.config.template_dir:
            # Use template from directory if provided
            template_loader = jinja2.FileSystemLoader(self.config.template_dir)
            self.env = jinja2.Environment(loader=template_loader)
            template_name = "kodict_entry.j2"
            try:
                self.template_obj = self.env.get_template(template_name)
                logger.info(f"Loaded template from {self.config.template_dir / template_name}")
            except jinja2.exceptions.TemplateNotFound:
                # Fall back to default template
                logger.warning(f"Template {template_name} not found, using default")
                self.template_obj = jinja2.Template(self.template)
        else:
            # Use provided or default template
            self.env = jinja2.Environment()
            self.template_obj = jinja2.Template(self.template)

    def _generate_stardict_header(self) -> str:
        """Generate the StarDict header for the dictionary.

        Returns:
            StarDict header string
        """
        return f"""#bookname:{self.graph.title} Character Encyclopedia
#author:{self.graph.author}
#description:Character dictionary for '{self.graph.title}' by {self.graph.author}
#year:{Path.cwd().stat().st_mtime_ns // 1_000_000_000}
#source:Generated from EPUB
#publisher:book-cartographer

"""

    def _format_entry(self, character: Dict) -> str:
        """Format a single dictionary entry for a character.

        Args:
            character: Character data dictionary

        Returns:
            Formatted dictionary entry
        """
        # Get character relationships
        relationships = self.graph.get_relationships(character["name"])
        
        # Render the template
        entry = self.template_obj.render(
            character=character,
            relationships=relationships,
        )
        
        return entry

    def generate(self) -> str:
        """Generate a KOReader dictionary with character entries.

        Returns:
            Dictionary content in StarDict format
        """
        # Start with the header
        content = self._generate_stardict_header()
        
        # Get all characters from the graph
        characters = self.graph.get_all_characters()
        
        # Sort characters by importance (major characters first)
        characters.sort(key=lambda x: (not x.get("is_major_character", False), x["name"]))
        
        # Add each character entry
        for character in characters:
            entry = self._format_entry(character)
            content += f"{character['name']}\n{entry}\n\n"
            
            # Add entries for alternative names that redirect to the main character
            for alt_name in character.get("alternative_names", []):
                content += f"{alt_name}\nSee {character['name']}\n\n"
        
        logger.info(f"Generated KOReader dictionary with {len(characters)} character entries")
        return content

    def generate_json_dictionary(self) -> Dict:
        """Generate a JSON dictionary with character entries.

        Returns:
            Dictionary in JSON format
        """
        # Get all characters from the graph
        characters = self.graph.get_all_characters()
        
        # Build the dictionary
        dictionary = {
            "metadata": {
                "title": f"{self.graph.title} Character Encyclopedia",
                "author": self.graph.author,
                "description": f"Character dictionary for '{self.graph.title}' by {self.graph.author}",
                "source": "Generated from EPUB",
                "generator": "book-cartographer",
            },
            "entries": {}
        }
        
        # Add each character entry
        for character in characters:
            # Get character relationships
            relationships = self.graph.get_relationships(character["name"])
            
            # Create entry
            dictionary["entries"][character["name"]] = {
                "name": character["name"],
                "alternative_names": character.get("alternative_names", []),
                "is_major_character": character.get("is_major_character", False),
                "physical_appearance": character.get("physical_appearance", {}),
                "personality": character.get("personality", []),
                "background": character.get("background", ""),
                "story_role": character.get("story_role", ""),
                "key_actions": character.get("key_actions", []),
                "character_arc": character.get("character_arc", ""),
                "relationships": [
                    {
                        "name": rel["character2"],
                        "relationship_type": rel["relationship_type"],
                        "description": rel["description"],
                    }
                    for rel in relationships
                ],
            }
            
            # Add entries for alternative names that redirect to the main character
            for alt_name in character.get("alternative_names", []):
                dictionary["entries"][alt_name] = {
                    "redirect": character["name"],
                }
        
        logger.info(f"Generated JSON dictionary with {len(characters)} character entries")
        return dictionary

    def save_json(self, output_file: Optional[Path] = None) -> None:
        """Save the dictionary as JSON.

        Args:
            output_file: Path to save the JSON file
        """
        dictionary = self.generate_json_dictionary()
        
        output_path = output_file or self.config.output_file
        if not output_path:
            logger.warning("No output file path provided, cannot save JSON")
            return
            
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(dictionary, f, indent=2)
            
        logger.info(f"Saved JSON dictionary to {output_path}")