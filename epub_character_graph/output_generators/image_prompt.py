"""Image prompt generator for character visualization."""

import logging
from typing import Optional

import jinja2

from epub_character_graph.character_graph import CharacterGraph
from epub_character_graph.output_generators.base import (
    BaseOutputGenerator,
    OutputGeneratorConfig,
)

logger = logging.getLogger(__name__)

# Default template for image prompts
DEFAULT_TEMPLATE = """# AI Image Generation Prompts for "{{ graph.title }}"

{% for character in characters %}
## {{ character.name }}

{{ character.image_generation_prompt }}

**Additional details:**
{% if character.physical_appearance %}
- **Physical features:** {{ character.physical_appearance.physical_features }}
- **Age:** {{ character.physical_appearance.age }}
- **Clothing:** {{ character.physical_appearance.clothing }}
{% endif %}
{% if character.personality %}
- **Personality:** {{ ", ".join(character.personality[:3]) }}
{% endif %}

---
{% endfor %}
"""


class ImagePromptGenerator(BaseOutputGenerator):
    """Generator for AI image generation prompts."""

    def __init__(
        self, 
        config: OutputGeneratorConfig, 
        graph: CharacterGraph,
        template: Optional[str] = None,
    ):
        """Initialize the image prompt generator.

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
            template_name = "image_prompt.j2"
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

    def generate(self) -> str:
        """Generate image prompts for characters.

        Returns:
            Formatted image prompts as string
        """
        # Get all characters from the graph
        characters = self.graph.get_all_characters()
        
        # Sort characters by importance (major characters first)
        characters.sort(key=lambda x: (not x.get("is_major_character", False), x["name"]))
        
        # Render the template
        content = self.template_obj.render(
            graph=self.graph,
            characters=characters,
        )
        
        logger.info(f"Generated image prompts for {len(characters)} characters")
        return content

    def generate_for_character(self, character_name: str) -> Optional[str]:
        """Generate image prompt for a specific character.

        Args:
            character_name: Character name

        Returns:
            Formatted image prompt or None if character not found
        """
        character = self.graph.get_character(character_name)
        if not character:
            logger.warning(f"Character '{character_name}' not found")
            return None
        
        # Create a simple template for single character
        single_template = """# AI Image Generation Prompt for {{ character.name }}

{{ character.image_generation_prompt }}

**Additional details:**
{% if character.physical_appearance %}
- **Physical features:** {{ character.physical_appearance.physical_features }}
- **Age:** {{ character.physical_appearance.age }}
- **Clothing:** {{ character.physical_appearance.clothing }}
{% endif %}
{% if character.personality %}
- **Personality:** {{ ", ".join(character.personality[:3]) }}
{% endif %}
"""
        template_obj = jinja2.Template(single_template)
        
        content = template_obj.render(
            character=character,
        )
        
        logger.info(f"Generated image prompt for character '{character_name}'")
        return content