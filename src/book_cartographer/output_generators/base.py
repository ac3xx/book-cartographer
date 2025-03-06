"""Base output generator for character graph data."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.book_cartographer.character_graph import CharacterGraph

logger = logging.getLogger(__name__)


class OutputGeneratorConfig(BaseModel):
    """Configuration for output generators."""

    output_file: Optional[Path] = None
    template_dir: Optional[Path] = None


class BaseOutputGenerator(ABC):
    """Base class for all output generators."""

    def __init__(self, config: OutputGeneratorConfig, graph: CharacterGraph):
        """Initialize the output generator.

        Args:
            config: Output generator configuration
            graph: Character graph to generate output from
        """
        self.config = config
        self.graph = graph

    @abstractmethod
    def generate(self) -> str:
        """Generate output from the character graph.

        Returns:
            Generated output as string
        """
        pass

    def save(self, content: str, output_file: Optional[Path] = None) -> None:
        """Save generated content to a file.

        Args:
            content: Content to save
            output_file: Path to save the content to, defaults to config
        """
        output_path = output_file or self.config.output_file
        
        if not output_path:
            logger.warning("No output file path provided, cannot save content")
            return
            
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(content)
            
        logger.info(f"Saved output to {output_path}")

    def run(self, output_file: Optional[Path] = None) -> str:
        """Run the output generator and save the output.

        Args:
            output_file: Optional output file path override

        Returns:
            Generated content
        """
        content = self.generate()
        
        if output_file or self.config.output_file:
            self.save(content, output_file)
            
        return content