"""Configuration management for the EPUB character graph generator."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import tomli
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    provider: str = Field(default="openai", description="LLM provider name")
    model: str = Field(default="gpt-4", description="Model name to use")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    max_tokens: int = Field(default=4096, description="Maximum tokens for completion")
    temperature: float = Field(default=0.7, description="Temperature for completions")
    fallback_providers: List[str] = Field(
        default_factory=list, description="Fallback providers if primary fails"
    )
    fallback_models: Dict[str, str] = Field(
        default_factory=dict, description="Model to use for each fallback provider"
    )
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    timeout: int = Field(default=60, description="Timeout in seconds for LLM requests")

    @field_validator("api_key")
    def validate_api_key(cls, v: Optional[str], info):
        """Validate API key by checking environment variables if not provided."""
        if v is not None:
            return v

        env_var = f"{info.data.get('provider', 'unknown').upper()}_API_KEY"
        if env_var in os.environ:
            return os.environ[env_var]
        return None


class ExcerptLimits(BaseModel):
    """Configuration for excerpt limits based on character type."""
    
    major_character: int = Field(default=20, description="Excerpt limit for major characters")
    supporting_character: int = Field(default=10, description="Excerpt limit for supporting characters")
    minor_character: int = Field(default=5, description="Excerpt limit for minor characters")
    relationship: int = Field(default=15, description="Excerpt limit for character relationships")


class SeriesConfig(BaseModel):
    """Configuration for series processing."""
    
    enable_series_tracking: bool = Field(default=False, description="Enable series tracking features")
    series_name: Optional[str] = Field(default=None, description="Series name (overrides auto-detection)")
    character_similarity_threshold: float = Field(default=0.75, description="Threshold for matching characters across books (0.0-1.0)")
    track_character_evolution: bool = Field(default=True, description="Track character evolution across books in the series")
    cross_book_references: bool = Field(default=True, description="Include cross-references between books")
    series_metadata_file: Optional[Path] = Field(default=None, description="Path to series metadata file")
    auto_detect_series: bool = Field(default=True, description="Attempt to auto-detect series information from book metadata")
    

class ProcessingConfig(BaseModel):
    """Configuration for NLP processing."""

    spacy_model: str = Field(default="en_core_web_lg", description="spaCy model to use")
    batch_size: int = Field(default=10, description="Batch size for processing")
    max_chunk_size: int = Field(
        default=1000, description="Maximum chunk size for processing"
    )
    min_entity_occurrences: int = Field(
        default=3, description="Minimum occurrences for entity to be considered relevant"
    )
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "book-cartographer",
        description="Directory for caching LLM responses",
    )
    use_cache: bool = Field(default=True, description="Whether to use caching")
    parallel_requests: int = Field(
        default=5, description="Number of parallel LLM requests"
    )
    extract_relationships: bool = Field(
        default=True, description="Whether to extract character relationships"
    )
    extract_groups: bool = Field(
        default=True, description="Whether to extract character groups/factions"
    )
    use_llm_for_nlp: bool = Field(
        default=False, description="Whether to use LLM for initial NLP entity extraction"
    )
    all_character_relationships: bool = Field(
        default=False, description="Extract relationships between all characters, not just major ones"
    )
    track_metrics: bool = Field(
        default=True, description="Track additional metrics like mention counts and centrality scores"
    )
    store_raw_metrics: bool = Field(
        default=True, description="Store raw metrics in output JSON"
    )
    track_character_evolution: bool = Field(
        default=True, description="Track character evolution throughout the narrative"
    )
    comprehensive_analysis: bool = Field(
        default=True, description="Perform comprehensive character analysis including dialogue, motivations, and emotions"
    )
    store_intermediate_outputs: bool = Field(
        default=False, description="Store intermediate processing outputs like character excerpts and LLM responses"
    )
    excerpt_limits: ExcerptLimits = Field(
        default_factory=ExcerptLimits, description="Limits for number of excerpts by character type"
    )


class OutputConfig(BaseModel):
    """Configuration for output generation."""

    mode: str = Field(
        default="image-prompts",
        description="Output mode (image-prompts or kodict)",
    )
    output_file: Optional[Path] = Field(
        default=None, description="Path to output file"
    )
    template_dir: Optional[Path] = Field(
        default=None, description="Path to template directory"
    )
    include_minor_characters: bool = Field(
        default=False, description="Whether to include minor characters in output"
    )
    image_size: str = Field(
        default="1024x1024", description="Size for image prompts (e.g., 1024x1024)"
    )
    image_style: str = Field(
        default="realistic", description="Style for image prompts (e.g., realistic, anime)"
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    series: SeriesConfig = Field(default_factory=SeriesConfig)
    log_level: str = Field(default="INFO", description="Logging level")
    save_interim_data: bool = Field(
        default=False, description="Whether to save interim processing data"
    )
    interim_data_dir: Path = Field(
        default=Path("./interim_data"), description="Directory for interim data"
    )

    @classmethod
    def from_file(cls, file_path: Path) -> "AppConfig":
        """Load configuration from a file.
        
        Args:
            file_path: Path to a TOML or JSON configuration file.
            
        Returns:
            An instance of AppConfig with values from the config file.
        """
        if not file_path.exists():
            logger.warning(f"Config file {file_path} not found, using defaults")
            return cls()

        try:
            # Determine file type by extension
            if file_path.suffix.lower() in ['.toml', '.tml']:
                with open(file_path, 'rb') as f:
                    config_data = tomli.load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {file_path.suffix}")
                return cls()
                
            # Convert Path strings to Path objects
            config_data = cls._process_path_values(config_data)
            
            # Create the config object
            return cls.model_validate(config_data)
            
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {str(e)}")
            return cls()
    
    @staticmethod
    def _process_path_values(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string path values to Path objects in nested dictionaries.
        
        Args:
            config_data: Dictionary with configuration data.
            
        Returns:
            Dictionary with string paths converted to Path objects.
        """
        result = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                result[key] = AppConfig._process_path_values(value)
            elif isinstance(value, str) and key in [
                'cache_dir', 'output_file', 'template_dir', 'interim_data_dir'
            ]:
                result[key] = Path(value)
            else:
                result[key] = value
                
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.
        
        Returns:
            A dictionary representation of the config.
        """
        return self.model_dump()
    
    def save_to_file(self, file_path: Path) -> None:
        """Save the configuration to a file.
        
        Args:
            file_path: Path to save the configuration to.
        """
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Convert Path objects to strings and handle None values
        def convert_values(data: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    result[key] = convert_values(value)
                elif isinstance(value, Path):
                    result[key] = str(value)
                elif value is None:
                    # Skip None values for TOML serialization
                    continue
                else:
                    result[key] = value
            return result
        
        config_dict = convert_values(config_dict)
        
        # Determine file type by extension and save
        if file_path.suffix.lower() in ['.toml', '.tml']:
            try:
                import tomli_w
                with open(file_path, 'wb') as f:
                    tomli_w.dump(config_dict, f)
            except ImportError:
                logger.error("tomli_w not installed, saving as JSON instead")
                with open(file_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            # Default to JSON
            with open(file_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)


def get_config(config_file: Optional[Path] = None) -> AppConfig:
    """Get the application configuration.
    
    Args:
        config_file: Optional path to configuration file.
        
    Returns:
        An AppConfig instance with values from the config file if provided.
    """
    if config_file and config_file.exists():
        return AppConfig.from_file(config_file)
    
    # Look for config in default locations if not provided
    default_locations = [
        Path("./config.toml"),
        Path("./config.json"),
        Path.home() / ".config" / "book-cartographer" / "config.toml",
        Path.home() / ".config" / "book-cartographer" / "config.json",
    ]
    
    for loc in default_locations:
        if loc.exists():
            logger.info(f"Using config file found at {loc}")
            return AppConfig.from_file(loc)
    
    return AppConfig()