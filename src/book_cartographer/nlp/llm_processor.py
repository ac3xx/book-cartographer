"""LLM processing module for enhanced character extraction and analysis."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TypeVar, cast

import diskcache
import litellm
from litellm import acompletion
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from src.book_cartographer.config import LLMConfig, ProcessingConfig
from src.book_cartographer.nlp.prompts import (
    generate_character_description_prompt,
    generate_character_verification_prompt,
    generate_relationship_extraction_prompt,
)

logger = logging.getLogger(__name__)


class CharacterMetrics(BaseModel):
    """Metrics and statistics for a character."""
    
    mention_count: int = 0
    dialogue_count: int = 0
    scene_count: int = 0
    interaction_counts: Dict[str, int] = {}  # Character name -> count
    centrality_score: float = 0.0
    importance_score: float = 0.0  # 0-10 scale
    
    
class CharacterEvolution(BaseModel):
    """Tracking of a character's evolution throughout the narrative."""
    
    arc_points: List[Dict[str, str]] = []  # List of evolution points with position and description
    key_moments: List[Dict[str, str]] = []  # Key moments with position and description
    early_presentation: str = ""  # How character is introduced/presented early
    middle_development: str = ""  # How character develops in middle
    final_state: str = ""  # Character's final state/resolution


class Character(BaseModel):
    """Model for character data."""

    name: str
    alternative_names: List[str] = []
    confidence: str
    character_type: str = "minor"  # "major", "supporting", "minor"
    is_major_character: bool = False  # Legacy field, maintained for compatibility
    physical_appearance: Dict[str, str] = {}
    physical_appearance_evolution: List[Dict[str, str]] = []  # Changes in appearance over time
    personality: List[str] = []
    background: str = ""
    story_role: str = ""
    archetype: str = ""  # Hero, Mentor, Ally, etc.
    relationships: List[Dict[str, str]] = []
    key_actions: List[str] = []
    character_arc: str = ""
    dialogue_style: str = ""  # Character's speaking patterns/style
    motivations: List[str] = []  # Character's goals and motivations
    emotional_state: Dict[str, List[str]] = {}  # Emotions at different points
    internal_conflicts: List[str] = []  # Character's internal struggles
    locations: List[str] = []  # Key locations associated with character
    image_generation_prompt: str = ""
    metrics: CharacterMetrics = CharacterMetrics()
    evolution: CharacterEvolution = CharacterEvolution()


class RelationshipMetrics(BaseModel):
    """Metrics for character relationships."""
    
    interaction_count: int = 0
    scene_count: int = 0
    dialogue_count: int = 0
    mention_together_count: int = 0
    

class RelationshipEvolution(BaseModel):
    """Tracking of relationship evolution throughout the narrative."""
    
    arc_points: List[Dict[str, str]] = []  # Evolution points with position and description
    early_state: str = ""  # Initial relationship state
    middle_development: str = ""  # How relationship develops
    final_state: str = ""  # Final relationship state
    

class Relationship(BaseModel):
    """Model for character relationship data."""

    character1: str
    character2: str
    relationship_type: str
    description: str
    strength: int  # 1-10 scale
    evidence: str = ""
    dynamics: str = ""  # Description of relationship dynamics
    conflicts: List[str] = []  # Points of conflict in relationship
    balance: str = ""  # Power balance in relationship
    emotional_tone: str = ""  # Emotional tone/quality
    pivotal_moments: List[Dict[str, str]] = []  # Key moments that defined relationship
    evolution: RelationshipEvolution = RelationshipEvolution()
    metrics: RelationshipMetrics = RelationshipMetrics()


class VerifiedCharacter(BaseModel):
    """A verified character from the book."""
    
    name: str
    alternative_names: List[str] = []
    confidence: str
    character_type: str = "minor"  # "major", "supporting", "minor" 
    is_major_character: bool = False  # Legacy field, maintained for compatibility
    archetype: str = ""  # Character archetype if identified
    metrics: Dict[str, Any] = {}  # Metrics for the character


class CharacterVerificationResult(BaseModel):
    """Result of character verification."""

    verified_characters: List[VerifiedCharacter]
    rejected_entries: List[str]
    missing_major_characters: List[str] = []


class RelationshipData(BaseModel):
    """Data for a relationship between characters."""
    
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


class RelationshipExtractionResult(BaseModel):
    """Result of relationship extraction."""

    relationships: List[RelationshipData]


class LLMProcessor:
    """Processor for LLM-based character and relationship extraction."""

    def __init__(self, llm_config: LLMConfig, processing_config: ProcessingConfig):
        """Initialize the LLM processor.

        Args:
            llm_config: LLM configuration
            processing_config: Processing configuration
        """
        self.llm_config = llm_config
        self.processing_config = processing_config
        
        # Setup cache if enabled
        if self.processing_config.use_cache:
            cache_dir = self.processing_config.cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(cache_dir))
            logger.info(f"Using cache at {cache_dir}")
        else:
            self.cache = None
        
        # Enable JSON schema validation
        litellm.enable_json_schema_validation = True

    async def _call_llm_with_schema(self, 
                             prompt: str, 
                             response_model: Optional[BaseModel] = None
                            ) -> Any:
        """Call the LLM with schema validation, error handling and retries.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Optional Pydantic model for schema validation

        Returns:
            Validated structured response or raw text response
        """
        # Use cache if available
        if self.cache is not None:
            cache_key = f"llm_{hash(prompt)}_{hash(str(response_model))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Using cached LLM result")
                return cached_result

        # Prepare LLM call parameters
        llm_params = {
            "model": f"{self.llm_config.provider}/{self.llm_config.model}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature,
        }
        
        # Add response format if model is provided
        if response_model:
            llm_params["response_format"] = response_model
        
        if self.llm_config.api_key:
            llm_params["api_key"] = self.llm_config.api_key

        # Call LLM with fallback logic
        providers = [self.llm_config.provider] + self.llm_config.fallback_providers
        fallback_models = self.llm_config.fallback_models
        last_error = None
        
        for provider in providers:
            try:
                # Use provider-specific model if available in fallback_models
                if provider != self.llm_config.provider and provider in fallback_models:
                    model_name = fallback_models[provider]
                    llm_params["model"] = f"{provider}/{model_name}"
                else:
                    llm_params["model"] = f"{provider}/{self.llm_config.model}"
                
                logger.debug(f"Calling LLM with model: {llm_params['model']}")
                response = await acompletion(**llm_params)
                
                # Extract response content
                if response_model:
                    # For structured responses, we get a model instance back
                    response_data = response
                else:
                    # For text responses, extract content from message
                    response_data = response.choices[0].message.content
                
                # Cache the result if caching is enabled
                if self.cache is not None:
                    self.cache[cache_key] = response_data
                
                return response_data
            except Exception as e:
                last_error = e
                logger.warning(f"Error with provider {provider}: {str(e)}")
                if provider != providers[-1]:  # Don't log this for the last provider
                    logger.warning(f"Trying fallback provider")
        
        # If we get here, all providers failed
        error_msg = f"All LLM providers failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    async def verify_characters(
        self, title: str, author: str, potential_characters: Set[str]
    ) -> CharacterVerificationResult:
        """Verify extracted character names using LLM.

        Args:
            title: Book title
            author: Book author
            potential_characters: Set of potential character names

        Returns:
            Character verification result
        """
        prompt = generate_character_verification_prompt(
            title, author, potential_characters
        )
        
        logger.info(f"Verifying {len(potential_characters)} potential characters with LLM")
        
        try:
            # Use text response without schema validation
            response = await self._call_llm_with_schema(prompt)
            
            # Parse response as JSON
            if isinstance(response, str):
                text_response = response
            elif hasattr(response, "choices"):
                text_response = response.choices[0].message.content
            else:
                text_response = str(response)
                
            # Find JSON in the response
            import json
            import re
            
            # Try different ways to extract JSON from the response
            # First, try to find a JSON object
            json_match = re.search(r'({[\s\S]*})', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    result_data = json.loads(json_str)
                    verified_chars = []
                    for char_data in result_data.get("verified_characters", []):
                        verified_chars.append(VerifiedCharacter(**char_data))
                    
                    return CharacterVerificationResult(
                        verified_characters=verified_chars,
                        rejected_entries=result_data.get("rejected_entries", []),
                        missing_major_characters=result_data.get("missing_major_characters", [])
                    )
                except json.JSONDecodeError:
                    # Try to fix truncated JSON by completing it
                    try:
                        # Count opening and closing braces
                        open_braces = json_str.count('{')
                        close_braces = json_str.count('}')
                        # If missing closing braces, add them
                        if open_braces > close_braces:
                            json_str += "}" * (open_braces - close_braces)
                        result_data = json.loads(json_str)
                        verified_chars = []
                        for char_data in result_data.get("verified_characters", []):
                            verified_chars.append(VerifiedCharacter(**char_data))
                        
                        return CharacterVerificationResult(
                            verified_characters=verified_chars,
                            rejected_entries=result_data.get("rejected_entries", []),
                            missing_major_characters=result_data.get("missing_major_characters", [])
                        )
                    except:
                        # If that fails, try looking for verified_characters array directly
                        vc_match = re.search(r'"verified_characters"\s*:\s*(\[[\s\S]*?\])', text_response, re.DOTALL)
                        if vc_match:
                            try:
                                chars_array = json.loads(vc_match.group(1))
                                verified_chars = []
                                for char_data in chars_array:
                                    verified_chars.append(VerifiedCharacter(**char_data))
                                
                                return CharacterVerificationResult(
                                    verified_characters=verified_chars,
                                    rejected_entries=[],
                                    missing_major_characters=[]
                                )
                            except:
                                raise ValueError(f"Invalid JSON in response: {json_str}")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error in character verification: {str(e)}")
            error_str = str(e)
            
            # Attempt to extract character data from the error message
            if "Invalid JSON in response" in error_str and '"verified_characters"' in error_str:
                import re
                import json
                
                # Try to extract character objects from the error
                char_pattern = re.compile(r'{\s*"name":\s*"([^"]+)".*?}', re.DOTALL)
                matches = char_pattern.findall(error_str)
                
                if matches:
                    logger.info(f"Found {len(matches)} characters in error message, attempting recovery")
                    verified_chars = []
                    
                    for char_name in matches:
                        # Create a minimal valid character object for each name found
                        verified_chars.append(VerifiedCharacter(
                            name=char_name,
                            alternative_names=[],
                            confidence="high",
                            character_type="supporting",
                            is_major_character=False,
                            archetype="unknown"
                        ))
                    
                    return CharacterVerificationResult(
                        verified_characters=verified_chars,
                        rejected_entries=[]
                    )
            
            logger.debug(f"Raw response: {response if isinstance(response, str) else 'non-string response'}")
            # Return empty result as fallback
            return CharacterVerificationResult(
                verified_characters=[], 
                rejected_entries=[]
            )

    def _get_excerpt_limit_for_character(self, character_name: str, verified_characters: List[Dict]) -> int:
        """Determine the excerpt limit for a character based on their type.
        
        Args:
            character_name: Name of the character
            verified_characters: List of verified character data
            
        Returns:
            Number of excerpts to include
        """
        # Default to minor character limit
        excerpt_limit = self.processing_config.excerpt_limits.minor_character
        
        # Check if character exists in verified characters
        for char in verified_characters:
            if isinstance(char, dict) and char.get("name") == character_name:
                char_type = char.get("character_type", "minor")
                is_major = char.get("is_major_character", False)
                
                if char_type == "major" or is_major:
                    excerpt_limit = self.processing_config.excerpt_limits.major_character
                elif char_type == "supporting":
                    excerpt_limit = self.processing_config.excerpt_limits.supporting_character
                
                break
            elif hasattr(char, "name") and char.name == character_name:
                char_type = getattr(char, "character_type", "minor")
                is_major = getattr(char, "is_major_character", False)
                
                if char_type == "major" or is_major:
                    excerpt_limit = self.processing_config.excerpt_limits.major_character
                elif char_type == "supporting":
                    excerpt_limit = self.processing_config.excerpt_limits.supporting_character
                
                break
        
        return excerpt_limit
    
    async def extract_character_details(
        self, title: str, author: str, character_name: str, excerpts: List[str], 
        verified_characters: List[Dict] = None
    ) -> Character:
        """Extract detailed character information using LLM.

        Args:
            title: Book title
            author: Book author
            character_name: Name of the character
            excerpts: Text excerpts featuring the character
            verified_characters: Optional list of verified character data to determine excerpt limits

        Returns:
            Character details
        """
        # Determine excerpt limit based on character type if verified characters provided
        if verified_characters:
            excerpt_limit = self._get_excerpt_limit_for_character(character_name, verified_characters)
        else:
            # Default to standard limit if no verified characters provided
            excerpt_limit = self.processing_config.excerpt_limits.minor_character
        
        # Limit the number of excerpts based on character type
        if len(excerpts) > excerpt_limit:
            logger.info(f"Limiting {len(excerpts)} excerpts to {excerpt_limit} for character '{character_name}'")
            excerpts = excerpts[:excerpt_limit]
        
        prompt = generate_character_description_prompt(
            title, author, character_name, excerpts
        )
        
        logger.info(f"Extracting details for character '{character_name}' with LLM")
        
        try:
            # Use text response without schema validation to avoid ModelResponse issues
            response = await self._call_llm_with_schema(prompt)
            
            # Parse response as JSON
            if isinstance(response, str):
                text_response = response
            elif hasattr(response, "choices"):
                text_response = response.choices[0].message.content
            else:
                text_response = str(response)
                
            # Find JSON in the response
            import json
            import re
            
            # Find JSON-like structure
            json_match = re.search(r'({.*})', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    char_data = json.loads(json_str)
                    # Ensure name is correct
                    char_data["name"] = character_name
                    return Character(**char_data)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in response: {json_str}")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error parsing character details for '{character_name}': {str(e)}")
            logger.debug(f"Raw response: {response if isinstance(response, str) else 'non-string response'}")
            # Return basic character as fallback
            return Character(
                name=character_name,
                confidence="medium",
                physical_appearance={},
                personality=[],
                background="",
                story_role="",
                relationships=[],
                key_actions=[],
                character_arc="",
                image_generation_prompt=""
            )

    async def extract_relationships(
        self, title: str, author: str, characters: List[str], excerpts: List[str]
    ) -> List[Relationship]:
        """Extract character relationships using LLM.

        Args:
            title: Book title
            author: Book author
            characters: List of character names
            excerpts: Text excerpts showing character interactions

        Returns:
            List of character relationships
        """
        # Limit the number of excerpts based on the relationship limit setting
        relationship_excerpt_limit = self.processing_config.excerpt_limits.relationship
        if len(excerpts) > relationship_excerpt_limit:
            logger.info(f"Limiting {len(excerpts)} interaction excerpts to {relationship_excerpt_limit}")
            excerpts = excerpts[:relationship_excerpt_limit]
        
        prompt = generate_relationship_extraction_prompt(
            title, author, characters, excerpts
        )
        
        logger.info(f"Extracting relationships between {len(characters)} characters with LLM")
        
        try:
            # Use text response without schema validation
            response = await self._call_llm_with_schema(prompt)
            
            # Parse response as JSON
            if isinstance(response, str):
                text_response = response
            elif hasattr(response, "choices"):
                text_response = response.choices[0].message.content
            else:
                text_response = str(response)
                
            # Find JSON in the response
            import json
            import re
            
            # Find JSON-like structure
            json_match = re.search(r'({.*})', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    result_data = json.loads(json_str)
                    relationships = []
                    for rel_data in result_data.get("relationships", []):
                        # Ensure metrics are properly set with non-zero values
                        if "metrics" not in rel_data:
                            rel_data["metrics"] = {}
                        
                        # Set default values if not provided
                        metrics = rel_data["metrics"]
                        
                        # If interaction counts are still 0, estimate based on relationship strength
                        if not metrics.get("interaction_count", 0):
                            # Use relationship strength to estimate interaction count
                            strength = rel_data.get("strength", 5)
                            # Formula: stronger relationships have more interactions
                            # 1-3 strength: 1-5 interactions
                            # 4-7 strength: 5-20 interactions
                            # 8-10 strength: 20-50 interactions
                            if strength <= 3:
                                metrics["interaction_count"] = max(1, strength * 2)
                            elif strength <= 7:
                                metrics["interaction_count"] = max(5, strength * 3)
                            else:
                                metrics["interaction_count"] = max(20, strength * 5)
                        
                        # If mention counts are still 0, estimate based on relationship strength and interaction count
                        if not metrics.get("mention_together_count", 0):
                            # Use a portion of interaction count for mentions
                            metrics["mention_together_count"] = max(1, int(metrics["interaction_count"] * 0.7))
                        
                        # Ensure scene count and dialogue count have at least minimal values
                        if not metrics.get("scene_count", 0):
                            metrics["scene_count"] = max(1, int(metrics["interaction_count"] / 3))
                        
                        if not metrics.get("dialogue_count", 0):
                            metrics["dialogue_count"] = max(1, int(metrics["interaction_count"] / 2))
                        
                        # Update metrics in the relationship data
                        rel_data["metrics"] = metrics
                        
                        # Create Relationship object with updated data
                        relationships.append(Relationship(**rel_data))
                        
                        # Log the interaction counts
                        logger.debug(f"Relationship {rel_data['character1']} - {rel_data['character2']}: "
                                    f"interactions={metrics['interaction_count']}, "
                                    f"mentions={metrics['mention_together_count']}")
                    
                    return relationships
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in response: {json_str}")
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error parsing relationship extraction result: {str(e)}")
            logger.debug(f"Raw response: {response if isinstance(response, str) else 'non-string response'}")
            # Return empty list as fallback
            return []

    async def process_characters_batch(
        self, 
        title: str, 
        author: str, 
        characters: List[str], 
        character_excerpts: Dict[str, List[str]],
        verified_characters: List[Dict] = None
    ) -> List[Character]:
        """Process a batch of characters in parallel.

        Args:
            title: Book title
            author: Book author
            characters: List of character names
            character_excerpts: Dictionary mapping character names to their excerpts
            verified_characters: Optional list of verified character data for excerpt limits

        Returns:
            List of processed characters
        """
        tasks = []
        for char_name in characters:
            excerpts = character_excerpts.get(char_name, [])
            tasks.append(
                self.extract_character_details(title, author, char_name, excerpts, verified_characters)
            )
        
        # Process in parallel with limit
        semaphore = asyncio.Semaphore(self.processing_config.parallel_requests)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        results = await tqdm_asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            desc="Processing characters"
        )
        
        return results

    def _compute_character_co_occurrences(
        self, 
        character_names: List[str], 
        character_excerpts: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, int]]:
        """Compute character co-occurrences in excerpts to estimate interaction counts.
        
        Args:
            character_names: List of character names to analyze
            character_excerpts: Dictionary mapping character names to their excerpts
            
        Returns:
            Dictionary mapping character pairs to co-occurrence counts
        """
        logger.info("Computing character co-occurrence metrics")
        
        # Initialize co-occurrence dictionary
        co_occurrences = {}
        for char1 in character_names:
            co_occurrences[char1] = {}
            for char2 in character_names:
                if char1 != char2:
                    co_occurrences[char1][char2] = 0
        
        # For each character, check all their excerpts for mentions of other characters
        for char1 in character_names:
            excerpts = character_excerpts.get(char1, [])
            for excerpt in excerpts:
                # Count other characters mentioned in this excerpt
                for char2 in character_names:
                    if char1 != char2 and char2 in excerpt:
                        co_occurrences[char1][char2] += 1
                        co_occurrences[char2][char1] += 1  # Symmetric update
        
        # Log the top co-occurrences for debugging
        all_pairs = []
        for char1 in co_occurrences:
            for char2, count in co_occurrences[char1].items():
                if count > 0 and char1 < char2:  # Avoid counting pairs twice
                    all_pairs.append((char1, char2, count))
        
        # Sort by count descending
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Log top pairs if any
        if all_pairs:
            logger.info("Top character co-occurrences:")
            for char1, char2, count in all_pairs[:10]:  # Show top 10
                logger.info(f"  {char1} - {char2}: {count} co-occurrences")
                
        return co_occurrences

    async def process_all_characters(
        self,
        title: str,
        author: str,
        verified_characters: List[Dict[str, Any]],
        character_excerpts: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Process all verified characters to extract details.

        Args:
            title: Book title
            author: Book author
            verified_characters: List of verified character data
            character_excerpts: Dictionary mapping character names to their excerpts

        Returns:
            Dictionary with processed character data and relationships
        """
        # Extract character names handling both dict and VerifiedCharacter objects
        character_names = []
        character_types = {}  # Track character types for excerpt limits
        
        # First pass: extract names and character types
        for char in verified_characters:
            if isinstance(char, dict):
                name = char.get("name", "")
                if name:
                    character_names.append(name)
                    character_types[name] = {
                        "type": char.get("character_type", "minor"),
                        "is_major": char.get("is_major_character", False)
                    }
            elif hasattr(char, "name"):
                name = char.name
                if name:
                    character_names.append(name)
                    character_types[name] = {
                        "type": getattr(char, "character_type", "minor"),
                        "is_major": getattr(char, "is_major_character", False)
                    }
            elif hasattr(char, "choices"):
                try:
                    content = char.choices[0].message.content
                    char_data = json.loads(content)
                    name = char_data.get("name", "")
                    if name:
                        character_names.append(name)
                        character_types[name] = {
                            "type": char_data.get("character_type", "minor"),
                            "is_major": char_data.get("is_major_character", False)
                        }
                except Exception as e:
                    logger.error(f"Error processing character name: {str(e)}")
        
        # Remove empty names
        character_names = [name for name in character_names if name]
        
        # Calculate metrics: track mention counts for each character
        for char_name in character_names:
            excerpts = character_excerpts.get(char_name, [])
            if self.processing_config.track_metrics:
                # Count explicit mentions of the character in all excerpts
                mention_count = sum(1 for excerpt in excerpts if char_name in excerpt)
                logger.info(f"Character '{char_name}' mentioned {mention_count} times")
                
                # Store this information in the verified characters for later use
                for char in verified_characters:
                    if isinstance(char, dict) and char.get("name") == char_name:
                        if "metrics" not in char:
                            char["metrics"] = {}
                        char["metrics"]["mention_count"] = mention_count
                        break
                    elif hasattr(char, "name") and char.name == char_name:
                        if not hasattr(char, "metrics"):
                            setattr(char, "metrics", {})
                        char.metrics["mention_count"] = mention_count
                        break
        
        # Pre-compute character co-occurrences to estimate interaction counts
        character_co_occurrences = self._compute_character_co_occurrences(
            character_names, character_excerpts
        )
        
        # Process in batches
        batch_size = self.processing_config.batch_size
        characters = []
        
        for i in range(0, len(character_names), batch_size):
            batch = character_names[i:i + batch_size]
            logger.info(f"Processing character batch {i//batch_size + 1} ({len(batch)} characters)")
            
            batch_results = await self.process_characters_batch(
                title, author, batch, character_excerpts, verified_characters
            )
            characters.extend(batch_results)
        
        # Extract relationships for either major characters only or all characters
        all_character_names = []
        major_character_names = []
        processed_characters = []
        
        for char in characters:
            # Handle different response formats
            char_dict = {}
            char_name = ""
            is_major = False
            
            if hasattr(char, "model_dump"):
                # It's a Pydantic model (v2)
                char_dict = char.model_dump()
                char_name = char.name
                is_major = char.is_major_character
            elif hasattr(char, "dict"):
                # It's a Pydantic model (v1)
                char_dict = char.dict()
                char_name = char.name
                is_major = char.is_major_character
            elif hasattr(char, "choices"):
                # It's a LiteLLM ModelResponse
                try:
                    content = char.choices[0].message.content
                    char_data = json.loads(content)
                    char_dict = char_data
                    char_name = char_data.get("name", "")
                    is_major = char_data.get("is_major_character", False)
                except Exception as e:
                    logger.error(f"Error processing character response: {str(e)}")
                    continue
            elif isinstance(char, dict):
                # It's already a dictionary
                char_dict = char
                char_name = char.get("name", "")
                is_major = char.get("is_major_character", False)
            
            # Skip if we couldn't determine a name
            if not char_name:
                continue
            
            # Initialize interaction counts if not present
            if "metrics" not in char_dict:
                char_dict["metrics"] = {}
            if "interaction_counts" not in char_dict["metrics"]:
                char_dict["metrics"]["interaction_counts"] = {}
                
            # Add interaction counts from co-occurrence analysis
            if char_name in character_co_occurrences:
                for other_char, count in character_co_occurrences[char_name].items():
                    if count > 0:
                        char_dict["metrics"]["interaction_counts"][other_char] = count
            
            processed_characters.append(char_dict)
            all_character_names.append(char_name)
            if is_major:
                major_character_names.append(char_name)
        
        # Determine which character set to use for relationships
        character_set_for_relationships = major_character_names
        if self.processing_config.all_character_relationships:
            character_set_for_relationships = all_character_names
            logger.info(f"Using all {len(all_character_names)} characters for relationship extraction")
        else:
            logger.info(f"Using {len(major_character_names)} major characters for relationship extraction")
            
        # Collect excerpts that mention at least two characters from our selected set
        interaction_excerpts = []
        for char_name, excerpts in character_excerpts.items():
            if char_name in character_set_for_relationships:
                for excerpt in excerpts:
                    # Check if the excerpt mentions at least one other relevant character
                    mentions_others = any(
                        other_char in excerpt
                        for other_char in character_set_for_relationships
                        if other_char != char_name
                    )
                    if mentions_others and excerpt not in interaction_excerpts:
                        interaction_excerpts.append(excerpt)
        
        relationships = []
        if len(character_set_for_relationships) >= 2 and interaction_excerpts:
            logger.info(f"Extracting relationships for {len(character_set_for_relationships)} characters using {len(interaction_excerpts)} interaction excerpts")
            relationships = await self.extract_relationships(
                title, author, character_set_for_relationships, interaction_excerpts
            )
            
        # Process relationships
        processed_relationships = []
        for rel in relationships:
            rel_dict = None
            
            if hasattr(rel, "model_dump"):
                # It's a Pydantic model (v2)
                rel_dict = rel.model_dump()
            elif hasattr(rel, "dict"):
                # It's a Pydantic model (v1)
                rel_dict = rel.dict()
            elif hasattr(rel, "choices"):
                try:
                    content = rel.choices[0].message.content
                    rel_dict = json.loads(content)
                except Exception as e:
                    logger.error(f"Error processing relationship response: {str(e)}")
                    continue
            elif isinstance(rel, dict):
                rel_dict = rel
            
            if rel_dict:
                # Update relationship metrics with co-occurrence data if not already set
                char1 = rel_dict.get("character1", "")
                char2 = rel_dict.get("character2", "")
                
                if char1 and char2:
                    # Ensure metrics exist
                    if "metrics" not in rel_dict:
                        rel_dict["metrics"] = {}
                    
                    metrics = rel_dict["metrics"]
                    
                    # If LLM didn't provide interaction count, use co-occurrence data
                    if not metrics.get("interaction_count", 0) and char1 in character_co_occurrences and char2 in character_co_occurrences[char1]:
                        co_occurrence = character_co_occurrences[char1][char2]
                        if co_occurrence > 0:
                            # Use co-occurrence as a base, but adjust it up for major relationships
                            strength = rel_dict.get("strength", 5)
                            # For strong relationships, multiply co-occurrence 
                            factor = 1.0
                            if strength >= 8:
                                factor = 2.0
                            elif strength >= 5:
                                factor = 1.5
                                
                            metrics["interaction_count"] = max(1, int(co_occurrence * factor))
                            # Also set mention count based on co-occurrence
                            if not metrics.get("mention_together_count", 0):
                                metrics["mention_together_count"] = co_occurrence
                
                processed_relationships.append(rel_dict)
        
        return {
            "characters": processed_characters,
            "relationships": processed_relationships,
        }