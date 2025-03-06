"""Entity extraction module using spaCy and LLM."""

import logging
import re
from typing import Dict, List, Set, Optional

import spacy
from spacy.tokens import Doc, Span
from tqdm import tqdm

from book_cartographer.config import ProcessingConfig
from book_cartographer.nlp.llm_processor import LLMProcessor
from book_cartographer.nlp.prompts import LLM_ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Entity extractor using spaCy or LLM for NLP processing."""

    def __init__(self, config: ProcessingConfig, llm_processor: Optional[LLMProcessor] = None):
        """Initialize the entity extractor.

        Args:
            config: Processing configuration
            llm_processor: Optional LLM processor for experimental direct LLM extraction
        """
        self.config = config
        self.llm_processor = llm_processor
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        """Load the spaCy model."""
        # Skip if we're only using LLM-based extraction
        if self.config.use_llm_for_nlp and self.llm_processor:
            logger.info("Using LLM for entity extraction; spaCy model will not be loaded")
            return
            
        try:
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
        except OSError:
            logger.warning(f"spaCy model {self.config.spacy_model} not found. Downloading...")
            spacy.cli.download(self.config.spacy_model)
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info(f"Downloaded and loaded spaCy model: {self.config.spacy_model}")

    def process_text(self, text: str) -> Doc:
        """Process text with spaCy.

        Args:
            text: Text to process

        Returns:
            Processed spaCy Doc
        """
        return self.nlp(text)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        max_chunk_size = self.config.max_chunk_size
        
        # Simple chunking by paragraphs first
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def process_large_text(self, text: str) -> List[Doc]:
        """Process large text by chunking it.

        Args:
            text: Large text to process

        Returns:
            List of processed spaCy Docs
        """
        chunks = self._chunk_text(text)
        docs = []
        
        for chunk in tqdm(chunks, desc="Processing text chunks"):
            docs.append(self.process_text(chunk))
        
        return docs

    def extract_named_entities(self, docs: List[Doc]) -> Dict[str, List[Span]]:
        """Extract named entities from processed spaCy Docs.

        Args:
            docs: List of processed spaCy Docs

        Returns:
            Dictionary mapping entity types to lists of entity spans
        """
        entities: Dict[str, List[Span]] = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical Entity
            "LOC": [],  # Location
            "DATE": [],
            "TIME": [],
            "MISC": [],  # Miscellaneous
        }
        
        for doc in docs:
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent)
                else:
                    entities["MISC"].append(ent)
        
        return entities

    def extract_characters(self, docs: List[Doc]) -> Set[str]:
        """Extract character names from processed spaCy Docs.

        Args:
            docs: List of processed spaCy Docs

        Returns:
            Set of character names
        """
        entities = self.extract_named_entities(docs)
        
        # Focus on PERSON entities for characters
        characters = set()
        for ent in entities["PERSON"]:
            characters.add(ent.text)
        
        logger.info(f"Extracted {len(characters)} potential character names")
        return characters

    def extract_locations(self, docs: List[Doc]) -> Set[str]:
        """Extract location names from processed spaCy Docs.

        Args:
            docs: List of processed spaCy Docs

        Returns:
            Set of location names
        """
        entities = self.extract_named_entities(docs)
        
        locations = set()
        for ent in entities["GPE"] + entities["LOC"]:
            locations.add(ent.text)
        
        logger.info(f"Extracted {len(locations)} potential location names")
        return locations

    def extract_organizations(self, docs: List[Doc]) -> Set[str]:
        """Extract organization names from processed spaCy Docs.

        Args:
            docs: List of processed spaCy Docs

        Returns:
            Set of organization names
        """
        entities = self.extract_named_entities(docs)
        
        organizations = set()
        for ent in entities["ORG"]:
            organizations.add(ent.text)
        
        logger.info(f"Extracted {len(organizations)} potential organization names")
        return organizations

    def extract_temporal_references(self, docs: List[Doc]) -> Set[str]:
        """Extract temporal references from processed spaCy Docs.

        Args:
            docs: List of processed spaCy Docs

        Returns:
            Set of temporal references
        """
        entities = self.extract_named_entities(docs)
        
        temporal = set()
        for ent in entities["DATE"] + entities["TIME"]:
            temporal.add(ent.text)
        
        logger.info(f"Extracted {len(temporal)} potential temporal references")
        return temporal

    async def extract_all_entities(
        self, text: str, title: str = "", author: str = ""
    ) -> Dict[str, Set[str]]:
        """Extract all entity types from text.

        Args:
            text: Text to process
            title: Book title (used for LLM extraction)
            author: Book author (used for LLM extraction)

        Returns:
            Dictionary with entities by type
        """
        # Check if we should use LLM for extraction
        if self.config.use_llm_for_nlp and self.llm_processor:
            return await self._extract_entities_with_llm(text, title, author)
            
        docs = self.process_large_text(text)
        
        entities = {
            "characters": self.extract_characters(docs),
            "locations": self.extract_locations(docs),
            "organizations": self.extract_organizations(docs),
            "temporal": self.extract_temporal_references(docs),
        }
        
        logger.info(f"Extracted entities: {sum(len(v) for v in entities.values())} total")
        return entities
        
    async def _extract_entities_with_llm(self, text: str, title: str, author: str) -> Dict[str, Set[str]]:
        """Extract entities using LLM.
        
        Args:
            text: Text to extract entities from
            title: Book title
            author: Book author
            
        Returns:
            Dictionary of entity types to sets of entity names
        """
        from book_cartographer.nlp.prompts import generate_llm_entity_extraction_prompt
        import json
        import time
        
        logger.info("Extracting entities using LLM with optimized batch processing")
        
        # For Gemini with 1M token context window, we can process much larger chunks
        # Estimate ~4 characters per token, so ~250K characters per chunk should be safe
        # Check if we're using Gemini model which has larger context
        is_gemini = "gemini" in self.llm_processor.llm_config.provider.lower() or "gemini" in self.llm_processor.llm_config.model.lower()
        
        # Use larger chunks for Gemini models, smaller for others
        max_chunk_size = 250000 if is_gemini else 4000
        logger.info(f"Using {'Gemini-optimized' if is_gemini else 'standard'} processing with {max_chunk_size} character chunks")
        
        # Process in chunks
        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i:i+max_chunk_size])
            
        logger.info(f"Split text into {len(chunks)} chunks for LLM processing")
        
        # Process each chunk with LLM
        all_characters = set()
        
        start_time = time.time()
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with LLM ({len(chunk)} characters)")
            prompt = generate_llm_entity_extraction_prompt(title, author, chunk)
            
            try:
                response = await self.llm_processor._call_llm_with_schema(prompt)
                
                # Parse the response
                if isinstance(response, str):
                    text_response = response
                elif hasattr(response, "choices"):
                    text_response = response.choices[0].message.content
                else:
                    text_response = str(response)
                
                # Extract JSON from response
                json_match = re.search(r'({.*})', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        data = json.loads(json_str)
                        characters = data.get("characters", [])
                        new_chars = len(characters)
                        for char in characters:
                            all_characters.add(char)
                        chunk_time = time.time() - chunk_start
                        logger.info(f"Chunk {i+1} processed in {chunk_time:.2f}s, found {new_chars} characters")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from LLM response: {json_str}")
                else:
                    logger.error(f"No JSON found in LLM response: {text_response}")
            except Exception as e:
                logger.error(f"Error extracting entities with LLM: {str(e)}")
        
        total_time = time.time() - start_time
        logger.info(f"LLM entity extraction completed in {total_time:.2f}s")
        
        # Create output in same format as spaCy extraction
        entities = {
            "characters": all_characters,
            "locations": set(),  # LLM extraction only focuses on characters
            "organizations": set(),
            "temporal": set(),
        }
        
        logger.info(f"LLM extracted {len(entities['characters'])} potential character names")
        
        return entities