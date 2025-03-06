"""LLM prompt templates for character extraction and analysis."""

from typing import List, Set

# LLM direct entity extraction prompt 
LLM_ENTITY_EXTRACTION_PROMPT = """
You are an expert literary analyst specializing in character identification. I need you to extract all character names 
from the following text excerpt from a book. Include major and minor characters.

Book Title: {title}
Author: {author}

Text excerpt:
{text_excerpt}

Please identify all character names mentioned in this text. Consider the following guidelines:
1. Include both major and minor characters
2. Include full names and nicknames
3. Include titles when they're used as part of the name (e.g., "Dr. Smith")
4. Don't include generic references like "the doctor" unless it's clear this is used as a character identifier
5. For animals or non-human characters that are named, include them as well

IMPORTANT: You MUST provide your response as a properly structured JSON object with the following schema:
{{
  "characters": [
    "Character Name 1",
    "Character Name 2",
    ...
  ]
}}

Do not include explanations or text outside the JSON structure. Return only valid JSON.
"""

# Character extraction and verification prompt
CHARACTER_VERIFICATION_PROMPT = """
You are an expert literary analyst. I have extracted potential character names from a book using NLP. 
Please analyze this list and help identify which entries are actual character names in the narrative.

Book Title: {title}
Author: {author}

Potential character names extracted:
{character_list}

For each name above:
1. Determine if it is a character in the book
2. If it is a character, provide a confidence level (high/medium/low)
3. For similar names or variations that likely refer to the same character, group them together
4. Identify any major characters that might be missing from this list based on your knowledge of the book

IMPORTANT GUIDANCE ON CHARACTER TYPES:
- Major characters should include protagonists, primary antagonists, and key supporting characters who drive the plot
- Supporting characters have recurring roles but don't significantly impact the main plot
- Minor characters appear briefly or have minimal impact on the story
- A major character typically has significant dialogue, influence on the plot, or character development
- In most novels, there are typically 3-7 major characters, 5-15 supporting characters, and many minor characters
- Characters that appear in multiple scenes and have names should at least be considered supporting characters
- Most named pets or animal companions of major characters should also be classified based on their story impact
- Side characters with minimal plot impact should be marked as minor

IMPORTANT: You MUST provide your response as a properly structured JSON object with the following schema:
{{
  "verified_characters": [
    {{
      "name": "Character's canonical name",
      "alternative_names": ["variation1", "variation2"],
      "confidence": "high|medium|low",
      "character_type": "major|supporting|minor",
      "is_major_character": true|false,
      "archetype": "hero|mentor|ally|trickster|guardian|shadow|etc"
    }}
  ],
  "rejected_entries": ["name1", "name2"],
  "missing_major_characters": ["name1", "name2"]
}}

Ensure all verified characters have the "confidence" field as "high", "medium", or "low".
Do not include explanations or text outside the JSON structure. Return only valid JSON.
"""

# Character description extraction prompt
CHARACTER_DESCRIPTION_PROMPT = """
You are an expert literary analyst specializing in character analysis. I need you to extract comprehensive, detailed information about a character from a book.

Book Title: {title}
Author: {author}
Character Name: {character_name}

Here are text excerpts where this character appears:
{character_excerpts}

Please analyze these excerpts and extract the following information about {character_name}:

1. BASIC CHARACTERIZATION
   - Physical appearance (age, height, build, hair, eyes, distinctive features, clothing style)
   - How physical appearance changes throughout the story (if applicable)
   - Personality traits (temperament, habits, values, strengths, flaws)
   - Background (origin, family, education, occupation, social status)
   - Character type (major, supporting, or minor)
   - Archetype (e.g., hero, mentor, ally, trickster, guardian, shadow)
   - Role in the story (protagonist, antagonist, supporting character)

2. NARRATIVE ELEMENTS
   - Key actions or events involving this character
   - Character development or arc through the story
   - Locations/settings associated with this character
   - Dialogue style and speaking patterns
   - Core motivations and goals
   - Internal conflicts and struggles

3. EMOTIONAL DIMENSION
   - Primary emotional states at different points in the narrative
   - How emotions evolve throughout the story
   - Key emotional moments and their impact

4. CHARACTER EVOLUTION
   - Initial presentation (how they first appear)
   - Middle development (how they change/grow)
   - Final state (how they end up)
   - Key moments that define their evolution
   
5. RELATIONSHIPS
   - Connections to other characters
   - How relationships evolve over time
   - Power dynamics within relationships

IMPORTANT: You MUST provide your response as a properly structured JSON object with the following schema:
{{
  "name": "{character_name}",
  "alternative_names": [],
  "confidence": "high|medium|low",
  "character_type": "major|supporting|minor", 
  "is_major_character": true|false,
  "physical_appearance": {{
    "age": "",
    "physical_features": "",
    "clothing": ""
  }},
  "physical_appearance_evolution": [
    {{
      "position": "early|middle|late",
      "description": ""
    }}
  ],
  "personality": ["trait1", "trait2"],
  "background": "",
  "story_role": "",
  "archetype": "",
  "relationships": [
    {{
      "character": "name",
      "relationship_type": "friend/enemy/family/etc",
      "description": ""
    }}
  ],
  "key_actions": ["action1", "action2"],
  "character_arc": "",
  "dialogue_style": "",
  "motivations": ["motivation1", "motivation2"],
  "emotional_state": {{
    "early": ["emotion1", "emotion2"],
    "middle": ["emotion1", "emotion2"],
    "late": ["emotion1", "emotion2"]
  }},
  "internal_conflicts": ["conflict1", "conflict2"],
  "locations": ["location1", "location2"],
  "evolution": {{
    "early_presentation": "",
    "middle_development": "",
    "final_state": "",
    "key_moments": [
      {{
        "position": "early|middle|late",
        "description": ""
      }}
    ]
  }},
  "image_generation_prompt": ""
}}

Make sure you include the "confidence" field with a value of "high", "medium", or "low".
The "background" field must be a string, not an object.
In the "image_generation_prompt" field, create a concise description (2-3 sentences) that could be used for AI image generation, focusing on physical appearance, distinctive features, clothing, and expression.
Do not include explanations or text outside the JSON structure. Return only valid JSON.
"""

# Relationship extraction prompt
RELATIONSHIP_EXTRACTION_PROMPT = """
You are an expert literary analyst specializing in character relationships. I need you to analyze the complex relationships between characters in a book.

Book Title: {title}
Author: {author}

Characters to analyze:
{character_list}

Here are text excerpts showing interactions between these characters:
{interaction_excerpts}

Please analyze these interactions and extract detailed information about the relationships between characters. Consider:

1. RELATIONSHIP BASICS
   - Relationship type (friend, enemy, family, lover, colleague, mentor, etc.)
   - Relationship strength (how important this relationship is to the characters and plot)
   - Key dynamics that define their interactions

2. INTERACTION METRICS (EXTREMELY IMPORTANT)
   - Estimate how many times these characters interact directly (interaction_count)
   - Count instances where both characters are mentioned together (mention_together_count)
   - Focus on quantifying the frequency of interactions, not just their quality
   - Base these counts on evidence from the provided excerpts
   - Use these metrics to reflect the importance of the relationship

3. EMOTIONAL DIMENSIONS
   - Emotional tone of the relationship (warm, tense, antagonistic, etc.)
   - Power balance (equal, one-sided, shifting)
   - Conflicts or points of tension

4. RELATIONSHIP EVOLUTION
   - How the relationship starts/is initially presented
   - How it develops through the middle of the story
   - Final state of the relationship
   - Key turning points or pivotal moments

5. INTERACTION PATTERNS
   - How they typically interact or communicate
   - Recurring patterns in their relationship
   - Frequency and significance of their interactions

IMPORTANT: You MUST provide your response as a properly structured JSON object with the following schema:
{{
  "relationships": [
    {{
      "character1": "Name1",
      "character2": "Name2",
      "relationship_type": "friend/enemy/family/lover/colleague/etc",
      "description": "Brief description of their relationship",
      "strength": 1-10,
      "evidence": "Brief quote or scene reference supporting this relationship",
      "dynamics": "Description of relationship dynamics",
      "conflicts": ["conflict1", "conflict2"],
      "balance": "Power balance description",
      "emotional_tone": "Emotional quality of relationship",
      "pivotal_moments": [
        {{
          "position": "early|middle|late",
          "description": "Description of key moment"
        }}
      ],
      "evolution": {{
        "early_state": "Initial relationship state",
        "middle_development": "How relationship develops",
        "final_state": "Final relationship state"
      }},
      "metrics": {{
        "interaction_count": 0,
        "mention_together_count": 0,
        "scene_count": 0,
        "dialogue_count": 0
      }}
    }}
  ]
}}

For the "strength" field, use a scale of 1-10 where:
1 = barely connected
5 = significant relationship
10 = central relationship to the story

CRITICAL REQUIREMENTS:
1. Ensure the "strength" field is an integer, not a string.
2. ALWAYS estimate numeric values for interaction_count and mention_together_count. NEVER leave these as 0.
3. Base these counts on evidence from the excerpts - count dialogues, shared scenes, and mentions.
4. For major character relationships with high strength (7-10), interaction_count should generally be higher (20+).
5. Use the provided excerpts to make informed estimates of interaction frequency.

Do not include explanations or text outside the JSON structure. Return only valid JSON.
"""

def generate_character_verification_prompt(
    title: str, 
    author: str, 
    characters: Set[str]
) -> str:
    """Generate a prompt for character verification.
    
    Args:
        title: Book title
        author: Book author
        characters: Set of extracted character names
        
    Returns:
        Formatted prompt string
    """
    character_list = "\n".join([f"- {char}" for char in sorted(characters)])
    
    return CHARACTER_VERIFICATION_PROMPT.format(
        title=title,
        author=author,
        character_list=character_list
    )

def generate_character_description_prompt(
    title: str,
    author: str,
    character_name: str,
    excerpts: List[str]
) -> str:
    """Generate a prompt for character description extraction.
    
    Args:
        title: Book title
        author: Book author
        character_name: Name of the character to analyze
        excerpts: List of text excerpts featuring the character
        
    Returns:
        Formatted prompt string
    """
    # Join excerpts with separator for clarity
    formatted_excerpts = "\n---\n".join(excerpts)
    
    return CHARACTER_DESCRIPTION_PROMPT.format(
        title=title,
        author=author,
        character_name=character_name,
        character_excerpts=formatted_excerpts
    )

def generate_relationship_extraction_prompt(
    title: str,
    author: str,
    characters: List[str],
    excerpts: List[str]
) -> str:
    """Generate a prompt for relationship extraction.
    
    Args:
        title: Book title
        author: Book author
        characters: List of character names to analyze
        excerpts: List of text excerpts showing character interactions
        
    Returns:
        Formatted prompt string
    """
    character_list = "\n".join([f"- {char}" for char in characters])
    formatted_excerpts = "\n---\n".join(excerpts)
    
    return RELATIONSHIP_EXTRACTION_PROMPT.format(
        title=title,
        author=author,
        character_list=character_list,
        interaction_excerpts=formatted_excerpts
    )
    
def generate_llm_entity_extraction_prompt(
    title: str,
    author: str,
    text_excerpt: str
) -> str:
    """Generate a prompt for LLM-based entity extraction.
    
    Args:
        title: Book title
        author: Book author
        text_excerpt: Text excerpt to analyze
        
    Returns:
        Formatted prompt string
    """
    return LLM_ENTITY_EXTRACTION_PROMPT.format(
        title=title,
        author=author,
        text_excerpt=text_excerpt
    )