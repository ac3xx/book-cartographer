"""Command-line interface for the EPUB character graph generator."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from book_cartographer.character_graph import CharacterGraph
from book_cartographer.config import (
    AppConfig,
    get_config,
)
from book_cartographer.epub_parser import EPUBParser
from book_cartographer.nlp.entity_extractor import EntityExtractor
from book_cartographer.nlp.llm_processor import LLMProcessor
from book_cartographer.output_generators.base import OutputGeneratorConfig
from book_cartographer.output_generators.image_prompt import ImagePromptGenerator
from book_cartographer.output_generators.kodict import KODictGenerator
from book_cartographer.utils import (
    extract_character_excerpts,
    sanitize_filename,
    setup_logging,
)

console = Console()
logger = logging.getLogger(__name__)


async def process_epub(
    epub_path: Path,
    config: AppConfig,
    output_file: Optional[Path] = None,
) -> None:
    """Process an EPUB file to extract character graph.

    Args:
        epub_path: Path to the EPUB file
        config: Application configuration
        output_file: Optional output file path override
    """
    # Parse the EPUB file
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description=f"Parsing EPUB file: {epub_path.name}", total=None)
        parser = EPUBParser(epub_path)
        metadata = parser.get_metadata()
        full_text = parser.get_full_text()

    # Setup LLM processor
    llm_processor = LLMProcessor(config.llm, config.processing)
    
    # Setup entity extractor
    entity_extractor = EntityExtractor(config.processing, llm_processor)

    # Extract initial entities
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        description = "Extracting entities with spaCy"
        if config.processing.use_llm_for_nlp:
            description = "Extracting entities with LLM"
            
        progress.add_task(description=description, total=None)
        
        if not config.processing.use_llm_for_nlp:
            docs = entity_extractor.process_large_text(full_text)
            entities = await entity_extractor.extract_all_entities(full_text)
        else:
            entities = await entity_extractor.extract_all_entities(
                full_text, metadata["title"], metadata["author"]
            )

    # Verify character names
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description="Verifying characters with LLM", total=None
        )
        verified_result = await llm_processor.verify_characters(
            metadata["title"], metadata["author"], entities["characters"]
        )

    # Extract excerpts for each character
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description="Extracting character excerpts", total=None
        )
        # Handle both Pydantic model and LiteLLM response formats
        if hasattr(verified_result, "verified_characters"):
            # It's a Pydantic model
            verified_characters = verified_result.verified_characters
        else:
            # It's a LiteLLM response
            import json
            try:
                if hasattr(verified_result, "choices"):
                    # Parse from message content
                    content = verified_result.choices[0].message.content
                    verified_data = json.loads(content)
                    verified_characters = verified_data.get("verified_characters", [])
                else:
                    # It might be directly the model response
                    verified_characters = verified_result.get("verified_characters", [])
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                logger.debug(f"Raw response: {verified_result}")
                verified_characters = []
        
        # Extract character names
        verified_names = []
        for char in verified_characters:
            if isinstance(char, dict):
                name = char.get("name")
                if name:
                    verified_names.append(name)
            elif hasattr(char, "name"):
                verified_names.append(char.name)
        
        # Extract character excerpts with metrics tracking if enabled
        character_excerpts = extract_character_excerpts(
            full_text, 
            set(verified_names),
            track_metrics=config.processing.track_metrics
        )
        
        # Store intermediate outputs if enabled
        if config.processing.store_intermediate_outputs:
            # Create directory if it doesn't exist
            interim_dir = config.interim_data_dir
            interim_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a sanitized filename from the title
            safe_filename = sanitize_filename(metadata["title"])
            
            # Save character excerpts
            output_file = interim_dir / f"{safe_filename}_character_excerpts.json"
            with open(output_file, 'w') as f:
                # Create a copy that can be serialized (remove _metrics special key if present)
                serializable_excerpts = character_excerpts.copy()
                if "_metrics" in serializable_excerpts:
                    metrics_data = serializable_excerpts.pop("_metrics")
                    # Save metrics separately
                    metrics_file = interim_dir / f"{safe_filename}_excerpt_metrics.json"
                    with open(metrics_file, 'w') as mf:
                        import json as json_module  # Local import to avoid conflicts
                        json_module.dump(metrics_data, mf, indent=2)
                
                import json as json_module
                json_module.dump(serializable_excerpts, f, indent=2)
            
            # Save verified characters
            verified_file = interim_dir / f"{safe_filename}_verified_characters.json"
            with open(verified_file, 'w') as f:
                # Convert to serializable format
                serializable_chars = []
                for char in verified_characters:
                    if isinstance(char, dict):
                        serializable_chars.append(char)
                    elif hasattr(char, "dict"):
                        serializable_chars.append(char.dict())
                    else:
                        # Best effort serialization
                        serializable_chars.append({"name": str(char)})
                
                import json as json_module
                json_module.dump({"verified_characters": serializable_chars}, f, indent=2)
            
            console.print(f"[green]Saved intermediate outputs to {interim_dir}[/green]")
        
        # If metrics were tracked, update character data with the metrics
        if config.processing.track_metrics and "_metrics" in character_excerpts:
            metrics = character_excerpts.pop("_metrics")  # Remove from excerpts dict
            
            # Update verified characters with metrics
            for char in verified_characters:
                if isinstance(char, dict):
                    name = char.get("name")
                    if name and name in metrics["mention_counts"]:
                        if "metrics" not in char:
                            char["metrics"] = {}
                        char["metrics"]["mention_count"] = metrics["mention_counts"][name]
                        char["metrics"]["dialogue_count"] = metrics["dialogue_counts"][name]
                        char["metrics"]["interaction_counts"] = metrics["interactions"][name]
                elif hasattr(char, "name") and char.name in metrics["mention_counts"]:
                    if not hasattr(char, "metrics"):
                        setattr(char, "metrics", {})
                    char.metrics["mention_count"] = metrics["mention_counts"][char.name]
                    char.metrics["dialogue_count"] = metrics["dialogue_counts"][char.name]
                    char.metrics["interaction_counts"] = metrics["interactions"][char.name]

    # Process all characters with LLM
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description="Processing characters with LLM", total=None
        )
        # Use the same verified_characters from above
        character_data = await llm_processor.process_all_characters(
            metadata["title"],
            metadata["author"],
            verified_characters,
            character_excerpts,
        )

    # Build character graph
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description="Building character graph", total=None
        )
        graph = CharacterGraph(metadata["title"], metadata["author"])
        graph.build_from_data(character_data)

    # Generate output
    output_config = OutputGeneratorConfig(
        output_file=output_file or config.output.output_file,
        template_dir=config.output.template_dir,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(
            description=f"Generating {config.output.mode} output", total=None
        )
        
        if config.output.mode == "image-prompts":
            generator = ImagePromptGenerator(output_config, graph)
            result = generator.run()
        elif config.output.mode == "kodict":
            generator = KODictGenerator(output_config, graph)
            result = generator.run()
        else:
            logger.error(f"Unknown output mode: {config.output.mode}")
            sys.exit(1)
            
    # Save graph for future use
    graph_file = output_file.parent / f"{sanitize_filename(metadata['title'])}_graph.json" if output_file else Path(f"{sanitize_filename(metadata['title'])}_graph.json")
    graph.save_to_file(graph_file)
    
    # Store additional intermediate outputs if enabled
    if config.processing.store_intermediate_outputs:
        # Save the processed data (final character and relationship data)
        interim_dir = config.interim_data_dir
        interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sanitized filename from the title
        safe_filename = sanitize_filename(metadata["title"])
        
        # Save character details
        characters_file = interim_dir / f"{safe_filename}_processed_characters.json"
        with open(characters_file, 'w') as f:
            import json as json_module
            json_module.dump(graph.get_all_characters(), f, indent=2)
        
        # Save relationship details
        relationships_file = interim_dir / f"{safe_filename}_processed_relationships.json"
        with open(relationships_file, 'w') as f:
            import json as json_module
            json_module.dump(graph.get_all_relationships(), f, indent=2)
            
        # Save centrality metrics
        central_chars = graph.get_central_characters(n=len(graph.get_all_characters()))
        metrics_file = interim_dir / f"{safe_filename}_centrality_metrics.json"
        with open(metrics_file, 'w') as f:
            import json as json_module
            json_module.dump({char: score for char, score in central_chars}, f, indent=2)
    
    console.print("[green]Processing complete![/green]")
    console.print(f"Character graph saved to: {graph_file}")
    console.print(f"Output saved to: {output_config.output_file}")
    
    # Print some statistics
    console.print(f"\n[bold]Book:[/bold] {metadata['title']} by {metadata['author']}")
    console.print(f"[bold]Characters extracted:[/bold] {len(graph.get_all_characters())}")
    console.print(f"[bold]Major characters:[/bold] {len(graph.get_major_characters())}")
    console.print(f"[bold]Relationships:[/bold] {len(graph.get_all_relationships())}")
    
    # Print central characters
    central_chars = graph.get_central_characters()
    if central_chars:
        console.print("\n[bold]Central characters:[/bold]")
        for char, score in central_chars:
            console.print(f"- {char} (centrality: {score:.2f})")
            
    # Print metrics summary if metrics tracking was enabled
    if config.processing.track_metrics and config.processing.store_raw_metrics:
        console.print("\n[bold]Character Metrics:[/bold]")
        
        # Get the top 5 characters by mention count
        characters = graph.get_all_characters()
        mention_sorted = sorted(
            characters, 
            key=lambda c: c.get("metrics", {}).get("mention_count", 0),
            reverse=True
        )[:5]
        
        if mention_sorted:
            console.print("\nTop characters by mention count:")
            for char in mention_sorted:
                name = char.get("name", "Unknown")
                mentions = char.get("metrics", {}).get("mention_count", 0)
                char_type = char.get("character_type", "unknown")
                console.print(f"- {name} ({char_type}): {mentions} mentions")


@click.group()
def main():
    """BookCartographer: Map and analyze the worlds within EPUB files.
    
    Extract entities, relationships, and narrative elements from EPUB books.
    Generate specialized outputs for readers, writers, and researchers.
    """
    pass


@main.command("process")
@click.argument(
    "epub_file",
    type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
    "--config",
    "-c",
    type=click.Path(readable=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Path to output file",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["image-prompts", "kodict"]),
    help="Output mode",
)
@click.option(
    "--llm-provider",
    type=str,
    help="LLM provider to use",
)
@click.option(
    "--llm-model",
    type=str,
    help="LLM model to use",
)
@click.option(
    "--api-key",
    type=str,
    help="API key for LLM provider",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--all-relationships",
    is_flag=True,
    help="Extract relationships between all characters, not just major ones",
)
@click.option(
    "--use-llm-for-nlp",
    is_flag=True,
    help="Use LLM for initial NLP entity extraction",
)
@click.option(
    "--track-metrics/--no-track-metrics",
    default=None,
    help="Track additional metrics like mention counts and centrality scores",
)
@click.option(
    "--store-raw-metrics/--no-store-raw-metrics",
    default=None,
    help="Store raw metrics in output JSON",
)
@click.option(
    "--track-evolution/--no-track-evolution",
    default=None,
    help="Track character evolution throughout the narrative",
)
@click.option(
    "--comprehensive-analysis/--no-comprehensive-analysis",
    default=None,
    help="Perform comprehensive character analysis including dialogue, motivations, and emotions",
)
@click.option(
    "--store-intermediate/--no-store-intermediate",
    default=None,
    help="Store intermediate processing outputs like character excerpts and LLM responses",
)
@click.option(
    "--major-excerpts",
    type=int,
    help="Number of excerpts to include for major characters",
)
@click.option(
    "--supporting-excerpts",
    type=int,
    help="Number of excerpts to include for supporting characters",
)
@click.option(
    "--minor-excerpts",
    type=int,
    help="Number of excerpts to include for minor characters",
)
@click.option(
    "--relationship-excerpts",
    type=int,
    help="Number of excerpts to include for relationships",
)
@click.option(
    "--series-name",
    type=str,
    help="Name of the series this book belongs to",
)
@click.option(
    "--book-number",
    type=int,
    help="Book number in the series (e.g., 1 for first book)",
)
@click.option(
    "--add-to-series",
    is_flag=True,
    help="Add this book to an existing series tracking file",
)
def process_command(
    epub_file: Path,
    config: Optional[Path],
    output: Optional[Path],
    mode: Optional[str],
    llm_provider: Optional[str],
    llm_model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
    all_relationships: bool,
    use_llm_for_nlp: bool,
    track_metrics: Optional[bool],
    store_raw_metrics: Optional[bool],
    track_evolution: Optional[bool],
    comprehensive_analysis: Optional[bool],
    store_intermediate: Optional[bool],
    major_excerpts: Optional[int],
    supporting_excerpts: Optional[int],
    minor_excerpts: Optional[int],
    relationship_excerpts: Optional[int],
    series_name: Optional[str],
    book_number: Optional[int],
    add_to_series: bool,
) -> None:
    """Process EPUB files to extract entity data and generate outputs.
    
    EPUB_FILE is the path to the EPUB file to process.
    
    This command extracts characters, locations, organizations, and other entities
    from an EPUB file, builds relationship graphs, and generates outputs in the
    specified format.
    """
    # Setup logging
    setup_logging("DEBUG" if verbose else "INFO")
    
    # Load configuration
    app_config = get_config(config)
    
    # Override config with command-line options
    if mode:
        app_config.output.mode = mode
        
    if output:
        app_config.output.output_file = output
        
    if llm_provider:
        app_config.llm.provider = llm_provider
        
    if llm_model:
        app_config.llm.model = llm_model
        
    if api_key:
        app_config.llm.api_key = api_key
        
    # Set relationship and experimental options
    if all_relationships:
        app_config.processing.all_character_relationships = True
        
    if use_llm_for_nlp:
        app_config.processing.use_llm_for_nlp = True
        
    # Set advanced analysis options
    if track_metrics is not None:
        app_config.processing.track_metrics = track_metrics
        
    if store_raw_metrics is not None:
        app_config.processing.store_raw_metrics = store_raw_metrics
        
    if track_evolution is not None:
        app_config.processing.track_character_evolution = track_evolution
        
    if comprehensive_analysis is not None:
        app_config.processing.comprehensive_analysis = comprehensive_analysis
        
    if store_intermediate is not None:
        app_config.processing.store_intermediate_outputs = store_intermediate
        
    # Set excerpt limits
    if major_excerpts is not None:
        app_config.processing.excerpt_limits.major_character = major_excerpts
        
    if supporting_excerpts is not None:
        app_config.processing.excerpt_limits.supporting_character = supporting_excerpts
        
    if minor_excerpts is not None:
        app_config.processing.excerpt_limits.minor_character = minor_excerpts
        
    if relationship_excerpts is not None:
        app_config.processing.excerpt_limits.relationship = relationship_excerpts
    
    # Set series-related options
    if series_name:
        app_config.series.series_name = series_name
        app_config.series.enable_series_tracking = True
        
    if book_number is not None:
        app_config.series.enable_series_tracking = True
        
    if add_to_series:
        app_config.series.enable_series_tracking = True
        
    # Set default output file if not provided
    if not app_config.output.output_file and not output:
        suffix = ".txt" if app_config.output.mode == "image-prompts" else ".dict"
        output_filename = f"{epub_file.stem}_characters{suffix}"
        app_config.output.output_file = Path(output_filename)
    
    # Process the EPUB file
    try:
        asyncio.run(process_epub(epub_file, app_config, output))
    except Exception as e:
        logger.exception(f"Error processing EPUB: {str(e)}")
        console.print(f"[red]Error processing EPUB: {str(e)}[/red]")
        sys.exit(1)
        
        
@main.command("series-init")
@click.argument(
    "series_name",
    type=str,
)
@click.option(
    "--author",
    "-a",
    type=str,
    help="Author of the series",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Path to output series file",
)
def series_init_command(series_name: str, author: Optional[str], output: Optional[Path]) -> None:
    """Initialize a new series tracking file.
    
    SERIES_NAME is the name of the book series to initialize.
    """
    from book_cartographer.series_graph import SeriesGraph
    
    # Default output file if not provided
    if not output:
        safe_name = ''.join(c if c.isalnum() else '_' for c in series_name)
        output = Path(f"{safe_name}_series.json")
    
    # Check if file exists
    if output.exists():
        console.print(f"[yellow]Series file {output} already exists.[/yellow]")
        return
    
    # Create the series graph
    series_graph = SeriesGraph(series_name, author or "Unknown")
    
    try:
        # Save it to the specified location
        series_graph.save_to_file(output)
        console.print(f"[green]Series tracking initialized for '{series_name}'[/green]")
        console.print(f"File saved to: {output}")
        console.print("Add books to this series using the 'process' command with --add-to-series and --series-name options.")
        
    except Exception as e:
        console.print(f"[red]Error creating series file: {str(e)}[/red]")
        sys.exit(1)


@main.command("series-analyze")
@click.argument(
    "series_file",
    type=click.Path(exists=True, readable=True, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Path to output analysis file",
)
def series_analyze_command(series_file: Path, output: Optional[Path]) -> None:
    """Analyze character evolution across a book series.
    
    SERIES_FILE is the path to the series tracking file.
    """
    from book_cartographer.series_graph import SeriesGraph
    
    # Load the series graph
    try:
        series_graph = SeriesGraph.load_from_file(series_file)
    except Exception as e:
        console.print(f"[red]Error loading series file: {str(e)}[/red]")
        sys.exit(1)
    
    # Default output file if not provided
    if not output:
        output = series_file.with_suffix('.analysis.txt')
    
    # Get series data
    books = series_graph.get_books_by_reading_order()
    recurring_chars = series_graph.get_recurring_characters()
    evolving_rels = series_graph.get_evolving_relationships()
    
    # Create analysis report
    report = [
        f"# Series Analysis: {series_graph.series_name}",
        f"Author: {series_graph.author}",
        f"Books: {len(books)}",
        f"Total characters: {len(series_graph.characters)}",
        f"Recurring characters: {len(recurring_chars)}",
        f"Evolving relationships: {len(evolving_rels)}",
        "",
        "## Books in Reading Order",
    ]
    
    for book in books:
        report.append(f"- {book.title} (#{book.book_number})")
    
    report.append("")
    report.append("## Top Recurring Characters")
    
    # Sort by number of appearances
    top_chars = sorted(
        recurring_chars, 
        key=lambda c: len(c.appearances), 
        reverse=True
    )[:10]
    
    for char in top_chars:
        report.append(f"### {char.canonical_name}")
        report.append(f"- Appears in {len(char.appearances)} books")
        report.append(f"- First appearance: {char.first_appearance}")
        report.append(f"- Last appearance: {char.last_appearance}")
        report.append("- Character type by book:")
        
        for book, char_type in char.character_type_by_book.items():
            report.append(f"  - {book}: {char_type}")
        
        report.append("")
    
    # Write the report
    try:
        with open(output, 'w') as f:
            f.write('\n'.join(report))
        console.print(f"[green]Series analysis report saved to: {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing analysis report: {str(e)}[/red]")
        sys.exit(1)


@main.command("init-config")
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    default=Path("./config.toml"),
    help="Path to output configuration file",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing config file if it exists",
)
def init_config_command(output: Path, overwrite: bool) -> None:
    """Initialize a new configuration file with default settings.
    
    This command creates a new configuration file with all default settings.
    It can be used as a starting point for customizing BookCartographer's behavior.
    """
    if output.exists() and not overwrite:
        console.print(f"[yellow]Config file {output} already exists. Use --overwrite to replace it.[/yellow]")
        return
        
    # Create a default config
    config = AppConfig()
    
    try:
        # Save it to the specified location
        config.save_to_file(output)
        console.print(f"[green]Configuration file created at: {output}[/green]")
        console.print("Edit this file to customize the settings for character extraction.")
        
    except Exception as e:
        console.print(f"[red]Error creating config file: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()