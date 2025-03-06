# Makefile for generating Mermaid diagrams

# Tools
MERMAID_CLI = npx @mermaid-js/mermaid-cli@latest
SVGEXPORT = npx svgexport

# Files
MERMAID_FILES = epub_character_graph_pipeline.mmd series_pipeline.mmd
SVG_FILES = $(MERMAID_FILES:.mmd=.svg)
PNG_FILES = $(MERMAID_FILES:.mmd=.png)

# Default target
all: $(PNG_FILES)

# Convert Mermaid files to SVG
%.svg: %.mmd
	$(MERMAID_CLI) -i $< -o $@

# Convert SVG to PNG
%.png: %.svg
	$(SVGEXPORT) $< $@

# Clean up generated files
clean:
	rm -f $(SVG_FILES) $(PNG_FILES)
