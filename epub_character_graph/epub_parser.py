"""EPUB parsing module for extracting text content from EPUB files."""

import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

logger = logging.getLogger(__name__)


class EPUBParser:
    """Parser for EPUB files to extract text content."""

    def __init__(self, file_path: Path):
        """Initialize the parser with the EPUB file path.

        Args:
            file_path: Path to the EPUB file
        """
        self.file_path = file_path
        self.book = None
        self._load_book()

    def _load_book(self) -> None:
        """Load the EPUB book from the file path."""
        try:
            self.book = epub.read_epub(str(self.file_path))
            logger.info(f"Successfully loaded EPUB: {self.file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load EPUB: {self.file_path.name}")
            logger.error(f"Error: {str(e)}")
            raise

    def get_metadata(self) -> Dict[str, str]:
        """Extract metadata from the EPUB file.

        Returns:
            Dictionary with metadata (title, author, etc.)
        """
        if not self.book:
            raise ValueError("Book not loaded")

        metadata = {}
        
        # Extract basic metadata
        title = self.book.get_metadata('DC', 'title')
        metadata['title'] = title[0][0] if title else "Unknown"
        
        creator = self.book.get_metadata('DC', 'creator')
        metadata['author'] = creator[0][0] if creator else "Unknown"
        
        language = self.book.get_metadata('DC', 'language')
        metadata['language'] = language[0][0] if language else "Unknown"
        
        return metadata

    def extract_html_content(self) -> List[Tuple[str, str]]:
        """Extract HTML content from the EPUB file.

        Returns:
            List of tuples with (item_id, html_content)
        """
        if not self.book:
            raise ValueError("Book not loaded")

        html_content = []
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content.append((item.id, item.get_content().decode('utf-8')))
        
        logger.info(f"Extracted {len(html_content)} HTML documents from EPUB")
        return html_content

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content by removing tags and extracting text.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def extract_text_content(self) -> Generator[Tuple[str, str], None, None]:
        """Extract text content from the EPUB file.

        Yields:
            Tuples of (item_id, text_content)
        """
        html_contents = self.extract_html_content()
        
        for item_id, html in html_contents:
            text = self._clean_html(html)
            yield (item_id, text)

    def get_full_text(self) -> str:
        """Extract and return the full text content from the EPUB file.

        Returns:
            Full text content of the book
        """
        text_contents = []
        for _, text in self.extract_text_content():
            text_contents.append(text)
        
        return "\n\n".join(text_contents)
        
    def get_chapters(self) -> List[Tuple[str, str]]:
        """Extract chapters from the EPUB file.

        Returns:
            List of tuples with (chapter_title, chapter_content)
        """
        if not self.book:
            raise ValueError("Book not loaded")
            
        chapters = []
        toc = self.book.toc
        
        if not toc:
            # If no ToC, treat each HTML document as a chapter
            for item_id, text in self.extract_text_content():
                chapters.append((f"Chapter {len(chapters) + 1}", text))
        else:
            # Extract chapters based on ToC
            # This is simplified and may need adaptation for different EPUB structures
            for entry in toc:
                if isinstance(entry, tuple):
                    title, href = entry[0], entry[1]
                    # Find the corresponding content
                    for item in self.book.get_items():
                        if item.get_name() == href:
                            text = self._clean_html(item.get_content().decode('utf-8'))
                            chapters.append((title, text))
                            break
        
        logger.info(f"Extracted {len(chapters)} chapters from EPUB")
        return chapters