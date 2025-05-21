# PDF-Text-Searcher-updated-

A cross-format desktop application for indexing and full-text search over PDF, text, CSV, Excel, JSON, and HTML files (and even web pages). Built with Tkinter GUI and Whoosh search engine, it normalizes and lemmatizes text via NLTK and presents fast, highlighted search results.

---

## Features

- **Multi-format indexing**: PDF, TXT, CSV, XLSX, JSON, HTML (local files and URLs)
- **Full-text search**: Whoosh-backed search with stemming/lemmatization support
- **Contextual snippets**: Highlighted excerpts around matched terms
- **GUI controls**:
  - Select directory to index
  - Re-index current folder
  - Index a single URL
  - Choose which file types to include
  - Progress bar and status messages
- **Smart deduplication**: Only one entry per file in results, even if multiple pages/rows match
- **Graceful fall-backs**: Basic lowercase filtering if NLTK resources aren’t installed

---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/omarfoud/PDF-Text-Searcher-updated-.git
   cd pdf-text-searcher
