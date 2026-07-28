# turkish-rag-assistant

A Turkish-language RAG (Retrieval-Augmented Generation) assistant that aims to
answer questions over document collections. This repository is an early-stage,
in-progress build with a focus on clean engineering and testing.

## Why Turkish?

Most RAG examples target English. Turkish is agglutinative, with rich morphology
and its own tokenization and normalization quirks, so a pipeline tuned for it
would behave differently than a generic English one. This project aims to tune
for those differences rather than treat Turkish as an afterthought — the
motivation and differentiator behind the project.

## Status

Early development. The data ingestion layer is being implemented first; the
retrieval and generation stages are planned (see Roadmap).

## Data layer

The ingestion pipeline turns raw text into clean, searchable chunks:

- **stats** — compute basic text statistics (word, character, line counts)
- **cleaner** — normalize whitespace, including Unicode whitespace commonly
  found in text copied from PDFs or web pages
- **chunker** — split cleaned text into fixed-size, overlapping chunks

## Roadmap

- [x] Data layer: text stats, cleaning, chunking
- [ ] Turkish-aware normalization and tokenization
- [ ] Embeddings and vector store
- [ ] Retrieval
- [ ] LLM answer generation
- [ ] Retrieval and answer evaluation

## Tech stack

- Python 3.12
- uv (packaging and environment management)
- pytest (testing)
- ruff (linting and formatting)

## Development

Install dependencies:

    uv sync

Run the tests:

    uv run pytest

Check linting and formatting:

    uv run ruff check
    uv run ruff format --check