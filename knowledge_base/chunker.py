import re
import json
import nltk
import pdfplumber
from nltk.tokenize import sent_tokenize

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def _extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF using pdfplumber.
    Returns a single string with page breaks replaced by newlines.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def fixed_chunk(text, size=500, overlap=50):
    """Split *text* into fixed-size character chunks.
    Each chunk will have *size* characters, with an *overlap* of characters
    shared with the next chunk.
    Returns a list of string chunks.
    """
    if size <= overlap:
        raise ValueError('size must be larger than overlap')
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def sentence_chunk(text, max_sentences=5):
    """Chunk *text* by sentences.
    Uses NLTK's ``sent_tokenize`` to split the text into sentences, then groups
    them into windows containing up to ``max_sentences`` sentences each.
    Returns a list of string chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks


def section_chunk(text):
    """Section‑aware chunking for regulatory documents.
    Splits *text* on common heading patterns:
        - Numbered headings like "1. Title"
        - All‑caps headings of 4+ letters
        - Explicit "Section X" markers
    The heading line is kept as the first line of the following chunk.
    Returns a list of string chunks.
    """
    heading_regex = re.compile(r"\n(?:(\d+\.\s)|([A-Z]{4,})|(Section\s\d+))")
    parts = heading_regex.split(text)
    chunks = []
    i = 0
    while i < len(parts):
        pre_text = parts[i]
        heading = None
        for j in range(1, 4):
            if i + j < len(parts) and parts[i + j] is not None:
                heading = parts[i + j]
                break
        content_index = i + 4
        content = parts[content_index] if content_index < len(parts) else ""
        if heading:
            chunk = f"{heading.strip()}\n{content.strip()}"
        else:
            chunk = pre_text.strip()
        if chunk:
            chunks.append(chunk)
        i = content_index
    return chunks


def compare_strategies(pdf_path):
    """Run all three chunking strategies on the same PDF and report results.
    Prints the number of chunks, average chunk size, and a sample chunk for each
    strategy. Saves a JSON summary to ``knowledge_base/chunking_comparison.json``.
    """
    raw_text = _extract_text_from_pdf(pdf_path)
    fixed = fixed_chunk(raw_text)
    sentence = sentence_chunk(raw_text)
    section = section_chunk(raw_text)
    def stats(chunks):
        sizes = [len(c) for c in chunks]
        avg = sum(sizes) / len(sizes) if sizes else 0
        sample = chunks[0] if chunks else ''
        return {'count': len(chunks), 'avg_size': avg, 'sample': sample}
    comparison = {
        'fixed': stats(fixed),
        'sentence': stats(sentence),
        'section': stats(section)
    }
    print(f'Chunking comparison for: {pdf_path}')
    for name, data in comparison.items():
        print(f"{name.title()} – chunks: {data['count']}, avg size: {data['avg_size']:.1f}")
        print('Sample chunk (first 200 chars):')
        print(data['sample'][:200])
        print('---')
    out_path = 'knowledge_base/chunking_comparison.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f'Comparison saved to {out_path}')

# Example usage (uncomment for manual run):
# if __name__ == '__main__':
#     compare_strategies('knowledge_base/raw/Food_Additives_Regulations.pdf')
