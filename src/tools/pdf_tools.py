from pathlib import Path
import pdfplumber

def extract_pages_text(pdf_path: str) -> list[dict]:

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"File {pdf_path} does not exist")

    pages = []

    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": idx, "text": text.strip()})

    return pages

def get_page_count(pdf_path: str) -> int:

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File {pdf_path} does not exist")

    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)
    
def load_corpus(corpus_dir: str = "corpus") -> list[dict]:
    root = Path(corpus_dir)

    if not root.exists():
        raise FileNotFoundError(f"Corpus directory {corpus_dir} does not exist")
    
    documents = []

    for pdf_path in root.rglob("*.pdf"):
        print(f"Se citeste PDF-ul: {pdf_path}")

        pages = extract_pages_text(str(pdf_path))
        page_count = get_page_count(str(pdf_path))

        full_text = "\n\n".join(
            f"[Pagina {page['page']}]\n{page['text']}"
            for page in pages if page["text"]
        )

        if not full_text.strip():
            print(f"ATENTIE: nu s-a extras text din PDF, este ignorat: {pdf_path}")
            continue

        documents.append(
            {
                "text": full_text,
                "metadata": {
                    "source": str(pdf_path),
                    "title": pdf_path.stem,
                    "category": pdf_path.parent.name,
                    "page_count": page_count,
                },
            }
        )

    return documents