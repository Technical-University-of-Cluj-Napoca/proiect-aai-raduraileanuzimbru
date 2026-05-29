from pathlib import Path
import pdfplumber

def extract_pages_text(pdf_path: str) -> list[dict]:

    """
    Extrage textul din PDF, pagina cu pagina.
    Returneaza o lista de forma:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]
    """

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
    """
    Returneaza numarul de pagini din PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File {pdf_path} does not exist")

    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)