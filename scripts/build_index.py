
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.pdf_tools import load_corpus
from src.tools.vector_tools import build_index

VECTORSTORE_DIR = ROOT / "vectorstore"
CORPUS_DIR = ROOT / "corpus"


def main(force: bool = False) -> None:
    print("=" * 60)
    print("Build Index — Corpus Juridic → ChromaDB")
    print("=" * 60)

    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        if not force:
            print(f"\n[AVERTISMENT] {VECTORSTORE_DIR} exista si nu e gol.")
            print("Adaugarea de documente ar produce duplicate in ChromaDB.")
            print("Folositi --force pentru a sterge si re-indexa de la zero.")
            print("Iesire fara modificari.")
            sys.exit(0)
        else:
            print(f"\n[--force] Sterg vectorstore existent: {VECTORSTORE_DIR}")
            shutil.rmtree(VECTORSTORE_DIR)

    print(f"\nIncarc corpus din: {CORPUS_DIR}")
    documents = load_corpus(str(CORPUS_DIR))

    if not documents:
        print("[EROARE] Nu s-au gasit documente in corpus/. Verificati structura directorului.")
        sys.exit(1)

    print(f"Documente gasite: {len(documents)}")
    for doc in documents:
        meta = doc["metadata"]
        print(f"  [{meta['category']}] {meta['title']} ({meta['page_count']} pag.)")

    print(f"\nIndexez {len(documents)} documente in ChromaDB...")
    print(f"  chunk_size=1200, chunk_overlap=200")
    print(f"  embedding model: text-embedding-3-small")
    print(f"  persist_directory: {VECTORSTORE_DIR}")

    vectorstore = build_index(
        documents=documents,
        persist_directory=str(VECTORSTORE_DIR),
        chunk_size=1200,
        chunk_overlap=200,
    )

    collection = vectorstore.get()
    n_chunks = len(collection["ids"])
    print(f"\n[OK] Indexare completa. Chunk-uri stocate: {n_chunks}")
    print(f"     Vectorstore salvat in: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexeaza corpusul juridic in ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Sterge vectorstore existent si re-indexeaza")
    args = parser.parse_args()
    main(force=args.force)
