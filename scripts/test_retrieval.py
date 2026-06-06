
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.agents.retrieval_agent import RAGRetrievalAgent
from src.dtos import ClauseDTO, ClauseType

TEST_CLAUSES = [
    ClauseDTO(
        id="test_penalitate",
        section="Penalitati de intarziere",
        text=(
            "In cazul in care prestatorul nu respecta termenele de executie prevazute in contract, "
            "acesta datoreaza beneficiarului penalitati de intarziere de 0.5% pe zi din valoarea "
            "contractului, fara plafon maxim, calculate incepand cu prima zi de intarziere."
        ),
        page=5,
        type=ClauseType.penalitate,
    ),
    ClauseDTO(
        id="test_date_personale",
        section="Prelucrarea datelor cu caracter personal",
        text=(
            "Prestatorul poate prelucra datele cu caracter personal ale angajatilor beneficiarului "
            "in scopul executarii contractului, fara a specifica temeiul legal al prelucrarii, "
            "perioada de stocare sau drepturile persoanelor vizate."
        ),
        page=8,
        type=ClauseType.date_personale,
    ),
    ClauseDTO(
        id="test_forta_majora",
        section="Forta majora",
        text=(
            "Forta majora reprezinta orice eveniment care impiedica executarea contractului. "
            "Partea care invoca forta majora va notifica cealalta parte in termen de 30 de zile "
            "de la producerea evenimentului."
        ),
        page=12,
        type=ClauseType.forta_majora,
    ),
    ClauseDTO(
        id="test_reziliere",
        section="Incetarea contractului",
        text=(
            "Beneficiarul poate rezilia contractul in orice moment, fara preaviz si fara a datora "
            "prestatorului despagubiri pentru lucrarile executate si materialele achizitionate "
            "in vederea executarii contractului."
        ),
        page=15,
        type=ClauseType.reziliere,
    ),
    ClauseDTO(
        id="test_confidentialitate",
        section="Obligatii de confidentialitate",
        text=(
            "Prestatorul se obliga sa pastreze confidentialitatea tuturor informatiilor primite "
            "de la beneficiar pe durata contractului si dupa incetarea acestuia, pe o perioada "
            "nedeterminata, fara nicio limitare temporala."
        ),
        page=18,
        type=ClauseType.confidentialitate,
    ),
]


def main() -> None:
    print("=" * 60)
    print("Test RAG Retrieval Agent — 5 clauze reprezentative")
    print("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    agent = RAGRetrievalAgent(persist_directory="vectorstore", threshold=0.05, k=5)

    all_scores: list[list[float]] = []
    all_sources: list[list[str]] = []

    for clause in TEST_CLAUSES:
        print(f"\n{'─'*50}")
        print(f"Clauza: {clause.id} | Tip: {clause.type.value}")
        print(f"Text: {clause.text[:100]}...")

        results = agent.retrieve(clause, k=5)
        scores = [r.score for r in results]
        sources = [r.source.split("|")[0].strip() for r in results]

        while len(scores) < 5:
            scores.append(0.0)
            sources.append("—")

        all_scores.append(scores[:5])
        all_sources.append(sources[:5])

        print(f"Chunk-uri recuperate: {len(results)}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] scor={r.score:.3f} | {r.source[:60]}")


    print("\n" + "─" * 50)
    print("OBSERVATIE: caz de chunk cu scor bun dar irelevant juridic:")
    print("  Clauza 'confidentialitate' poate recupera fragmente GDPR despre")
    print("  drepturile persoanelor vizate (scor ~0.45) datorita vocabularului")
    print("  comun ('informatii', 'date', 'terte parti'), desi contextul GDPR")
    print("  nu este relevant pentru evaluarea unei obligatii NDA contractuale.")
    print("  Solutie: reranker sau filtru pe metadata (category != 'gdpr' pentru NDA).")

    score_matrix = np.array(all_scores)

    clause_labels = [c.id.replace("test_", "") for c in TEST_CLAUSES]
    chunk_labels = [f"chunk {i+1}" for i in range(5)]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(score_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_yticks(range(len(TEST_CLAUSES)))
    ax.set_xticklabels(chunk_labels, fontsize=10)
    ax.set_yticklabels(clause_labels, fontsize=10)

    for i in range(len(TEST_CLAUSES)):
        for j in range(5):
            val = score_matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Scor similaritate cosinus")
    ax.set_title("Heatmap retrieval: scor similaritate per clauza si chunk", fontsize=13, fontweight="bold")
    ax.set_xlabel("Chunk recuperat (ordonat descrescator dupa scor)")
    ax.set_ylabel("Clauza testata")

    fig.tight_layout()
    output_path = "logs/retrieval_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nHeatmap salvat: {output_path}")


if __name__ == "__main__":
    main()
