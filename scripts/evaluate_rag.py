
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.dtos import ClauseDTO, ClauseType
from src.agents.retrieval_agent import RAGRetrievalAgent

load_dotenv()

EVAL_QUESTIONS = [
    {
        "question": "Ce prevederi sunt relevante pentru penalitati de intarziere in contractele de achizitii publice?",
        "clause_type": ClauseType.penalitate,
        "clause_text": "Prestatorul datoreaza penalitati de intarziere pentru nerespectarea termenelor contractuale.",
        "section": "Penalitati",
    },
    {
        "question": "Ce obligatii de informare exista pentru prelucrarea datelor personale conform GDPR?",
        "clause_type": ClauseType.date_personale,
        "clause_text": "Beneficiarul colecteaza si proceseaza date personale ale angajatilor prestatorului.",
        "section": "Date personale",
    },
    {
        "question": "Cum trebuie definita forta majora intr-un contract?",
        "clause_type": ClauseType.forta_majora,
        "clause_text": "Forta majora exonereaza partile de raspundere in cazul evenimentelor imprevizibile.",
        "section": "Forta majora",
    },
    {
        "question": "Cand poate deveni problematica o clauza de reziliere unilaterala?",
        "clause_type": ClauseType.reziliere,
        "clause_text": "Beneficiarul poate rezilia contractul oricand, fara a datora despagubiri prestatorului.",
        "section": "Incetarea contractului",
    },
    {
        "question": "Ce riscuri exista pentru clauzele de confidentialitate cu durata nedefinita?",
        "clause_type": ClauseType.confidentialitate,
        "clause_text": "Prestatorul se obliga sa pastreze confidentialitatea informatiilor pe termen nedefinit.",
        "section": "Confidentialitate",
    },
    {
        "question": "Ce reguli se aplica garantiei de buna executie in achizitii publice?",
        "clause_type": ClauseType.obligatie,
        "clause_text": "Prestatorul constituie garantia de buna executie in cuantum de 10% din valoarea contractului.",
        "section": "Garantii",
    },
    {
        "question": "Ce conditii trebuie respectate pentru modificarea contractului de achizitie publica?",
        "clause_type": ClauseType.altele,
        "clause_text": "Orice modificare a contractului se face prin act aditional semnat de ambele parti.",
        "section": "Dispozitii generale",
    },
    {
        "question": "Ce reguli sunt relevante pentru cesiunea contractului?",
        "clause_type": ClauseType.altele,
        "clause_text": "Prestatorul poate ceda contractul unui tert fara acordul prealabil al beneficiarului.",
        "section": "Cesiunea contractului",
    },
    {
        "question": "Ce prevederi sunt relevante pentru raspunderea contractuala?",
        "clause_type": ClauseType.obligatie,
        "clause_text": "Raspunderea prestatorului este limitata la valoarea contractului.",
        "section": "Raspundere",
    },
    {
        "question": "Ce reguli sunt relevante pentru solutionarea litigiilor si arbitraj?",
        "clause_type": ClauseType.altele,
        "clause_text": "Litigiile se solutioneaza pe cale amiabila sau prin arbitraj comercial.",
        "section": "Litigii",
    },
]


def build_clause(q: dict, idx: int) -> ClauseDTO:
    return ClauseDTO(
        id=f"eval_clause_{idx}",
        section=q["section"],
        text=q["clause_text"],
        page=1,
        type=q["clause_type"],
    )


def generate_answer(question: str, contexts: list[str], llm: ChatOpenAI) -> str:
    if not contexts:
        return "Nu a fost gasit context relevant pentru aceasta intrebare."

    context_str = "\n\n".join(contexts[:3])
    prompt = (
        f"Pe baza urmatoarelor fragmente juridice, raspunde la intrebare:\n\n"
        f"Context:\n{context_str}\n\n"
        f"Intrebare: {question}\n\n"
        f"Raspuns concis (3-5 fraze):"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def main() -> None:
    print("=" * 60)
    print("Evaluare RAG cu RAGAS")
    print("=" * 60)

    Path("logs").mkdir(exist_ok=True)

    retrieval_agent = RAGRetrievalAgent(persist_directory="vectorstore", threshold=0.05, k=5)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for idx, q in enumerate(EVAL_QUESTIONS):
        print(f"\n[{idx+1}/{len(EVAL_QUESTIONS)}] {q['question'][:60]}...")
        clause = build_clause(q, idx)
        results = retrieval_agent.retrieve(clause, k=5)

        retrieved_texts = [r.text for r in results]
        answer = generate_answer(q["question"], retrieved_texts, llm)

        questions.append(q["question"])
        answers.append(answer)
        contexts_list.append(retrieved_texts if retrieved_texts else [""])
        ground_truths.append("")

        print(f"  → {len(retrieved_texts)} chunk-uri recuperate")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("\nRulez evaluarea RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    threshold_passed = all(v >= 0.5 for v in scores.values())

    output = {
        "model": "gpt-4o-mini + text-embedding-3-small",
        "corpus_size": len(EVAL_QUESTIONS),
        "retrieval_threshold": 0.05,
        "scores": scores,
        "threshold_0_5_passed": threshold_passed,
        "comment": (
            "Evaluare RAGAS reference-free pe 10 intrebari juridice reprezentative. "
            "faithfulness: cat de bine ancoreaza LLM-ul raspunsul in context (previne halucinatiile). "
            "answer_relevancy: cat de relevant e raspunsul fata de intrebare. "
            "context_precision: cat de utile sunt chunk-urile recuperate. "
            "context_recall: cat din informatia necesara e acoperita de context."
        ),
        "per_question": [
            {
                "question": q["question"],
                "clause_type": q["clause_type"].value,
                "contexts_retrieved": len(contexts_list[i]),
            }
            for i, q in enumerate(EVAL_QUESTIONS)
        ],
    }

    output_path = "logs/rag_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Rezultate RAGAS:")
    for metric, score in scores.items():
        status = "✓" if score >= 0.5 else "✗"
        print(f"  {status} {metric}: {score:.3f}")
    print(f"\nFisier salvat: {output_path}")


if __name__ == "__main__":
    main()
