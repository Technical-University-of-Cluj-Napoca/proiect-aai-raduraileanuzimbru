import os
from dotenv import load_dotenv

from src.dtos import ClauseDTO, RetrievalResultDTO
from src.tools.vector_tools import load_vectorstore

class RAGRetrievalAgent:
    def __init__(
            self,
            persist_directory: str = "vectorstore",
            threshold: float | None = None,
            k: int = 5,
    ):
        self.threshold = (
            threshold if threshold is not None else float(os.getenv("RETRIEVAL THRESHOLD", "0.55"))
        )
        self.k = k
        self.vectorstore = load_vectorstore(persist_directory=persist_directory)

    def retrieve(self, 
                 clause: ClauseDTO, 
                 k: int | None = None,
                 threshold: float | None = None,
    ) -> list[RetrievalResultDTO]:
        final_k = k or self.k
        final_threshold = threshold if threshold is not None else self.threshold

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=self._build_query(clause),
            k=final_k,
        )

        output: list[RetrievalResultDTO] = []

        for doc, score in results:
            if score < final_threshold:
                continue
            
            output.append(
                RetrievalResultDTO(
                    text=doc.page_content,
                    source=(
                        f"{doc.metadata.get('category', '')}/"
                        f"{doc.metadata.get('title', '')} | "
                        f"{doc.metadata.get('source', 'sursa necunoscuta')} | "
                        f"chunk {doc.metadata.get('chunk_id', '')}"
                    ),
                    score=float(score),
                )
            )
        return sorted(output, key=lambda item: item.score, reverse=True)
    
    def _build_query(self, clause: ClauseDTO) -> str:
        return (
            f"Tip clauza: {clause.type.value}\n"
            f"Sectiune: {clause.section}\n"
            f"Text clauza:\n{clause.text}"
        )