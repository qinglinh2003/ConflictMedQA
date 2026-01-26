"""
GAR: Generation-Augmented Retrieval.

Paper: https://arxiv.org/abs/2207.02578

Algorithm:
1. Generate hypothetical content (answer/title/sentence/context) using LLM
2. Augment the retrieval query with generated content
3. Retrieve documents using the augmented query

Supported generation modes:
- answer: Generate a hypothetical answer
- title: Generate a hypothetical document title
- sentence: Generate a hypothetical relevant sentence
- context: Generate a hypothetical passage/context (HyDE-style)

Adapted from rag_experiments/workflows/gar.py
"""

from typing import Callable, Literal, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


GenerationMode = Literal["answer", "title", "sentence", "context"]


# Default prompts for different generation modes
DEFAULT_PROMPTS = {
    "answer": """Given the following question, generate a short hypothetical answer that would likely appear in a relevant document. Do not explain, just provide the answer.

Question: {query}

Hypothetical answer:""",

    "title": """Given the following question, generate a hypothetical document title that would contain the answer. Do not explain, just provide the title.

Question: {query}

Hypothetical document title:""",

    "sentence": """Given the following question, generate a hypothetical sentence from a document that would answer this question. Do not explain, just provide the sentence.

Question: {query}

Hypothetical sentence:""",

    "context": """Given the following question, generate a hypothetical paragraph from a document that would contain information to answer this question. Do not explain, just provide the paragraph.

Question: {query}

Hypothetical paragraph:""",
}


class GARRAG(RAGMethod):
    """Generation-Augmented Retrieval (GAR).

    This method generates hypothetical content based on the query,
    then uses it to augment retrieval. Different modes generate
    different types of content.

    Example:
        def generate(prompt: str) -> str:
            return llm.generate(prompt)

        method = GARRAG(
            retriever=retriever,
            generate_fn=generate,
            mode="answer",
            top_k=10,
        )
        evidence = method.retrieve("Does aspirin prevent heart attacks?")
    """

    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str], str],
        mode: GenerationMode = "answer",
        top_k: int = 10,
        custom_prompt: str | None = None,
    ):
        """Initialize GAR.

        Args:
            retriever: Underlying retriever.
            generate_fn: Function to generate text. Signature: (prompt) -> str
            mode: Generation mode - one of "answer", "title", "sentence", "context".
            top_k: Number of documents to retrieve.
            custom_prompt: Custom prompt template (must contain {query} placeholder).
        """
        super().__init__(retriever, top_k)
        if mode not in DEFAULT_PROMPTS:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(DEFAULT_PROMPTS.keys())}")
        self.generate_fn = generate_fn
        self.mode = mode
        self.prompt_template = custom_prompt or DEFAULT_PROMPTS[mode]

    @property
    def name(self) -> str:
        return f"gar_{self.mode}_top{self.top_k}"

    def retrieve(self, query: str) -> list["Evidence"]:
        """Generate hypothetical content, augment query, and retrieve."""
        # Step 1: Generate hypothetical content
        generated_content = self._generate_hypothetical(query)

        # Step 2: Augment query with generated content
        augmented_query = self._augment_query(query, generated_content)

        # Step 3: Retrieve documents using augmented query
        evidence = self.retriever.retrieve(augmented_query, top_k=self.top_k)

        # Add metadata about the GAR process
        for e in evidence:
            e.metadata["original_query"] = query
            e.metadata["gar_mode"] = self.mode
            e.metadata["generated_content"] = generated_content[:500]  # Truncate for storage
            e.metadata["augmented_query"] = augmented_query[:500]
            e.metadata["rag_method"] = self.name

        return evidence

    def _generate_hypothetical(self, query: str) -> str:
        """Generate hypothetical content based on the configured mode."""
        prompt = self.prompt_template.format(query=query)
        return self.generate_fn(prompt).strip()

    def _augment_query(self, query: str, generated_content: str) -> str:
        """
        Combine original query with generated content for retrieval.

        The augmentation strategy depends on the mode:
        - answer: Append the hypothetical answer to help match answer-containing passages
        - title: Use title to find documents with similar titles
        - sentence: Use sentence to find semantically similar passages
        - context: Use the full context for dense retrieval (HyDE-style)
        """
        if self.mode == "context":
            # For context mode, the generated content is rich enough to use alone
            # (similar to HyDE - Hypothetical Document Embeddings)
            return generated_content
        else:
            # For other modes, concatenate query with generated content
            return f"{query} {generated_content}"


# Convenience aliases for common GAR variants
class GARAnswerRAG(GARRAG):
    """GAR with answer generation mode."""

    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str], str],
        top_k: int = 10,
    ):
        super().__init__(retriever, generate_fn, mode="answer", top_k=top_k)


class GARContextRAG(GARRAG):
    """GAR with context generation mode (HyDE-style)."""

    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str], str],
        top_k: int = 10,
    ):
        super().__init__(retriever, generate_fn, mode="context", top_k=top_k)
