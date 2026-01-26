"""
MADAM-RAG: Multi-Agent Debate for RAG.

This module provides two components:

1. MADAMRetrieveRAG: Simple retriever that fetches N documents (one per agent).
   Fits the standard RAGMethod interface for use with RAGSetting.

2. MADAMDebateWorkflow: Full multi-agent debate workflow for standalone use.
   This includes the debate logic and aggregation, which doesn't fit the
   standard RAGMethod interface (since it includes answer generation).

Algorithm (full workflow):
1. Retrieve N documents (one per agent)
2. Round 1: Each agent answers based ONLY on their assigned document
3. Round 2+: Each agent sees other agents' responses and updates their answer
4. Early stopping: If all agents' answers converge with previous round, stop
5. Aggregation: Combine all agent responses into final answer

Adapted from rag_experiments/workflows/madam_rag.py
"""

import re
import string
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


# ============================================================================
# Simple Retriever (fits RAGMethod interface)
# ============================================================================

class MADAMRetrieveRAG(RAGMethod):
    """Simple retriever for MADAM-RAG: retrieves N documents (one per agent).

    This is a minimal RAGMethod that just retrieves documents.
    For the full debate workflow, use MADAMDebateWorkflow.

    Example:
        method = MADAMRetrieveRAG(retriever, num_agents=3)
        evidence = method.retrieve("Does aspirin prevent heart attacks?")
        # Returns 3 documents, one for each agent
    """

    def __init__(
        self,
        retriever: "BaseRetriever",
        num_agents: int = 3,
    ):
        """Initialize MADAM retriever.

        Args:
            retriever: Underlying retriever.
            num_agents: Number of agents/documents to retrieve.
        """
        super().__init__(retriever, top_k=num_agents)
        self.num_agents = num_agents

    @property
    def name(self) -> str:
        return f"madam_retrieve_n{self.num_agents}"

    def retrieve(self, query: str) -> list["Evidence"]:
        """Retrieve one document per agent."""
        evidence = self.retriever.retrieve(query, top_k=self.num_agents)

        # Add agent assignment to metadata
        for i, e in enumerate(evidence):
            e.metadata["agent_id"] = i + 1
            e.metadata["rag_method"] = self.name

        return evidence


# ============================================================================
# Full Debate Workflow (standalone use)
# ============================================================================

@dataclass
class AgentResponse:
    """Structured response from an agent."""
    raw: str
    answer: str
    explanation: str


@dataclass
class RoundRecord:
    """Record of a single debate round."""
    responses: list[AgentResponse] = field(default_factory=list)
    aggregation: str | None = None


@dataclass
class DebateResult:
    """Result of the full MADAM debate workflow."""
    answer: str
    documents: list["Evidence"]
    num_rounds: int
    debate_records: dict[str, RoundRecord]
    final_aggregation: str


# Default prompts
AGENT_INITIAL_PROMPT = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{answer}}. Explanation: {{explanation}}.'"""

AGENT_DEBATE_PROMPT = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following responses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{answer}}. Explanation: {{explanation}}.'"""

AGGREGATION_PROMPT = """You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963.
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{agent_responses}
"""


class MADAMDebateWorkflow:
    """Full MADAM-RAG debate workflow.

    This is a standalone workflow that includes multi-agent debate.
    It does NOT fit the standard RAGMethod interface because it
    includes answer generation, not just retrieval.

    Use this for custom evaluation scenarios or standalone experiments.

    Example:
        workflow = MADAMDebateWorkflow(
            retriever=retriever,
            generate_fn=lambda prompt: llm.generate(prompt),
            num_agents=3,
            num_rounds=3,
        )
        result = workflow.run("Does aspirin prevent heart attacks?")
        print(result.answer)
        print(result.debate_records)
    """

    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str], str],
        num_agents: int = 3,
        num_rounds: int = 3,
    ):
        """Initialize MADAM debate workflow.

        Args:
            retriever: Underlying retriever.
            generate_fn: Function to generate text. Signature: (prompt) -> str
            num_agents: Number of agents (and documents).
            num_rounds: Maximum number of debate rounds.
        """
        self.retriever = retriever
        self.generate_fn = generate_fn
        self.num_agents = num_agents
        self.num_rounds = num_rounds

    @property
    def name(self) -> str:
        return f"madam_debate_n{self.num_agents}_r{self.num_rounds}"

    def run(self, query: str) -> DebateResult:
        """Execute the full MADAM debate workflow."""
        # Step 1: Retrieve documents (one per agent)
        documents = self.retriever.retrieve(query, top_k=self.num_agents)

        # Ensure we have enough documents for all agents
        if len(documents) < self.num_agents:
            # Pad with available documents if not enough
            while len(documents) < self.num_agents:
                documents.append(documents[len(documents) % len(documents)])

        records: dict[str, RoundRecord] = {}

        # Step 2: Round 1 - Each agent answers based only on their document
        round1 = RoundRecord()
        agent_outputs = []

        for i, doc in enumerate(documents[:self.num_agents]):
            response = self._agent_response(query, doc.text, history=None)
            parsed = self._parse_response(response)
            round1.responses.append(parsed)
            agent_outputs.append(response)

        round1.aggregation = self._aggregate_responses(query, agent_outputs)
        records["round1"] = round1

        # Step 3: Additional debate rounds
        final_aggregation = round1.aggregation

        for t in range(2, self.num_rounds + 1):
            round_key = f"round{t}"
            current_round = RoundRecord()
            new_outputs = []

            for i, doc in enumerate(documents[:self.num_agents]):
                # Build history from other agents (exclude current agent)
                history = self._build_history(agent_outputs, exclude_index=i)
                response = self._agent_response(query, doc.text, history=history)
                parsed = self._parse_response(response)
                current_round.responses.append(parsed)
                new_outputs.append(response)

            # Check for early stopping (convergence)
            prev_round = records[f"round{t-1}"]
            if self._check_convergence(current_round.responses, prev_round.responses):
                # Answers converged, use previous aggregation
                final_aggregation = prev_round.aggregation
                records[round_key] = current_round
                break

            # Continue debate
            current_round.aggregation = self._aggregate_responses(query, new_outputs)
            final_aggregation = current_round.aggregation
            records[round_key] = current_round
            agent_outputs = new_outputs

        # Extract final answer from aggregation
        final_answer = self._extract_final_answer(final_aggregation)

        return DebateResult(
            answer=final_answer,
            documents=documents[:self.num_agents],
            num_rounds=len(records),
            debate_records=records,
            final_aggregation=final_aggregation,
        )

    def _agent_response(self, query: str, document: str, history: str | None = None) -> str:
        """Generate a response from an agent."""
        if history:
            prompt = AGENT_DEBATE_PROMPT.format(
                query=query,
                document=document,
                history=history,
            )
        else:
            prompt = AGENT_INITIAL_PROMPT.format(
                query=query,
                document=document,
            )

        return self.generate_fn(prompt)

    def _build_history(self, agent_outputs: list[str], exclude_index: int) -> str:
        """Build history string from other agents' responses."""
        history_parts = []
        for j, output in enumerate(agent_outputs):
            if j != exclude_index:
                history_parts.append(f"Agent {j + 1}: {output}")
        return "\n".join(history_parts)

    def _parse_response(self, response: str) -> AgentResponse:
        """Parse agent response to extract answer and explanation."""
        answer = ""
        explanation = ""

        # Extract answer
        answer_match = response.find("Answer: ")
        explanation_match = response.find("Explanation:")

        if answer_match != -1:
            if explanation_match != -1:
                answer = response[answer_match + len("Answer: "):explanation_match].strip()
                # Remove trailing period if present
                answer = answer.rstrip(". ")
            else:
                answer = response[answer_match + len("Answer: "):].strip()

        if explanation_match != -1:
            explanation = response[explanation_match + len("Explanation:"):].strip()

        return AgentResponse(raw=response, answer=answer, explanation=explanation)

    def _aggregate_responses(self, query: str, agent_outputs: list[str]) -> str:
        """Aggregate all agent responses into a final answer."""
        joined = "\n".join(
            f"Agent {i + 1}: {output}" for i, output in enumerate(agent_outputs)
        )

        prompt = AGGREGATION_PROMPT.format(
            query=query,
            agent_responses=joined,
        )

        return self.generate_fn(prompt)

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for comparison (lowercase, remove articles/punctuation)."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            return "".join(ch for ch in text if ch not in string.punctuation)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _check_convergence(
        self,
        current_responses: list[AgentResponse],
        prev_responses: list[AgentResponse],
    ) -> bool:
        """Check if agent answers have converged between rounds."""
        if len(current_responses) != len(prev_responses):
            return False

        for curr, prev in zip(current_responses, prev_responses):
            curr_norm = self._normalize_answer(curr.answer)
            prev_norm = self._normalize_answer(prev.answer)

            # Check if answers are similar (one contains the other)
            if curr_norm not in prev_norm and prev_norm not in curr_norm:
                return False

        return True

    def _extract_final_answer(self, aggregation: str) -> str:
        """Extract the final answer from aggregation response."""
        # Try to find "All Correct Answers: [...]"
        match = re.search(r"All Correct Answers:\s*\[([^\]]*)\]", aggregation)
        if match:
            answers_str = match.group(1)
            # Parse the list of answers
            answers = re.findall(r'"([^"]*)"', answers_str)
            if answers:
                return ", ".join(answers)

        # Fallback: return the full aggregation
        return aggregation
