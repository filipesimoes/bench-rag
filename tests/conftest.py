"""Pytest fixtures and test helpers."""

import pytest

from bench_rag.core.evaluator import GroundTruthItem
from bench_rag.core.parser import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide a list of realistic sample documents."""
    return [
        Document(
            id="IP_001_2023_FRAUD",
            text=(
                "Nexus Global Holdings Ltd. was incorporated on 2021-04-10. "
                "Carlos Mendes is registered as nominee director with monthly salary of $1,200. "
                "Eduardo Silveira holds irrevocable power of attorney over all bank accounts. "
                "AeroTech Consulting LLC acts as a pass-through shell company."
            ),
            metadata={"source_file": "IP_001_2023_FRAUD.pdf", "total_pages": 1},
        ),
        Document(
            id="IP_002_2023_LAUNDERING",
            text=(
                "Orion Logistics conducted smurfing with 142 structured cash deposits "
                "between $9,500 and $9,900 below regulatory threshold across 42 bank branches. "
                "Over-invoiced industrial drills imported at $450,000 against market value $45,000. "
                "128 ETH routed through Tornado Cash crypto mixer."
            ),
            metadata={"source_file": "IP_002_2023_LAUNDERING.pdf", "total_pages": 1},
        ),
        Document(
            id="IP_003_2023_CONSPIRACY",
            text=(
                "Chronological financial flow: On 2023-01-10, $2.5M transferred from Nexus to Orion. "
                "On 2023-02-14, $1.8M wired to Banque de Genève. "
                "On 2023-03-22, $450,000 wired to shipyard for luxury vessel Valkyrie under Carlos Mendes. "
                "On 2023-04-05, Eduardo Silveira ordered destruction of ledgers and wiping of backup server."
            ),
            metadata={"source_file": "IP_003_2023_CONSPIRACY.pdf", "total_pages": 1},
        ),
    ]


@pytest.fixture
def sample_ground_truth_items() -> list[GroundTruthItem]:
    """Provide representative ground-truth items across T1-T3 and C1-C3."""
    return [
        GroundTruthItem(
            id="GT-T1-C1",
            typology="T1",
            complexity="C1",
            question="Who is the nominee director of Nexus Global Holdings?",
            reference_answer="Carlos Mendes is the nominee director of Nexus Global Holdings with a monthly salary of $1,200.",
            inquiry_references=["IP_001_2023_FRAUD"],
        ),
        GroundTruthItem(
            id="GT-T2-C1",
            typology="T2",
            complexity="C1",
            question="What cash smurfing deposits were conducted by Orion Logistics?",
            reference_answer="Orion Logistics made 142 deposits between $9,500 and $9,900 below reporting thresholds.",
            inquiry_references=["IP_002_2023_LAUNDERING"],
        ),
        GroundTruthItem(
            id="GT-T3-C2",
            typology="T3",
            complexity="C2",
            question="Trace the funds from Nexus Global Holdings to the purchase of the luxury vessel Valkyrie.",
            reference_answer="Nexus wired $2.5M to Orion Logistics, which wired $1.8M to Banque de Genève, which wired $450,000 to the shipyard for vessel Valkyrie.",
            inquiry_references=[
                "IP_001_2023_FRAUD",
                "IP_002_2023_LAUNDERING",
                "IP_003_2023_CONSPIRACY",
            ],
        ),
    ]
