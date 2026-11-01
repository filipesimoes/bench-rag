"""Generate synthetic forensic police inquiry reports (PDF) and ground truth benchmark CSV."""

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_pdf_report(
    file_path: Path | str,
    title: str,
    inquiry_id: str,
    sections: list[dict[str, str | list]],
) -> None:
    """Create a structured forensic police inquiry PDF using ReportLab."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "ForensicHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        alignment=1,  # Center
    )
    title_style = ParagraphStyle(
        "ForensicTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        alignment=1,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1A365D"),
    )
    body_style = ParagraphStyle(
        "ForensicBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
    )

    story = []
    # Official Forensic Header
    story.append(Paragraph("FEDERAL INVESTIGATIVE POLICE DEPARTMENT", header_style))
    story.append(Paragraph("DIRECTORATE FOR COMBATING FINANCIAL CRIMES & CORRUPTION", header_style))
    story.append(Paragraph(f"FINAL INVESTIGATIVE REPORT — {inquiry_id}", header_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 15))

    for sec in sections:
        sec_title = sec["title"]
        story.append(Paragraph(sec_title.upper(), section_heading))
        story.append(Spacer(1, 6))

        content = sec["content"]
        if isinstance(content, str):
            for para in content.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))
                    story.append(Spacer(1, 6))
        elif isinstance(content, list):
            # Formatted table
            t = Table(content, colWidths=[1.8 * inch, 5.0 * inch])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 10))

        story.append(Spacer(1, 10))

    doc.build(story)


def generate_sample_dataset(data_dir: Path | str = "data") -> None:
    """Generate 3 realistic forensic investigation PDF reports and benchmark ground_truth.csv."""
    base_dir = Path(data_dir)
    raw_dir = base_dir / "raw_reports"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Report 1: Shell Companies & Strawman (T1 Focus)
    pdf_1 = raw_dir / "IP_001_2023_FRAUD.pdf"
    create_pdf_report(
        pdf_1,
        title="Operation Phantom Shield: Corporate Fraud & Strawman Network",
        inquiry_id="INQUIRY NO. IP-001/2023-DECIF",
        sections=[
            {
                "title": "I. Formal Qualification of Suspects and Legal Entities",
                "content": [
                    ["Field", "Investigative Findings"],
                    [
                        "Primary Legal Entity",
                        "Nexus Global Holdings Ltd. (Reg. No. 84.102.934/0001-12)",
                    ],
                    ["Registered Address", "Av. Paulista, 1800, Suite 1402, São Paulo/SP"],
                    [
                        "Nominee Director",
                        "Carlos Mendes, Tax ID 392.109.847-22, monthly salary $1,200",
                    ],
                    [
                        "Beneficial Owner",
                        "Eduardo Silveira, Tax ID 109.485.738-99, sole holder of Power of Attorney",
                    ],
                    [
                        "Secondary Shell Entity",
                        "AeroTech Consulting LLC (Reg. No. 91.203.491/0001-55)",
                    ],
                ],
            },
            {
                "title": "II. Summary of Investigative Evidence (Typology T1)",
                "content": (
                    "During the execution of Search and Seizure Warrant No. 44/2023, forensic police officers "
                    "seized extensive corporate incorporation files from the accounting offices of Nexus Global Holdings Ltd.\n\n"
                    "Although Carlos Mendes appears as the 99% majority equity shareholder on official registration documents, "
                    "forensic financial audit confirms he possesses no personal net worth, resides in a modest rented residence, "
                    "and has never participated in executive management meetings. Forensic analysis of seized computer hard drives "
                    "revealed a confidential Power of Attorney dated March 15, 2022, granting Eduardo Silveira irrevocable and total "
                    "administrative power over all corporate bank accounts, contracts, and treasury assets.\n\n"
                    "Furthermore, AeroTech Consulting LLC was incorporated without any physical operational offices or registered employees, "
                    "operating strictly as a pass-through shell conduit to issue fictitious consulting invoices."
                ),
            },
            {
                "title": "III. Conclusion and Criminal Indictment",
                "content": (
                    "The investigative evidence proves beyond reasonable doubt that Carlos Mendes acted solely as a nominee strawman "
                    "('laranja') to insulate the true beneficial owner, Eduardo Silveira, from civil and criminal liability. "
                    "Eduardo Silveira exercised direct operational and financial command over the fraudulent corporate network."
                ),
            },
        ],
    )

    # 2. Report 2: Trade-Based Money Laundering & Crypto Smurfing (T2 Focus)
    pdf_2 = raw_dir / "IP_002_2023_LAUNDERING.pdf"
    create_pdf_report(
        pdf_2,
        title="Operation Clean River: Trade-Based Money Laundering & Crypto Smurfing",
        inquiry_id="INQUIRY NO. IP-002/2023-DELAV",
        sections=[
            {
                "title": "I. Modus Operandi & Smurfing Mechanisms",
                "content": (
                    "Financial Intelligence Unit (FIU) Suspicious Activity Report No. 892/2023 triggered an in-depth "
                    "forensic examination into cash deposit patterns associated with Orion Logistics Import-Export (Reg. No. 47.920.103/0001-80).\n\n"
                    "The criminal organization employed a classic 'smurfing' technique. Between March 1 and March 25, 2023, "
                    "cash couriers ('pombos') conducted 142 structured cash deposits, each strictly calibrated between $9,500 and $9,900 "
                    "— precisely below the $10,000 regulatory reporting threshold mandated by central bank anti-money laundering regulations. "
                    "These deposits were executed across 42 distinct retail banking branches in three metropolitan areas."
                ),
            },
            {
                "title": "II. Trade-Based Over-Invoicing & Crypto Mixing",
                "content": [
                    ["Laundering Vector", "Forensic Accounting Findings"],
                    [
                        "Import Machinery Over-invoicing",
                        "Industrial drills imported at $450,000/unit with verified customs market value of $45,000",
                    ],
                    [
                        "Fictitious Service Invoices",
                        "$2.8M paid to AeroTech Consulting LLC for non-existent IT auditing",
                    ],
                    [
                        "Crypto Mixer Protocol",
                        "128 ETH routed through Tornado Cash and Sinbad mixers to unhosted private cold wallets",
                    ],
                    [
                        "Criminal Jargon Intercepted",
                        "'pedágio' (10% kickback for customs clearance), 'lavagem seca' (pure invoice trade)",
                    ],
                ],
            },
            {
                "title": "III. Criminal Dynamics Analysis (Typology T2)",
                "content": (
                    "The illicit capital originated from public procurement embezzlement. Orion Logistics converted liquid cash into "
                    "over-invoiced imported industrial hardware and offshore digital assets, effectively severing the audit trail and "
                    "re-integrating cleaned capital into legitimate luxury real estate acquisitions."
                ),
            },
        ],
    )

    # 3. Report 3: Command Chronology & Transaction Timelines (T3 Focus)
    pdf_3 = raw_dir / "IP_003_2023_CONSPIRACY.pdf"
    create_pdf_report(
        pdf_3,
        title="Operation Chronos: Command Hierarchy & Transaction Chronology",
        inquiry_id="INQUIRY NO. IP-003/2023-DRACO",
        sections=[
            {
                "title": "I. Chronological Timeline of Financial Transactions (Typology T3)",
                "content": [
                    ["Date", "Transaction Details & Evidence"],
                    [
                        "2023-01-10",
                        "Wire transfer of $2,500,000 from Nexus Global Holdings Ltd. to Orion Logistics Import-Export.",
                    ],
                    [
                        "2023-02-14",
                        "International SWIFT transfer of $1,800,000 from Orion Logistics to Banque de Genève, Switzerland (Acct #CH93-002).",
                    ],
                    [
                        "2023-03-22",
                        "Wire transfer of $450,000 from Banque de Genève to shipyard for purchase of luxury vessel 'Valkyrie' registered under Carlos Mendes.",
                    ],
                    [
                        "2023-04-05",
                        "Encrypted communication intercept: Eduardo Silveira commands courier to destroy all physical ledger copies.",
                    ],
                ],
            },
            {
                "title": "II. Telecommunications & Cyber Forensic Intercepts",
                "content": (
                    "Pursuant to Judicial Wiretap Authorization No. 118/2023, cyber intelligence units intercepted messages "
                    "exchanged over Signal between executive Eduardo Silveira and nominee Carlos Mendes.\n\n"
                    "On March 20, 2023, Silveira instructed: 'Sign the shipyard contract for the Valkyrie tomorrow morning as sole buyer. "
                    "Your monthly bonus will be wired immediately to your wife's account.'\n\n"
                    "On April 5, 2023, following intelligence that federal warrants were imminent, Silveira messaged: "
                    "'Burn the binders in the basement storage and wipe the backup server in Campinas.'"
                ),
            },
            {
                "title": "III. Synthetic Assessment of Intent and Criminal Command",
                "content": (
                    "The chronological reconstruction proves sequential intent (mens rea / dolo). Eduardo Silveira did not merely hold "
                    "passive corporate ties; he orchestrated each step of the asset displacement timeline, directed the strawman's signatures, "
                    "and attempted active obstruction of justice when law enforcement approached."
                ),
            },
        ],
    )

    # 4. Generate Ground Truth CSV (data/ground_truth.csv)
    gt_file = base_dir / "ground_truth.csv"
    benchmark_data = [
        # --- Typology T1: Identification & Associations ---
        {
            "id": "GT-T1-C1",
            "typology": "T1",
            "complexity": "C1",
            "question": "Who is the registered nominee director of Nexus Global Holdings Ltd. and what is his registered monthly salary?",
            "reference_answer": "The registered nominee director of Nexus Global Holdings Ltd. is Carlos Mendes (Tax ID 392.109.847-22), who has a registered monthly salary of $1,200.",
            "inquiry_references": "IP_001_2023_FRAUD",
        },
        {
            "id": "GT-T1-C2",
            "typology": "T1",
            "complexity": "C2",
            "question": "What corporate connection exists between Carlos Mendes and the purchase of the luxury vessel Valkyrie?",
            "reference_answer": "Carlos Mendes, the nominee strawman director of Nexus Global Holdings Ltd. identified in IP-001/2023, was used by Eduardo Silveira to register the ownership of the luxury vessel Valkyrie purchased via offshore wire transfers from Banque de Genève as recorded in IP-003/2023.",
            "inquiry_references": "IP_001_2023_FRAUD; IP_003_2023_CONSPIRACY",
        },
        {
            "id": "GT-T1-C3",
            "typology": "T1",
            "complexity": "C3",
            "question": "Analyze the command structure and evidence of beneficial ownership demonstrating that Eduardo Silveira is the actual head of the corporate network.",
            "reference_answer": "Eduardo Silveira maintained absolute command of the corporate network through an irrevocable Power of Attorney dated March 15, 2022, giving him sole control over Nexus Global Holdings' bank accounts and contracts. Furthermore, telecommunications intercepts prove he directly commanded the nominee Carlos Mendes to sign asset purchase contracts for the vessel Valkyrie and later ordered the destruction of accounting ledgers.",
            "inquiry_references": "IP_001_2023_FRAUD; IP_003_2023_CONSPIRACY",
        },
        # --- Typology T2: Criminal Dynamics ---
        {
            "id": "GT-T2-C1",
            "typology": "T2",
            "complexity": "C1",
            "question": "What smurfing technique was utilized by Orion Logistics and what deposit thresholds were observed?",
            "reference_answer": "Orion Logistics utilized a smurfing technique involving 142 structured cash deposits calibrated between $9,500 and $9,900 across 42 retail banking branches between March 1 and March 25, 2023, intentionally staying below the $10,000 regulatory reporting threshold.",
            "inquiry_references": "IP_002_2023_LAUNDERING",
        },
        {
            "id": "GT-T2-C2",
            "typology": "T2",
            "complexity": "C2",
            "question": "How did Orion Logistics utilize shell entities like AeroTech Consulting LLC to facilitate trade-based money laundering?",
            "reference_answer": "Orion Logistics engaged in trade-based money laundering by paying $2.8 million in fictitious service invoices to AeroTech Consulting LLC (a pass-through shell company with no employees or physical offices established under Nexus Global Holdings) and by over-invoicing imported industrial drills at $450,000 per unit against a market value of $45,000.",
            "inquiry_references": "IP_001_2023_FRAUD; IP_002_2023_LAUNDERING",
        },
        {
            "id": "GT-T2-C3",
            "typology": "T2",
            "complexity": "C3",
            "question": "Synthesize the complete multi-vector asset laundering and concealment mechanism employed across corporate shell structures, trade over-invoicing, and crypto assets.",
            "reference_answer": "The criminal organization deployed a sophisticated three-tiered laundering mechanism: first, cash was structured via smurfing below reporting limits; second, illicit capital was converted into fictitious trade invoices through shell entities like AeroTech Consulting and over-invoiced industrial machinery; third, capital was routed through Ethereum mixers (Tornado Cash and Sinbad) and offshore accounts at Banque de Genève before being re-integrated into tangible luxury assets like the vessel Valkyrie registered under nominees.",
            "inquiry_references": "IP_001_2023_FRAUD; IP_002_2023_LAUNDERING; IP_003_2023_CONSPIRACY",
        },
        # --- Typology T3: Tracking & Chronology ---
        {
            "id": "GT-T3-C1",
            "typology": "T3",
            "complexity": "C1",
            "question": "On what date was the initial wire transfer of $2,500,000 made from Nexus Global Holdings Ltd. to Orion Logistics?",
            "reference_answer": "The initial wire transfer of $2,500,000 was executed on January 10, 2023 from Nexus Global Holdings Ltd. to Orion Logistics Import-Export.",
            "inquiry_references": "IP_003_2023_CONSPIRACY",
        },
        {
            "id": "GT-T3-C2",
            "typology": "T3",
            "complexity": "C2",
            "question": "Trace the chronological flow of funds starting from Nexus Global Holdings Ltd. on January 10, 2023 to the shipyard purchase on March 22, 2023.",
            "reference_answer": "On January 10, 2023, Nexus Global Holdings wired $2,500,000 to Orion Logistics. On February 14, 2023, Orion Logistics transferred $1,800,000 to Banque de Genève in Switzerland. Finally, on March 22, 2023, $450,000 was wired from Banque de Genève directly to the shipyard to complete the purchase of the luxury yacht Valkyrie.",
            "inquiry_references": "IP_001_2023_FRAUD; IP_002_2023_LAUNDERING; IP_003_2023_CONSPIRACY",
        },
        {
            "id": "GT-T3-C3",
            "typology": "T3",
            "complexity": "C3",
            "question": "What actions did Eduardo Silveira take on April 5, 2023, and how does this demonstrate conscious obstruction of justice?",
            "reference_answer": "On April 5, 2023, upon learning that federal search warrants were imminent, Eduardo Silveira sent encrypted Signal instructions commanding his subordinates to burn physical accounting ledgers in basement storage and wipe the backup server in Campinas. This demonstrates intentional, conscious obstruction of justice to destroy evidentiary audit trails.",
            "inquiry_references": "IP_003_2023_CONSPIRACY",
        },
    ]

    with open(gt_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "typology",
                "complexity",
                "question",
                "reference_answer",
                "inquiry_references",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark_data)
