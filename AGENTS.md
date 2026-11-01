# Project Guidelines & Rules — Bench-RAG

## Critical Rule: Data Privacy & Prohibition of Real Data in Code and Tests

> [!CAUTION]
> **FUNDAMENTAL AND INVIOLABLE PRIVACY & SECURITY RULE**
> It is strictly forbidden to insert, hardcode, commit, or record in the source code (`src/`), tests (`tests/`), configurations (`configs/`), or documentation (`docs/`, `README.md`):
> 1. **Real names of individuals** (investigated individuals, police officers, public servants, witnesses, parties, or signatories);
> 2. **Real company names or corporate entities** mentioned in inquiries or legal proceedings;
> 3. **Real judicial process, inquiry, or electronic document numbers** (e.g., electronic court document IDs, hashes, links with real tracking identifiers, digital signature numbers);
> 4. **Real personal data** (tax IDs, national IDs, physical addresses, phone numbers, vehicle license plates, bank accounts);
> 5. **Real case facts, specific events, or literal report transcripts** extracted from investigative test or evaluation documents (such as files contained in `data/real/`).

---

### Guidelines for Tests and Mocks

1. **Exclusive Use of Synthetic Data in Mocks and Fixtures**:
   - Any fixture, mock, test string, or programmatically generated document must use **fictitious or generic placeholders** (e.g., `"TEST OFFICER"`, `"TEST AGENT"`, `"FICTITIOUS CORP LLC"`).
   - Numeric identifiers and metadata must use dummy numbers or zeroed masks (e.g., `"00000000000000000000000000000"`, `"9999999999"`, `"000/0000"`).

2. **Integration Tests with Real Files (e.g., `test_real_pdf_cleaning.py`)**:
   - When a test processes a real PDF to validate parsing or cleaning pipelines:
     - Assertions must validate **only structural and quantitative properties** (e.g., `assert len(signatures) >= 1`, `assert bool(s["signatory"].strip())`, character removal counts, chunk reduction, noise elimination).
     - **Never** write assertions (`assert`) checking for the presence or equality of real person names, real document numbers, or sensitive case facts.

3. **Automated Verification**:
   - The repository includes the automated compliance test `tests/unit/test_privacy_compliance.py`, which scans the codebase for prohibited real terms or identifiers. Any accidental inclusion will immediately fail the test suite.
