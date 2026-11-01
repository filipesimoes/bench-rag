# Data Privacy and Compliance Guidelines

This document establishes the mandatory policies for handling real data, generating synthetic data, and developing tests within the **Bench-RAG** project.

---

## 1. Core Principles

The Bench-RAG framework evaluates and benchmarks Retrieval-Augmented Generation (RAG) architectures on criminal investigation reports and forensic documents.

Many of these reports contain sensitive personal data, confidential judicial records, and protected investigative findings. Consequently, the boundary between **real evaluation data** and the **source code/test suite** must be strictly maintained.

---

## 2. Strict Prohibitions in Code and Tests

It is strictly forbidden to hardcode, commit, or register in source code (`src/`), test suites (`tests/`), configurations (`configs/`), or documentation (`docs/`, `README.md`):

1. **Real Names of Individuals**: Suspects, victims, witnesses, legal counsel, judges, police investigators, clerks, or any public officials signing electronic documents (e.g., electronic court/PJe signatures).
2. **Real Corporate Entities**: Tax identifiers, legal business names, trade names, or registered addresses of real companies mentioned in investigation files.
3. **Real Judicial and Case Identifiers**:
   - Electronic court document IDs (e.g., 29-digit electronic filing tracking numbers like `00000000000000000000000000000`);
   - Page/document sequence IDs (e.g., `Num. 0000000000`);
   - Full URLs containing live case query parameters;
   - Real police inquiry numbers or court case numbers in literal test assertions.
4. **Real Case Facts and Evidence**: Literal excerpts of depositions, allegations, real financial transaction amounts, or details of alleged criminal acts.

---

## 3. Standards for Mocks and Unit Tests

- **Identities**: Use generic, fictitious roles such as `"TEST OFFICER"`, `"TEST AGENT"`, `"TEST INVESTIGATOR"`, `"FICTITIOUS ENTERPRISES LLC"`.
- **Document Numbers**: Use masked dummy sequences such as `"00000000000000000000000000000"` or `"9999999999"`.
- **Timestamps**: Use standardized neutral dates and times (e.g., `"01/01/2025 10:00:00"`).
- **URLs**: Use mock domain names or endpoints without case-specific parameters.

---

## 4. Tests with Real Files (`tests/integration/`)

When real document parsing and preprocessing (such as header/footer sanitization in `HeaderFooterCleaner`) must be tested against real evaluation files (in `data/real/`):
- The test must validate **functional and structural properties**, such as:
  - Total character volume removed;
  - Removal of repeated pagination and header/footer noise;
  - Chunk count reduction and preservation of substantive sections;
  - Presence of required schema keys in metadata dictionaries (`"signatory"`, `"datetime"`, `"document_number"`);
  - Non-empty presence of required fields (`assert bool(s["signatory"].strip())`).
- **NEVER** write assertions comparing literal values of real personal names, company names, or real case numbers.

---

## 5. Automated Compliance Auditing

The project includes an automated regression test: `tests/unit/test_privacy_compliance.py`.
Every CI/CD or `pytest` execution audits `src/`, `tests/`, and `configs/` to ensure no sensitive real entities or identifiers have been introduced.
