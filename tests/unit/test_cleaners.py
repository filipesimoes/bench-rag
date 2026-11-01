"""Unit tests for DocumentCleaner and HeaderFooterCleaner."""

from pathlib import Path

from bench_rag.cleaners import HeaderFooterCleaner, get_cleaner
from bench_rag.config import PipelineConfig
from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.parser import Document
from bench_rag.data_gen.seed import create_pdf_report
from bench_rag.parsers import PyMuPDFParser


def test_cleaner_protocol_implementation():
    cleaner = HeaderFooterCleaner()
    assert isinstance(cleaner, DocumentCleaner)


def test_get_cleaner_factory():
    cfg_disabled = PipelineConfig(clean_headers_footers=False)
    assert get_cleaner(cfg_disabled) is None

    cfg_enabled = PipelineConfig(
        clean_headers_footers=True,
        header_margin=45.0,
        footer_margin=35.0,
        repeated_header_threshold=0.25,
    )
    cleaner = get_cleaner(cfg_enabled)
    assert isinstance(cleaner, HeaderFooterCleaner)
    assert cleaner.header_margin == 45.0
    assert cleaner.footer_margin == 35.0
    assert cleaner.repeated_header_threshold == 0.25


def test_pje_signature_extraction():
    cleaner = HeaderFooterCleaner()
    footer_text = (
        "Num. 9999999999 - Pág. 1\n"
        "Assinado eletronicamente por: SERVIDOR TESTE - 01/01/2025 10:00:00\n"
        "https://pje1g.trf1.jus.br:443/pje/Processo/ConsultaDocumento/listView.seam?x=00000000000000000000000000000\n"
        "Número do documento: 00000000000000000000000000000"
    )
    signatures = cleaner.extract_pje_signatures(footer_text)
    assert len(signatures) == 1
    assert signatures[0]["signatory"] == "SERVIDOR TESTE"
    assert signatures[0]["datetime"] == "01/01/2025 10:00:00"
    assert signatures[0]["document_number"] == "00000000000000000000000000000"


def test_is_pje_or_pagination_noise():
    cleaner = HeaderFooterCleaner()
    assert cleaner.is_pje_or_pagination_noise("Documento id 12345 - Inquérito policial") is True
    assert (
        cleaner.is_pje_or_pagination_noise(
            "Num. 123 - Pág. 2\nAssinado eletronicamente por: Delegado X"
        )
        is True
    )
    assert cleaner.is_pje_or_pagination_noise("Justiça Federal da 1ª Região") is True
    assert (
        cleaner.is_pje_or_pagination_noise(
            "https://pje1g.trf1.jus.br/pje/Processo/ConsultaDocumento"
        )
        is True
    )
    assert cleaner.is_pje_or_pagination_noise("1") is True
    assert cleaner.is_pje_or_pagination_noise("Página 4 de 10") is True
    assert cleaner.is_pje_or_pagination_noise("2\n999/2024") is True
    assert (
        cleaner.is_pje_or_pagination_noise("Relatório de Investigação Policial sobre Fraude")
        is False
    )


def test_clean_page_blocks_margin_and_repetition():
    cleaner = HeaderFooterCleaner(
        header_margin=50.0,
        footer_margin=40.0,
        detect_repeated_headers=True,
        repeated_header_threshold=0.3,
        preserve_first_header=True,
    )

    page_dims = [(595.0, 842.0), (595.0, 842.0), (595.0, 842.0)]
    pages_blocks = [
        # Page 1
        [
            (
                36.0,
                30.0,
                400.0,
                45.0,
                "Documento id 99999 - Processo Eletrônico",
            ),  # y1 <= 50 (margin)
            (100.0, 120.0, 400.0, 140.0, "Ministério da Justiça e Segurança Pública"),  # repeated
            (50.0, 200.0, 500.0, 300.0, "Conteúdo da Capa do Processo"),  # body
            (
                60.0,
                810.0,
                500.0,
                830.0,
                "Num. 999 - Pág. 1\nAssinado eletronicamente por: AGENTE A - 01/01/2025 10:00:00\nNúmero do documento: 12345",
            ),  # y0 >= 802
        ],
        # Page 2
        [
            (36.0, 30.0, 400.0, 45.0, "Documento id 99999 - Processo Eletrônico"),
            (100.0, 120.0, 400.0, 140.0, "Ministério da Justiça e Segurança Pública"),
            (50.0, 200.0, 500.0, 300.0, "Depoimento da Testemunha 1"),
            (
                60.0,
                810.0,
                500.0,
                830.0,
                "Num. 999 - Pág. 2\nAssinado eletronicamente por: AGENTE A - 01/01/2025 10:00:00\nNúmero do documento: 12345",
            ),
        ],
        # Page 3
        [
            (36.0, 30.0, 400.0, 45.0, "Documento id 99999 - Processo Eletrônico"),
            (100.0, 120.0, 400.0, 140.0, "Ministério da Justiça e Segurança Pública"),
            (50.0, 200.0, 500.0, 300.0, "Conclusões do Laudo Pericial"),
            (
                60.0,
                810.0,
                500.0,
                830.0,
                "Num. 999 - Pág. 3\nAssinado eletronicamente por: AGENTE A - 01/01/2025 10:00:00\nNúmero do documento: 12345",
            ),
        ],
    ]

    cleaned_pages, metadata = cleaner.clean_page_blocks(pages_blocks, page_dims)

    assert len(cleaned_pages) == 3
    assert metadata["cleaned_headers_footers"] is True
    assert len(metadata["pje_signatures"]) == 1
    assert metadata["pje_signatures"][0]["signatory"] == "AGENTE A"
    assert metadata["characters_removed"] > 0

    # Page 1 should preserve the first occurrence of "Ministério da Justiça"
    assert "Ministério da Justiça e Segurança Pública" in cleaned_pages[0]
    assert "Conteúdo da Capa do Processo" in cleaned_pages[0]
    assert "Documento id" not in cleaned_pages[0]
    assert "Assinado eletronicamente" not in cleaned_pages[0]

    # Page 2 and 3 should NOT contain "Ministério da Justiça e Segurança Pública"
    assert "Ministério da Justiça e Segurança Pública" not in cleaned_pages[1]
    assert "Depoimento da Testemunha 1" in cleaned_pages[1]
    assert "Ministério da Justiça e Segurança Pública" not in cleaned_pages[2]
    assert "Conclusões do Laudo Pericial" in cleaned_pages[2]


def test_clean_document_instance():
    cleaner = HeaderFooterCleaner(preserve_first_header=True)
    doc = Document(
        id="DOC_TEST",
        text="",
        metadata={
            "pages_text": [
                "Ministério da Justiça\nDocumento id 12345\nLinha de Corpo 1\nNum. 123 - Pág. 1",
                "Ministério da Justiça\nDocumento id 12345\nLinha de Corpo 2\nNum. 123 - Pág. 2",
            ]
        },
    )
    cleaned_docs = cleaner.clean([doc])
    assert len(cleaned_docs) == 1
    c_doc = cleaned_docs[0]
    assert c_doc.id == "DOC_TEST"
    assert c_doc.metadata["cleaned_headers_footers"] is True
    assert "Linha de Corpo 1" in c_doc.text
    assert "Linha de Corpo 2" in c_doc.text
    # First occurrence preserved, second removed
    assert c_doc.text.count("Ministério da Justiça") == 1
    assert "Documento id" not in c_doc.text
    assert "Pág. 1" not in c_doc.text


def test_pymupdf_parser_with_cleaner(tmp_path: Path):
    pdf_path = tmp_path / "IP_TEST_CLEAN.pdf"
    create_pdf_report(
        file_path=pdf_path,
        title="Inquérito Policial Limpeza",
        inquiry_id="IP-001/2025",
        sections=[
            {"title": "Fatos", "content": "Descrição dos fatos periciados."},
        ],
    )

    # Parser without cleaner
    parser_no_clean = PyMuPDFParser(clean_headers_footers=False)
    docs_raw = parser_no_clean.parse(pdf_path)
    assert docs_raw[0].metadata["cleaned_headers_footers"] is False

    # Parser with cleaner
    parser_clean = PyMuPDFParser(clean_headers_footers=True)
    docs_clean = parser_clean.parse(pdf_path)
    assert docs_clean[0].metadata["cleaned_headers_footers"] is True
    assert "FATOS" in docs_clean[0].text
