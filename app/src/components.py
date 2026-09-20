"""Componentes de UI reutilizaveis entre paginas do app Sentinela."""
import streamlit as st

from src.config import APP_ICON, APP_TITLE


def render_sidebar_brand():
    st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.sidebar.caption("Assistente operacional inteligente — Locaweb x FIAP Challenge")
    st.sidebar.divider()


def status_badge(status: str) -> str:
    """Retorna um badge em markdown para o status metodológico de uma
    capacidade do Sentinela.

    - Validado: evidência temporal mais consolidada.
    - Em validação: sinal quantitativo favorável, mas amostra de
      validação ainda limitada (ex.: D+7, P2, P3).
    - Exploratório: análises descritivas/iniciais (ex.: clustering).
    """
    status = status.strip()
    if status.lower() == "validado":
        return "🟢 **Validado**"
    if status.lower() in ("em validação", "em validacao"):
        return "🟡 **Em validação**"
    if status.lower() in ("experimental", "exploratório", "exploratorio"):
        return "🟠 **Exploratório**"
    return f"⚪ {status}"


def em_validacao_note(text: str):
    """Caixa visual padrao para capacidades com sinal favoravel porem
    amostra de validacao ainda limitada (nao usar linguagem que
    prometa melhora futura garantida)."""
    st.info(f"🟡 **Em validação** — {text}")


def cluster_badge(amostra_pequena: bool = False) -> str:
    """Badge especifico para achados de clustering (NB05) — nunca usa
    o rotulo Validado, pois clustering e analise exploratoria/
    descritiva, nao preditiva."""
    if amostra_pequena:
        return "🟠 **Amostra muito pequena**"
    return "🔵 **Padrão identificado — análise exploratória**"


def exploratory_note(text: str):
    """Caixa visual padrao para marcar achados exploratorios (nao
    validados estatisticamente com o mesmo rigor dos modelos oficiais)."""
    st.warning(f"🟠 **Achado exploratório** — {text}")


def validated_note(text: str):
    st.success(f"🟢 **Validado (Sprint 3)** — {text}")


def data_missing_warning(missing: list[str]):
    st.error(
        "⚠️ Artefatos de dados ausentes: "
        + ", ".join(missing)
        + ".\n\nExecute `python scripts/build_artifacts.py` antes de rodar o app."
    )
