"""
Sentinela — App Streamlit (Sprint 4 MVP)

Ponto de entrada da aplicacao multi-pagina. As paginas reais ficam em
`pages/`; este arquivo cuida apenas da configuracao global e da tela
inicial de boas-vindas/navegacao.

Como executar:
    streamlit run app.py
"""
import streamlit as st

from src.config import APP_ICON, APP_TITLE
from src.components import render_sidebar_brand
from src.data_loader import data_health_check

st.set_page_config(
    page_title=f"{APP_TITLE} — Locaweb x FIAP",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar_brand()

st.title(f"{APP_ICON} Sentinela")
st.subheader("Do ruído ao sinal — operação preditiva para a Locaweb")

st.markdown(
    """
Este aplicativo reúne, em um único painel, os resultados analíticos e
preditivos desenvolvidos ao longo do projeto **Sentinela**
(FIAP Enterprise Challenge + Locaweb).

Use o menu à esquerda para navegar entre as páginas:
"""
)

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_Visao_Executiva.py", label="📊 Visão Executiva", icon="📊")
    st.caption("Situação geral da operação e principais sinais encontrados.")
    st.page_link("pages/2_Previsao.py", label="📈 Previsão", icon="📈")
    st.caption("Previsão de volume D+1 e D+7.")
with col2:
    st.page_link("pages/3_Triagem_Inteligente.py", label="🧭 Triagem Inteligente", icon="🧭")
    st.caption("Priorização de tickets prováveis de fechar sem intervenção.")
    st.page_link("pages/4_Risco_OLA_KPI.py", label="⚠️ Risco OLA/KPI", icon="⚠️")
    st.caption("Projeção de risco de violação de metas.")

st.divider()

missing = data_health_check()
if missing:
    st.error(
        "⚠️ Artefatos de dados ausentes: " + ", ".join(missing)
        + ". Execute `python scripts/build_artifacts.py` antes de navegar pelas páginas."
    )
else:
    st.success("✅ Artefatos de dados carregados e prontos.")
