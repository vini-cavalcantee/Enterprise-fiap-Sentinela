"""
Página: Risco OLA/KPI

Responde: "A partir do volume previsto de P2 e P3, qual impacto
operacional de OLA/KPI o Sentinela consegue antecipar?"

Fonte: `data/processed/risco_kpi.json`, gerado por
`scripts/build_artifacts.py`. Os ganhos percentuais (-27,8% P3 /
-3,17% P2) são números oficiais do NB04 (constantes citadas, não
recalculadas). Volume histórico, % de entrada em KPI e violações
observadas são agregações diretas do dataset — estatística
descritiva, não modelo, não previsão.
"""
import streamlit as st

from src.config import APP_ICON, APP_TITLE
from src.components import render_sidebar_brand, status_badge
from src.data_loader import load_risco_kpi, risco_kpi_health_check

st.set_page_config(page_title=f"Risco OLA/KPI — {APP_TITLE}", page_icon=APP_ICON, layout="wide")
render_sidebar_brand()

st.title("⚠️ Risco OLA/KPI")
st.caption("A partir do volume de P2 e P3, qual impacto operacional de OLA/KPI o Sentinela consegue antecipar?")

missing = risco_kpi_health_check()
if missing:
    st.error(
        "⚠️ Artefatos ausentes: " + ", ".join(missing)
        + ". Execute `python scripts/build_artifacts.py` antes de abrir esta página."
    )
    st.stop()

risco = load_risco_kpi()
p2, p3 = risco["p2"], risco["p3"]

# ======================================================================
# 1. VISÃO DE RISCO — P2 x P3 lado a lado
# ======================================================================
c2, c3 = st.columns(2)
with c2:
    with st.container(border=True):
        st.markdown("**P2 — Alta prioridade**")
        st.metric("Volume histórico (pós-regime)", f"{p2['volume_historico']:,}".replace(",", "."))
        m1, m2 = st.columns(2)
        m1.metric("Entrou em KPI", f"{p2['pct_entrou_kpi']:.1f}%")
        m2.metric("Violação (condicional)", f"{p2['pct_violacao_condicional']:.2f}%")
        st.caption(f"{p2['n_violacoes_historico']} violações observadas em {p2['n_entrou_kpi']} tickets que entraram em KPI.")
with c3:
    with st.container(border=True):
        st.markdown("**P3 — Média prioridade**")
        st.metric("Volume histórico (pós-regime)", f"{p3['volume_historico']:,}".replace(",", "."))
        m1, m2 = st.columns(2)
        m1.metric("Entrou em KPI", f"{p3['pct_entrou_kpi']:.1f}%")
        m2.metric("Violação (condicional)", f"{p3['pct_violacao_condicional']:.2f}%")
        st.caption(f"{p3['n_violacoes_historico']} violações observadas em {p3['n_entrou_kpi']} tickets que entraram em KPI.")

st.divider()

# ======================================================================
# 2. P3 — PRINCIPAL SINAL (destaque)
# ======================================================================
with st.container(border=True):
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("### 🟡 P3 — principal sinal encontrado")
    with h2:
        st.markdown(status_badge(risco["p3_status"]))

    st.metric("Ganho na projeção de violações vs. baseline", f"-{risco['p3_ganho_pct']:.1f}%")
    st.caption(f"Modelo: {risco['p3_modelo']}")
    st.info(
        "A projeção de P3 apresentou sinal consistente de melhora frente ao baseline, porém a validação "
        "ainda possui amostra curta."
    )

    meses_p3 = p3["entrada_kpi_por_mes"]
    primeiro, ultimo = list(meses_p3.items())[0], list(meses_p3.items())[-1]
    st.warning(
        f"📉 **Sinal de atenção** — a taxa de entrada em KPI de P3 caiu de {primeiro[1]:.1f}% ({primeiro[0]}) "
        f"para {ultimo[1]:.1f}% ({ultimo[0]}) ao longo do período observado. Mudanças nesse padrão afetam "
        "diretamente a projeção de violações e merecem acompanhamento."
    )

# ======================================================================
# 3. P2 — RESULTADO INCONCLUSIVO (secundário)
# ======================================================================
with st.container(border=True):
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("#### P2 — resultado inconclusivo")
    with h2:
        st.markdown(status_badge("Em validação"))

    st.metric("Ganho na projeção de violações vs. baseline", f"-{risco['p2_ganho_pct']:.2f}%")
    st.caption(
        f"Modelo: {risco['p2_modelo']} — o ganho observado é pequeno e ainda não há evidência suficiente "
        "para afirmar vantagem operacional consistente."
    )

st.divider()

# ======================================================================
# 4. DE ONDE VEM O RISCO — fluxo simples
# ======================================================================
st.markdown("#### De onde vem o risco")
f1, f2, f3, f4 = st.columns(4)
f1.markdown(f"**Volume**\n\n{p3['volume_historico']:,}".replace(",", ".") + "\n\n*(P3, histórico)*")
f2.markdown(f"**× Taxa de entrada em KPI**\n\n{p3['pct_entrou_kpi']:.1f}%")
f3.markdown(f"**× Taxa de violação**\n\n{p3['pct_violacao_condicional']:.2f}%")
f4.markdown(f"**= Violações**\n\n≈ {p3['n_violacoes_historico']}")
st.caption("Exemplo com números históricos observados de P3 · violações esperadas = volume × taxa de entrada em KPI × taxa de violação.")

st.divider()

# ======================================================================
# 5. RECOMENDAÇÕES OPERACIONAIS
# ======================================================================
st.markdown("#### Como usar este sinal")
st.markdown(
    "- Usar a projeção de P3 como sinal antecipado para planejamento operacional.\n"
    "- Acompanhar a taxa de entrada em KPI de P3 — houve queda relevante no período observado.\n"
    "- Não usar P2 como alerta automatizado enquanto o ganho permanecer inconclusivo; manter acompanhamento humano."
)
