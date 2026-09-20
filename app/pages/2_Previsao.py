"""
Página: Previsão

Responde: "Qual volume de incidentes o Sentinela prevê e qual é o
nível de evidência dessa previsão?"

Fonte de dados: artefatos processados em `data/processed/`, gerados
por `scripts/build_artifacts.py` a partir de `src/forecasting.py`
(reconstrução da metodologia da NB02 — RandomForest para D+1, Ridge
multi-horizonte para D+7). Nenhum modelo é treinado nesta página; os
modelos rodam apenas no pipeline de geração de artefatos.
"""
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_ICON, APP_TITLE, COLOR_WARNING
from src.components import em_validacao_note, render_sidebar_brand, status_badge
from src.data_loader import (
    load_kpis_executivos,
    load_previsao_d1,
    load_previsao_d7,
    load_previsao_p2p3,
    previsao_health_check,
)

st.set_page_config(page_title=f"Previsão — {APP_TITLE}", page_icon=APP_ICON, layout="wide")
render_sidebar_brand()

st.title("📈 Previsão de Volume")
st.caption("Qual volume de incidentes o Sentinela prevê e qual é o nível de evidência dessa previsão?")

missing = previsao_health_check()
if missing:
    st.error(
        "⚠️ Artefatos de previsão ausentes: " + ", ".join(missing)
        + ". Execute `python scripts/build_artifacts.py` antes de abrir esta página."
    )
    st.stop()

kpis = load_kpis_executivos()
previsao_d1 = load_previsao_d1()
previsao_d7 = load_previsao_d7()
p2p3 = load_previsao_p2p3()

# ======================================================================
# 1. PREVISÃO D+1 — DESTAQUE PRINCIPAL
# ======================================================================
with st.container(border=True):
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown("### 🎯 Previsão para o próximo dia (D+1)")
    with top_r:
        st.markdown(status_badge(previsao_d1["status"]))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Volume previsto — {previsao_d1['data_prevista']}",
        f"{previsao_d1['volume_previsto']:.0f}",
        delta=f"{previsao_d1['volume_previsto'] - previsao_d1['ultimo_valor_real']:+.0f} vs. último dia real",
    )
    c2.metric(
        f"Último dia real ({previsao_d1['ultimo_dia']})",
        f"{previsao_d1['ultimo_valor_real']:.0f}",
    )
    c3.metric(
        "Média móvel 7 dias (recente)",
        f"{previsao_d1['media_movel_7d_recente']:.0f}",
    )

    st.caption(
        f"Modelo: **{previsao_d1['modelo']}** · previsão gerada a partir do histórico disponível até "
        f"{previsao_d1['ultimo_dia']} (nenhuma informação futura é usada nas features)."
    )

st.write("")

# ======================================================================
# 2. EVIDÊNCIA DE DESEMPENHO DO MODELO (D+1)
# ======================================================================
st.markdown("### 📐 Evidência de desempenho do modelo")

m1, m2, m3 = st.columns(3)
m1.metric("Modelo", kpis["modelo_d1_algoritmo"])
m2.metric(
    "Baseline de referência", "Média móvel 7d",
    help="Baseline = previsão simples usando a média dos últimos 7 dias. É a referência mais forte disponível para comparar o modelo.",
)
m3.metric(
    "Ganho (MAE)",
    f"-{kpis['modelo_d1_mae_reducao_pct']:.1f}%",
    help="MAE = Erro Médio Absoluto. Quanto menor, melhor. -8,2% significa que o RandomForest erra, em média, 8,2% menos que o baseline.",
)

st.success(
    f"🟢 **Validado (Sprint 3)** — Na validação temporal, o Sentinela reduziu o erro médio absoluto em "
    f"{kpis['modelo_d1_mae_reducao_pct']:.1f}% frente à referência simples, considerando 91 previsões "
    "distribuídas em três janelas."
)

st.divider()

# ======================================================================
# 3. PREVISÃO D+7 — EM VALIDAÇÃO
# ======================================================================
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown("### 🔭 Previsão estendida (D+7)")
with head_r:
    st.markdown(status_badge(kpis["modelo_d7_status"]))

st.caption("Ridge multi-horizonte — sete modelos independentes, um para cada dia entre D+1 e D+7.")

fig_d7 = go.Figure()
fig_d7.add_trace(go.Bar(
    x=[f"D+{h} ({d.strftime('%d/%m')})" for h, d in zip(previsao_d7["horizonte"], previsao_d7["data"])],
    y=previsao_d7["volume_previsto"],
    marker_color=COLOR_WARNING,
    text=previsao_d7["volume_previsto"].round(0).astype(int),
    textposition="outside",
))
fig_d7.update_layout(
    height=300, margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Volume previsto", xaxis_title=None,
)
st.plotly_chart(fig_d7, width="stretch")

em_validacao_note(
    "O modelo apresentou sinal favorável frente ao baseline, porém a amostra de validação ainda é "
    "limitada (12 avaliações semanais independentes). Com a entrada de novos dados, será possível "
    "verificar se esse ganho se mantém de forma consistente."
)

st.divider()

# ======================================================================
# 4. P2/P3 — BLOCO COMPACTO
# ======================================================================
st.markdown("#### Previsões por prioridade")
pc1, pc2 = st.columns(2)
with pc1:
    with st.container(border=True):
        st.markdown(f"**P2 — {p2p3['p2_modelo']}** {status_badge(p2p3['p2_status'])}")
        st.caption(f"MAE {p2p3['p2_mae_modelo']} vs. baseline {p2p3['p2_mae_baseline']} → ganho de {p2p3['p2_ganho_pct']:.1f}%")
with pc2:
    with st.container(border=True):
        st.markdown(f"**P3 — {p2p3['p3_modelo']}** {status_badge(p2p3['p3_status'])}")
        st.caption(f"MAE {p2p3['p3_mae_modelo']} vs. baseline {p2p3['p3_mae_baseline']} → ganho de {p2p3['p3_ganho_pct']:.1f}%")

em_validacao_note("Sinal favorável frente ao baseline, mas amostra de validação ainda limitada.")

st.caption("Detalhamento completo de P2/P3 e risco de OLA/KPI: página *Risco OLA/KPI* (próxima etapa).")
