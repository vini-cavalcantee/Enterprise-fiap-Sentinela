"""
Página: Visão Executiva

Responde: "Qual é a situação geral da operação e quais são os
principais sinais encontrados pelo Sentinela?"

Fonte de dados: artefatos processados em `data/processed/`, gerados a
partir dos notebooks NB01 (série diária / regime) e NB05 (clusters).
Nenhum modelo é treinado ou recalculado nesta página.
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_ICON, APP_TITLE, COLOR_ACCENT, COLOR_PRIMARY
from src.components import render_sidebar_brand, status_badge, exploratory_note, data_missing_warning
from src.data_loader import (
    data_health_check,
    load_kpis_executivos,
    load_perfil_clusters,
    load_serie_diaria,
    load_top_categorias,
)

st.set_page_config(page_title=f"Visão Executiva — {APP_TITLE}", page_icon=APP_ICON, layout="wide")
render_sidebar_brand()

st.title("📊 Visão Executiva")
st.caption("Qual é a situação geral da operação e quais são os principais sinais encontrados pelo Sentinela?")

missing = data_health_check()
if missing:
    data_missing_warning(missing)
    st.stop()

kpis = load_kpis_executivos()
serie = load_serie_diaria()
top_cat = load_top_categorias()
clusters = load_perfil_clusters()

st.caption(
    f"Período analisado: **{kpis['periodo_inicio']}** a **{kpis['periodo_fim']}** "
    f"({kpis['dias_periodo']} dias, pós mudança de regime operacional identificada na NB01)."
)

tab_panorama, tab_sinais = st.tabs(
    ["📌 Panorama Operacional", "🧩 Sinais Encontrados (Clustering)"]
)

# ======================================================================
# TAB 1 — PANORAMA OPERACIONAL
# ======================================================================
with tab_panorama:
    st.markdown("#### Indicadores-chave do período")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume total no período", f"{kpis['volume_total_periodo']:,}".replace(",", "."))
    c2.metric("Volume médio diário", f"{kpis['volume_medio_diario']:.0f}/dia")
    c3.metric(
        "Crescimento vs. baseline (jan–ago/25)",
        f"+{kpis['crescimento_pct_vs_baseline']:.1f}%",
        help=f"Baseline: {kpis['volume_medio_diario_baseline']:.0f} incidentes/dia em média (jan–ago/2025).",
    )
    c4.metric("Encerrados sem intervenção", f"{kpis['pct_sem_intervencao']:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Origem: Monitoramento", f"{kpis['pct_monitoramento']:.1f}%")
    c6.metric("Origem: Manual", f"{kpis['pct_manual']:.1f}%")
    c7.metric("Concentração no Team14", f"{kpis['pct_team14']:.1f}%")
    c8.metric("Sem Produto/Categoria", f"{kpis['pct_sem_categoria_produto']:.1f}%")

    st.divider()

    st.markdown("#### Evolução do volume diário de incidentes")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=serie["data"], y=serie["volume_total"], mode="lines", name="Volume diário",
        line=dict(color="#B0B7C3", width=1),
    ))
    fig_trend.add_trace(go.Scatter(
        x=serie["data"], y=serie["media_movel_7d"], mode="lines", name="Média móvel 7 dias",
        line=dict(color=COLOR_PRIMARY, width=3),
    ))
    fig_trend.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title="Incidentes/dia",
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, width='stretch')

    st.markdown("#### Volume diário por prioridade crítica (P2 / P3)")
    fig_pri = go.Figure()
    fig_pri.add_trace(go.Scatter(x=serie["data"], y=serie["volume_p2"], mode="lines", name="P2 — Alta", line=dict(color="#D64545")))
    fig_pri.add_trace(go.Scatter(x=serie["data"], y=serie["volume_p3"], mode="lines", name="P3 — Média", line=dict(color=COLOR_ACCENT)))
    fig_pri.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Incidentes/dia", hovermode="x unified",
    )
    st.plotly_chart(fig_pri, width='stretch')
    st.caption(
        f"P2 representa {kpis['pct_p2']:.1f}% do volume do período e P3 representa {kpis['pct_p3']:.1f}% "
        "— são as únicas prioridades monitoradas para efeito de KPI, conforme dicionário de dados."
    )

    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Composição por status")
        fig_status = px.pie(
            values=[kpis["pct_sem_intervencao"], 100 - kpis["pct_sem_intervencao"]],
            names=["Sem intervenção", "Com intervenção/outros"],
            color_discrete_sequence=[COLOR_PRIMARY, "#D9D9D9"],
            hole=0.55,
        )
        fig_status.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig_status, width='stretch')

    with col_b:
        st.markdown("#### Composição por origem")
        fig_origem = px.pie(
            values=[kpis["pct_monitoramento"], kpis["pct_manual"]],
            names=["Monitoramento", "Manual"],
            color_discrete_sequence=[COLOR_ACCENT, "#D9D9D9"],
            hole=0.55,
        )
        fig_origem.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig_origem, width='stretch')

    st.divider()

    st.markdown("#### Categorias com maior volume de trabalho operacional real")
    st.caption(
        f"Exclui tickets sem Produto/Categoria informado ({kpis['pct_sem_categoria_produto']:.1f}% do volume — "
        "tratados separadamente como ruído de monitoramento na aba de Sinais Encontrados)."
    )
    fig_cat = px.bar(
        top_cat.sort_values("volume"),
        x="volume", y="Categoria", orientation="h",
        color_discrete_sequence=[COLOR_PRIMARY],
        labels={"volume": "Volume no período", "Categoria": ""},
    )
    fig_cat.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cat, width='stretch')

# ======================================================================
# TAB 2 — SINAIS ENCONTRADOS (CLUSTERING, NB05)
# ======================================================================
with tab_sinais:
    st.markdown("#### Agrupamentos operacionais identificados (NB05)")
    st.caption(
        "Clusterização de combinações Produto × Categoria × Prioridade × Origem, "
        "por comportamento observado (não pela identidade da combinação). "
        "Análise **exploratória/descritiva** — não é um modelo preditivo."
    )

    fig_clusters = px.bar(
        clusters.sort_values("share_volume_%"),
        x="share_volume_%", y="label", orientation="h",
        color="exploratorio",
        color_discrete_map={False: COLOR_PRIMARY, True: "#D97706"},
        labels={"share_volume_%": "% do volume pós-regime", "label": ""},
        text="share_volume_%",
    )
    fig_clusters.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_clusters.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_clusters, width='stretch')

    st.markdown("#### Leitura de negócio por cluster")
    for _, row in clusters.sort_values("share_volume_%", ascending=False).iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Cluster {int(row['cluster'])} — {row['label']}**")
            with col2:
                st.markdown(
                    status_badge("Exploratório" if row["exploratorio"] else "Validado (NB05)")
                )
            st.markdown(
                f"- **{row['share_volume_%']:.1f}%** do volume pós-regime · "
                f"{int(row['n_combos'])} combinações\n"
                f"- {row['pct_sem_intervencao']*100:.1f}% sem intervenção · "
                f"{row['pct_contorno']*100:.1f}% contorno · "
                f"{row['pct_definitiva']*100:.1f}% resolução definitiva\n"
                f"- {row['pct_entrou_kpi']*100:.1f}% entra em KPI · "
                f"{row['pct_kpi_violado_cond']*100:.2f}% violação condicional de KPI"
            )
            if row["exploratorio"]:
                exploratory_note(
                    "amostra pequena (apenas 3 combinações) — maior taxa de violação de KPI "
                    "observada, mas ainda não é uma conclusão consolidada."
                )

    st.divider()
    st.info(
        "🔎 **Achado que conecta clustering e explicabilidade (NB05):** a origem do ticket "
        "(Monitoramento) e a ausência de Produto/Categoria informado são, de forma independente, "
        "os principais indicadores de ruído — confirmado tanto pelo clustering quanto pelo SHAP "
        "do modelo de Triagem. Detalhes completos na página *Padrões e Explicabilidade*."
    )
