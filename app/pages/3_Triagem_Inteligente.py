"""
Página: Triagem Inteligente

Responde: "Quais incidentes têm maior sinal de encerramento sem
intervenção e podem ser priorizados para revisão humana?"

Fonte: pipeline real (Logistic Regression) treinado e serializado por
`scripts/build_artifacts.py` a partir de `src/triage.py` (reconstrução
da metodologia da NB03). O modelo é carregado uma única vez
(`st.cache_resource`) e nunca retreinado a cada clique.
"""
import streamlit as st

from src.config import APP_ICON, APP_TITLE
from src.components import render_sidebar_brand, status_badge
from src.data_loader import load_triage_categorias, load_triage_model, triagem_health_check
from src.triage import DOW_LABELS, OFFICIAL_METRICS, THRESHOLD, build_input_row, make_triage_model, predict_and_explain

st.set_page_config(page_title=f"Triagem Inteligente — {APP_TITLE}", page_icon=APP_ICON, layout="wide")
render_sidebar_brand()

st.title("🧭 Triagem Inteligente")
st.caption("Quais incidentes têm maior sinal de encerramento sem intervenção e podem ser priorizados para revisão humana?")

missing = triagem_health_check()
if missing:
    st.error(
        "⚠️ Artefatos de triagem ausentes: " + ", ".join(missing)
        + ". Execute `python scripts/build_artifacts.py` antes de abrir esta página."
    )
    st.stop()

_model = load_triage_model()
_cat = load_triage_categorias()
triage = make_triage_model(_model, _cat["feature_columns"], _cat["top_produtos"], _cat["top_categorias"])

PRIORIDADES = ["2 - Alta", "3 - Média", "4 - Baixa", "5 - Muito Baixa"]


def _select_options(codigos: list[str]) -> list[str]:
    """Monta as opções de select para Produto/Categoria: referência
    (Não informado), códigos observados no histórico, e um genérico
    'Outro'."""
    opts = ["Não informado"]
    opts += [c for c in codigos if c != "NAO_INFORMADO"]
    opts += ["Outro / não listado"]
    return opts


def _to_raw(display_value: str) -> str:
    if display_value == "Não informado":
        return "NAO_INFORMADO"
    if display_value == "Outro / não listado":
        return "OUTROS"
    return display_value


# ======================================================================
# 1. TRIAGEM INTERATIVA
# ======================================================================
st.markdown("### 🧪 Analisar um incidente")

with st.form("form_triagem"):
    col1, col2 = st.columns(2)
    with col1:
        prioridade = st.selectbox("Prioridade", PRIORIDADES, index=2)
        aberto_por = st.selectbox("Aberto por", ["Monitoramento", "Manual"])
        produto_disp = st.selectbox("Produto", _select_options(triage.top_produtos))
        categoria_disp = st.selectbox("Categoria", _select_options(triage.top_categorias))
    with col2:
        hora = st.slider("Horário de abertura", 0, 23, 12)
        dow_label = st.selectbox("Dia da semana de abertura", DOW_LABELS)
        tem_item_config = st.checkbox("Possui item de configuração vinculado")
        tem_incidente_pai = st.checkbox("Vinculado a um incidente relacionado (pai)")

    submitted = st.form_submit_button("🔍 Analisar incidente", width="stretch")

if submitted:
    dow_idx = DOW_LABELS.index(dow_label)
    row = build_input_row(
        triage,
        prioridade=prioridade,
        produto=_to_raw(produto_disp),
        categoria=_to_raw(categoria_disp),
        aberto_por=aberto_por,
        hora=hora,
        dow=dow_idx,
        tem_item_config=tem_item_config,
        tem_incidente_pai=tem_incidente_pai,
    )
    score, fatores = predict_and_explain(triage, row)
    st.session_state["triagem_resultado"] = {
        "score": score,
        "fatores": fatores,
        "inputs": {
            "Prioridade": prioridade,
            "Origem": aberto_por,
            "Produto": produto_disp,
            "Categoria": categoria_disp,
            "Horário": hora,
            "Dia da semana": dow_label,
            "Item de configuração": tem_item_config,
            "Incidente pai": tem_incidente_pai,
        },
    }

if "triagem_resultado" in st.session_state:
    resultado = st.session_state["triagem_resultado"]
    score = resultado["score"]
    fatores = resultado["fatores"]

    # Valores ATUAIS do formulário (lidos a cada rerun), para detectar
    # se o usuário alterou algum campo sem clicar em "Analisar" de novo.
    inputs_atuais = {
        "Prioridade": prioridade,
        "Origem": aberto_por,
        "Produto": produto_disp,
        "Categoria": categoria_disp,
        "Horário": hora,
        "Dia da semana": dow_label,
        "Item de configuração": tem_item_config,
        "Incidente pai": tem_incidente_pai,
    }
    desatualizado = inputs_atuais != resultado["inputs"]

    st.write("")
    with st.container(border=True):
        if desatualizado:
            st.caption("⚠️ Os campos foram alterados. Clique em \"Analisar incidente\" para atualizar o resultado abaixo.")

        r1, r2 = st.columns([2, 1])
        with r1:
            if score >= THRESHOLD:
                st.success("🔎 **Priorizar revisão de possível \"sem intervenção\"**")
            else:
                st.info("➡️ **Manter na fila normal de tratamento**")
        with r2:
            st.metric("Score de triagem", f"{score:.2f}")

        pct_score = max(0.0, min(1.0, score))
        st.progress(pct_score, text=f"Score {score:.2f}  ·  limiar operacional {THRESHOLD:.2f}")
        st.caption(f"Threshold operacional: {THRESHOLD:.2f}")

        st.caption(
            "ℹ️ O Sentinela apoia a priorização. O encerramento ou descarte do incidente permanece sob decisão humana."
        )

        inp = resultado["inputs"]
        st.caption(
            f"Registro analisado: Prioridade {inp['Prioridade']} · Origem {inp['Origem']} · "
            f"Produto {inp['Produto']} · Categoria {inp['Categoria']} · {inp['Horário']}h ({inp['Dia da semana']}) · "
            f"Item de configuração: {'Sim' if inp['Item de configuração'] else 'Não'} · "
            f"Incidente pai: {'Sim' if inp['Incidente pai'] else 'Não'}"
        )

        if fatores:
            st.markdown("**Principais sinais considerados**")
            for f in fatores:
                seta = "↑" if f["direcao"] == "aumenta" else "↓"
                verbo = "aumenta" if f["direcao"] == "aumenta" else "reduz"
                st.markdown(f"- {seta} {f['label']} — {verbo} o sinal de \"sem intervenção\"")

st.divider()

# ======================================================================
# 2. DESEMPENHO VALIDADO
# ======================================================================
st.markdown("### 📐 Desempenho validado")
st.markdown("**Regressão Logística (Logistic Regression)**")

d1, d2, d3 = st.columns(3)
d1.metric("Precisão", f"{OFFICIAL_METRICS['precisao_pct']:.1f}%")
d2.metric("Cobertura", f"{OFFICIAL_METRICS['cobertura_pct']:.1f}%")
d3.metric("Redução de falsos positivos", f"~{OFFICIAL_METRICS['reducao_fp_pct']:.0f}%")

st.markdown(status_badge(OFFICIAL_METRICS["status"]))
st.caption(
    "O modelo prioriza precisão para reduzir o risco de sinalizar indevidamente incidentes que ainda "
    "exigem atuação humana."
)

st.divider()

# ======================================================================
# 3. CONTEXTO OPERACIONAL
# ======================================================================
st.info(
    "🔎 **Monitoramento** apareceu independentemente no clustering e no modelo de triagem como um dos "
    "principais sinais associados a incidentes sem intervenção."
)
