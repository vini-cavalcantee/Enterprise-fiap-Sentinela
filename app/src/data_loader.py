"""
Camada de acesso a dados do app Sentinela.

Principio: o app NUNCA le o LWDATASET.xlsx bruto nem treina/recalcula
modelos em tempo de execucao. Ele apenas le os artefatos ja
processados em `data/processed/`, gerados por
`scripts/build_artifacts.py` a partir dos notebooks NB01-NB05.

Todas as funcoes usam @st.cache_data (dados) ou @st.cache_resource
(modelos/objetos nao triviais de serializar) para evitar releitura ou
recarregamento a cada interacao do usuario.
"""
import json

import joblib
import pandas as pd
import streamlit as st

from src.config import DATA_DIR


@st.cache_data(show_spinner=False)
def load_serie_diaria() -> pd.DataFrame:
    path = DATA_DIR / "serie_diaria.csv"
    df = pd.read_csv(path, parse_dates=["data"])
    return df


@st.cache_data(show_spinner=False)
def load_kpis_executivos() -> dict:
    path = DATA_DIR / "kpis_executivos.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_top_categorias() -> pd.DataFrame:
    path = DATA_DIR / "top_categorias.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_perfil_clusters() -> pd.DataFrame:
    path = DATA_DIR / "perfil_clusters.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_previsao_d1() -> dict:
    path = DATA_DIR / "previsao_d1.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_walkforward_d1() -> pd.DataFrame:
    path = DATA_DIR / "walkforward_d1.csv"
    return pd.read_csv(path, parse_dates=["data"])


@st.cache_data(show_spinner=False)
def load_previsao_d7() -> pd.DataFrame:
    path = DATA_DIR / "previsao_d7.csv"
    return pd.read_csv(path, parse_dates=["data"])


@st.cache_data(show_spinner=False)
def load_previsao_p2p3() -> dict:
    path = DATA_DIR / "previsao_p2p3.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_risco_kpi() -> dict:
    path = DATA_DIR / "risco_kpi.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
@st.cache_resource(show_spinner="Carregando modelo de triagem...")
def load_triage_model():
    """Carrega o modelo (LogisticRegression) ja treinado e serializado
    por build_artifacts.py. cache_resource garante que o objeto e
    carregado uma unica vez por sessao do servidor, nunca retreinado."""
    path = DATA_DIR / "triage_model.joblib"
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_triage_categorias() -> dict:
    path = DATA_DIR / "triage_categorias.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def data_health_check() -> list[str]:
    """Retorna uma lista de artefatos ausentes (usado para alertar o
    usuario na interface em vez de quebrar a aplicacao)."""
    required = [
        "serie_diaria.csv",
        "kpis_executivos.json",
        "top_categorias.csv",
        "perfil_clusters.csv",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    return missing


def previsao_health_check() -> list[str]:
    """Health check especifico dos artefatos da pagina Previsao."""
    required = ["previsao_d1.json", "walkforward_d1.csv", "previsao_d7.csv", "previsao_p2p3.json"]
    return [name for name in required if not (DATA_DIR / name).exists()]


def triagem_health_check() -> list[str]:
    """Health check especifico dos artefatos da pagina Triagem Inteligente."""
    required = ["triage_model.joblib", "triage_categorias.json"]
    return [name for name in required if not (DATA_DIR / name).exists()]


def risco_kpi_health_check() -> list[str]:
    """Health check especifico dos artefatos da pagina Risco OLA/KPI."""
    required = ["risco_kpi.json"]
    return [name for name in required if not (DATA_DIR / name).exists()]


def padroes_health_check() -> list[str]:
    """Health check especifico dos artefatos da pagina Padroes e Explicabilidade."""
    required = [
        "perfil_clusters.csv",
        "figures/03_share_volume_cluster.png",
        "figures/05_shap_d1_bar.png",
        "figures/06_coef_triagem.png",
    ]
    return [name for name in required if not (DATA_DIR / name).exists()]
