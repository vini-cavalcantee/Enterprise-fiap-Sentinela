"""
Modulo de Triagem Inteligente do Sentinela — reconstrucao da
metodologia descrita para a NB03 (Logistic Regression, threshold 0,70,
sem uso de `Grupo designado` ou qualquer campo pos-fechamento).

IMPORTANTE — mesma ressalva já aplicada em `src/forecasting.py`: o
notebook NB03 original nao esta disponivel neste ambiente. O modelo
aqui e a MESMA reconstrucao ja usada e apresentada na NB05 (secao de
explicabilidade) — nao um modelo novo criado so para esta pagina. Os
numeros OFICIAIS da Sprint 3 (precisao 95,9% / cobertura 61,5% /
reducao de falsos positivos ~49% / threshold 0,70) sao constantes
citadas, nunca recalculadas a partir deste pipeline.

O pipeline e treinado UMA VEZ em `scripts/build_artifacts.py` e
serializado em disco (joblib). A pagina apenas carrega o artefato e
roda inferencia — nunca retreina a cada clique.

Features utilizadas (todas disponiveis no momento da ABERTURA do
incidente, sem leakage):
- Prioridade
- Produto (top-12 mais frequentes + "OUTROS" + "NAO_INFORMADO" como
  categoria de referencia)
- Categoria (mesma logica de Produto)
- Aberto por (Monitoramento / Manual)
- Hora de abertura, dia da semana de abertura
- Possui item de configuracao vinculado (sim/nao)
- Esta vinculado a um incidente relacionado / pai (sim/nao)

Explicitamente NAO utilizado: Grupo designado, Duracao, Status,
Solucao, Codigo de fechamento, Entrou para KPI / KPI Violado — todos
so existem ou fazem sentido apos a abertura/fechamento do incidente.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REGIME_START = pd.Timestamp("2025-09-01")
THRESHOLD = 0.70
TOPN = 12

CAT_COLS = ["Prioridade", "Produto_r", "Categoria_r", "Aberto por"]
NUM_COLS = ["hora_abertura", "dow_abertura", "tem_item_config", "tem_incidente_pai"]

DOW_LABELS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

# Numeros OFICIAIS da Sprint 3 (NB03) - constantes citadas, nunca recalculadas.
OFFICIAL_METRICS = {
    "modelo": "Logistic Regression",
    "threshold": THRESHOLD,
    "precisao_pct": 95.9,
    "cobertura_pct": 61.5,
    "reducao_fp_pct": 49.0,
    "status": "Validado",
}


@dataclass
class TriageModel:
    model: LogisticRegression
    feature_columns: list
    top_produtos: list
    top_categorias: list


def make_triage_model(model, feature_columns: list, top_produtos: list, top_categorias: list) -> "TriageModel":
    """Reconstroi o objeto TriageModel a partir do modelo serializado
    (joblib) + metadados de categorias (json) — usado pela pagina, que
    nunca retreina o modelo."""
    return TriageModel(model=model, feature_columns=feature_columns, top_produtos=top_produtos, top_categorias=top_categorias)


def _prepare_base(df_raw: pd.DataFrame) -> pd.DataFrame:
    dft = df_raw[df_raw["Aberto"] >= REGIME_START].copy()
    dft["sem_intervencao"] = (dft["Status"] == "Sem Intervenção").astype(int)
    dft["Produto_g"] = dft["Produto"].fillna("NAO_INFORMADO")
    dft["Categoria_g"] = dft["Categoria"].fillna("NAO_INFORMADO")
    dft["hora_abertura"] = dft["Aberto"].dt.hour
    dft["dow_abertura"] = dft["Aberto"].dt.dayofweek
    dft["tem_item_config"] = dft["Item de configuração"].notna().astype(int)
    dft["tem_incidente_pai"] = dft["Incidente Pai"].notna().astype(int)
    return dft


def build_training_frame(df_raw: pd.DataFrame):
    dft = _prepare_base(df_raw)

    top_prod = dft["Produto_g"].value_counts().head(TOPN).index.tolist()
    top_cat = dft["Categoria_g"].value_counts().head(TOPN).index.tolist()
    dft["Produto_r"] = np.where(dft["Produto_g"].isin(top_prod), dft["Produto_g"], "OUTROS")
    dft["Categoria_r"] = np.where(dft["Categoria_g"].isin(top_cat), dft["Categoria_g"], "OUTROS")

    X = pd.get_dummies(dft[CAT_COLS], drop_first=True)
    X[NUM_COLS] = dft[NUM_COLS].values
    y = dft["sem_intervencao"]
    return X, y, top_prod, top_cat


def train_triage_model(df_raw: pd.DataFrame) -> TriageModel:
    X, y, top_prod, top_cat = build_training_frame(df_raw)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return TriageModel(model=model, feature_columns=list(X.columns), top_produtos=top_prod, top_categorias=top_cat)


def build_input_row(
    triage: TriageModel,
    prioridade: str,
    produto: str,
    categoria: str,
    aberto_por: str,
    hora: int,
    dow: int,
    tem_item_config: bool,
    tem_incidente_pai: bool,
) -> pd.DataFrame:
    """Monta uma linha de entrada com EXATAMENTE as mesmas colunas
    (dummies) vistas no treino. Categorias de referencia (ex.: Produto
    'NAO_INFORMADO') simplesmente nao ativam nenhum dummy — o que e o
    comportamento correto de um one-hot com drop_first."""
    row = pd.DataFrame(np.zeros((1, len(triage.feature_columns))), columns=triage.feature_columns)

    def set_dummy(prefix: str, value: str):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row.loc[0, col] = 1.0

    set_dummy("Prioridade", prioridade)
    set_dummy("Produto_r", produto)
    set_dummy("Categoria_r", categoria)
    set_dummy("Aberto por", aberto_por)

    row.loc[0, "hora_abertura"] = hora
    row.loc[0, "dow_abertura"] = dow
    row.loc[0, "tem_item_config"] = int(tem_item_config)
    row.loc[0, "tem_incidente_pai"] = int(tem_incidente_pai)
    return row


def _friendly_label(feature: str, value: float) -> str:
    if feature.startswith("Prioridade_"):
        return f"Prioridade: {feature.replace('Prioridade_', '')}"
    if feature.startswith("Produto_r_"):
        prod = feature.replace("Produto_r_", "")
        return "Produto: não listado (Outros)" if prod == "OUTROS" else f"Produto identificado: {prod}"
    if feature.startswith("Categoria_r_"):
        cat = feature.replace("Categoria_r_", "")
        return "Categoria: não listada (Outros)" if cat == "OUTROS" else f"Categoria identificada: {cat}"
    if feature.startswith("Aberto por_"):
        return f"Origem: {feature.replace('Aberto por_', '')}"
    if feature == "hora_abertura":
        return f"Horário de abertura: {int(value)}h"
    if feature == "dow_abertura":
        idx = int(value) % 7
        return f"Dia da semana: {DOW_LABELS[idx]}"
    if feature == "tem_item_config":
        return "Possui item de configuração vinculado" if value else "Sem item de configuração vinculado"
    if feature == "tem_incidente_pai":
        return "Vinculado a incidente relacionado (pai)" if value else "Sem vínculo com incidente pai"
    return feature


def predict_and_explain(triage: TriageModel, row: pd.DataFrame, top_k: int = 4):
    score = float(triage.model.predict_proba(row[triage.feature_columns])[0, 1])

    coefs = pd.Series(triage.model.coef_[0], index=triage.feature_columns)
    values = row.iloc[0]
    contrib = coefs * values
    ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_k)

    fatores = []
    for feat, val in ranked.items():
        if val == 0:
            continue
        fatores.append({
            "label": _friendly_label(feat, values[feat]),
            "contribuicao": float(val),
            "direcao": "aumenta" if val > 0 else "reduz",
        })
    return score, fatores
