"""
Gera os artefatos processados (CSV/JSON) consumidos pelo app Streamlit.

Este script NAO treina nenhum modelo. Apenas reproduz, a partir do
LWDATASET.xlsx, as mesmas agregacoes/decisoes metodologicas ja
documentadas em NB01 (corte de regime, serie diaria) e NB05 (perfil de
clusters). Deve ser executado uma vez (ou sempre que o dataset for
atualizado) - o app le apenas os artefatos gerados aqui, nunca o xlsx
bruto em tempo de execucao.

Uso:
    python scripts/build_artifacts.py
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # garante que `src` seja importavel rodando de qualquer lugar

from src.forecasting import (  # noqa: E402
    build_daily_series,
    forecast_d1_live,
    forecast_d7_live,
    walkforward_reconstruction_d1,
)
from src.triage import train_triage_model  # noqa: E402
RAW_PATH = BASE_DIR / "data" / "raw" / "LWDATASET.xlsx"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mesmo corte de regime usado desde a NB01
REGIME_START = pd.Timestamp("2025-09-01")
BASELINE_START = pd.Timestamp("2025-01-01")
BASELINE_END = pd.Timestamp("2025-09-01")  # exclusivo


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW_PATH, sheet_name="Dataset Geral")
    return df


def build_serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """Serie diaria pos-regime, com quebra por prioridade, origem e status."""
    dfr = df[df["Aberto"] >= REGIME_START].copy()
    dfr["data"] = dfr["Aberto"].dt.floor("D")

    dfr["is_p2"] = dfr["Prioridade"] == "2 - Alta"
    dfr["is_p3"] = dfr["Prioridade"] == "3 - Média"
    dfr["is_monitoramento"] = dfr["Aberto por"] == "Monitoramento"
    dfr["is_manual"] = dfr["Aberto por"] == "Manual"
    dfr["is_sem_intervencao"] = dfr["Status"] == "Sem Intervenção"

    full_idx = pd.date_range(dfr["data"].min(), dfr["data"].max(), freq="D")

    serie = dfr.groupby("data").agg(
        volume_total=("Número", "count"),
        volume_p2=("is_p2", "sum"),
        volume_p3=("is_p3", "sum"),
        volume_monitoramento=("is_monitoramento", "sum"),
        volume_manual=("is_manual", "sum"),
        volume_sem_intervencao=("is_sem_intervencao", "sum"),
    ).reindex(full_idx, fill_value=0)

    serie.index.name = "data"
    serie["media_movel_7d"] = serie["volume_total"].rolling(7, min_periods=1).mean().round(1)
    return serie.reset_index()


def build_kpis_executivos(df: pd.DataFrame, serie: pd.DataFrame) -> dict:
    dfr = df[df["Aberto"] >= REGIME_START].copy()
    baseline = df[(df["Aberto"] >= BASELINE_START) & (df["Aberto"] < BASELINE_END)].copy()

    # media diaria baseline (jan-ago/2025), considerando todos os dias do periodo
    baseline_days = pd.date_range(BASELINE_START, BASELINE_END - pd.Timedelta(days=1), freq="D")
    baseline_daily = (
        baseline.groupby(baseline["Aberto"].dt.floor("D")).size().reindex(baseline_days, fill_value=0)
    )
    media_baseline = float(baseline_daily.mean())
    media_pos = float(serie["volume_total"].mean())
    crescimento_pct = (media_pos / media_baseline - 1) * 100

    kpis = {
        "periodo_inicio": str(dfr["Aberto"].min().date()),
        "periodo_fim": str(dfr["Aberto"].max().date()),
        "dias_periodo": int(serie.shape[0]),
        "volume_total_periodo": int(dfr.shape[0]),
        "volume_medio_diario": round(media_pos, 1),
        "volume_medio_diario_baseline": round(media_baseline, 1),
        "crescimento_pct_vs_baseline": round(crescimento_pct, 1),
        "pct_monitoramento": round((dfr["Aberto por"] == "Monitoramento").mean() * 100, 1),
        "pct_manual": round((dfr["Aberto por"] == "Manual").mean() * 100, 1),
        "pct_sem_intervencao": round((dfr["Status"] == "Sem Intervenção").mean() * 100, 1),
        "pct_team14": round((dfr["Grupo designado"] == "Team14").mean() * 100, 1),
        "pct_p2": round((dfr["Prioridade"] == "2 - Alta").mean() * 100, 1),
        "pct_p3": round((dfr["Prioridade"] == "3 - Média").mean() * 100, 1),
        "pct_p4": round((dfr["Prioridade"] == "4 - Baixa").mean() * 100, 1),
        "pct_p5": round((dfr["Prioridade"] == "5 - Muito Baixa").mean() * 100, 1),
        "pct_sem_categoria_produto": round(
            (dfr["Categoria"].isna() & dfr["Produto"].isna()).mean() * 100, 1
        ),
        # resultados VALIDADOS na Sprint 3 - constantes de referencia, nao recalculados aqui
        "modelo_d1_mae_reducao_pct": 8.2,
        "modelo_d1_algoritmo": "RandomForest",
        "modelo_d1_status": "Validado",
        "modelo_triagem_precisao_pct": 95.9,
        "modelo_triagem_cobertura_pct": 61.5,
        "modelo_triagem_reducao_fp_pct": 49.0,
        "modelo_triagem_status": "Validado",
        "modelo_d7_status": "Em validação",
        "modelo_p2p3_status": "Em validação",
        "modelo_ola_kpi_status": "Experimental",
    }
    return kpis


def build_top_categorias(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Top categorias por volume, EXCLUINDO NAO_INFORMADO (que e' ruido de
    monitoramento, ja tratado separadamente). Representa onde esta o
    trabalho operacional real identificado."""
    dfr = df[df["Aberto"] >= REGIME_START].copy()
    dfr = dfr[dfr["Categoria"].notna()]

    dfr["sem_intervencao"] = (dfr["Status"] == "Sem Intervenção").astype(int)
    dfr["contorno"] = (dfr["Solução"] == "Contorno").astype(int)

    top = (
        dfr.groupby("Categoria")
        .agg(volume=("Número", "count"), pct_sem_intervencao=("sem_intervencao", "mean"), pct_contorno=("contorno", "mean"))
        .sort_values("volume", ascending=False)
        .head(top_n)
        .reset_index()
    )
    top["pct_sem_intervencao"] = (top["pct_sem_intervencao"] * 100).round(1)
    top["pct_contorno"] = (top["pct_contorno"] * 100).round(1)
    return top


def copy_cluster_profile() -> pd.DataFrame:
    """Copia o perfil de clusters ja calculado e validado na NB05
    (nao recalcula clustering aqui). O arquivo fonte fica empacotado
    dentro do proprio app (data/reference/), gerado uma vez a partir
    da NB05 - assim o app nao depende de uma pasta externa ao pacote."""
    src = BASE_DIR / "data" / "reference" / "perfil_clusters_nb05.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"perfil_clusters_nb05.csv nao encontrado em {src}. "
            "Esse arquivo deveria vir junto do pacote do app (data/reference/)."
        )
    perfil = pd.read_csv(src)

    labels = {
        0: "Nicho de risco de KPI",
        1: "Ruído de monitoramento",
        2: "Correção paliativa recorrente",
        3: "Resolução definitiva manual",
    }
    exploratorio = {0: True, 1: False, 2: False, 3: False}  # cluster 0 = amostra curta (3 combos)

    perfil["label"] = perfil["cluster"].map(labels)
    perfil["exploratorio"] = perfil["cluster"].map(exploratorio)
    return perfil


def build_previsao_d1(vol: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Gera a previsao D+1 "ao vivo" (RandomForest, sem leakage) e a
    reconstrucao ilustrativa da validacao walk-forward.

    IMPORTANTE: a reconstrucao walk-forward NAO e o resultado oficial
    da Sprint 3 (NB02 nao esta disponivel neste ambiente). O numero
    oficial (-8,2% MAE, 91 previsoes, 3 janelas) e mantido como
    constante citada em kpis_executivos.json e nunca sobrescrito por
    este calculo.
    """
    forecast = forecast_d1_live(vol)
    previsao_dict = {
        "data_prevista": str(forecast.data_prevista.date()),
        "volume_previsto": round(forecast.volume_previsto, 0),
        "ultimo_dia": str(forecast.ultimo_dia.date()),
        "ultimo_valor_real": forecast.ultimo_valor_real,
        "media_movel_7d_recente": round(forecast.media_movel_7d_recente, 1),
        "modelo": forecast.modelo,
        "status": forecast.status,
    }

    wf = walkforward_reconstruction_d1(vol)
    return previsao_dict, wf


def build_previsao_d7(vol: pd.Series) -> pd.DataFrame:
    return forecast_d7_live(vol)


def build_previsao_p2p3() -> dict:
    """Numeros oficiais da Sprint 3 (NB02, slide 7) para o bloco
    compacto de P2/P3. Constantes citadas, nao recalculadas."""
    return {
        "p2_modelo": "ElasticNet",
        "p2_mae_baseline": 52.4,
        "p2_mae_modelo": 49.8,
        "p2_ganho_pct": 5.1,
        "p2_status": "Em validação",
        "p3_modelo": "Abordagem hierárquica",
        "p3_mae_baseline": 59.6,
        "p3_mae_modelo": 57.3,
        "p3_ganho_pct": 3.7,
        "p3_status": "Em validação",
    }


def build_triage_artifacts():
    """Treina (uma unica vez) e serializa o pipeline de Triagem
    Inteligente (Logistic Regression, mesma reconstrucao da NB05) e as
    listas de categorias usadas no treino (para popular o formulario
    da pagina sem hardcode)."""
    df = load_raw()
    triage = train_triage_model(df)

    joblib.dump(triage.model, OUT_DIR / "triage_model.joblib")

    categorias = {
        "feature_columns": triage.feature_columns,
        "top_produtos": triage.top_produtos,
        "top_categorias": triage.top_categorias,
    }
    with open(OUT_DIR / "triage_categorias.json", "w", encoding="utf-8") as f:
        json.dump(categorias, f, ensure_ascii=False, indent=2)

    return triage


def build_risco_kpi(df: pd.DataFrame) -> dict:
    """Indicadores de risco OLA/KPI para P2/P3.

    IMPORTANTE: os ganhos percentuais (-27,8% P3 / -3,17% P2) são
    NUMEROS OFICIAIS do NB04 (constantes citadas, nao recalculados
    aqui — o NB04 nao esta disponivel neste ambiente para reproduzir
    o pipeline completo). Os demais campos (volume historico, % de
    entrada em KPI, violacoes observadas, tendencia mensal) sao
    agregacoes DIRETAS e reais do dataset bruto — nao sao previsoes
    nem reconstrucoes de modelo, apenas estatistica descritiva,
    fornecida como referencia historica/observada.
    """
    dfr = df[df["Aberto"] >= REGIME_START].copy()
    dfr["mes"] = dfr["Aberto"].dt.to_period("M").astype(str)

    def stats_prioridade(codigo: str) -> dict:
        sub = dfr[dfr["Prioridade"] == codigo].copy()
        sub["entrou"] = (sub["Entrou para KPI?"] == "SIM").astype(int)
        entrou_mask = sub["entrou"] == 1
        viol_cond = sub.loc[entrou_mask, "KPI Violado?"]
        por_mes = sub.groupby("mes")["entrou"].mean() * 100
        return {
            "volume_historico": int(len(sub)),
            "pct_entrou_kpi": round(sub["entrou"].mean() * 100, 1),
            "n_entrou_kpi": int(entrou_mask.sum()),
            "pct_violacao_condicional": round((viol_cond == "SIM").mean() * 100, 2),
            "n_violacoes_historico": int((viol_cond == "SIM").sum()),
            "entrada_kpi_por_mes": {k: round(v, 1) for k, v in por_mes.items()},
        }

    p2 = stats_prioridade("2 - Alta")
    p3 = stats_prioridade("3 - Média")

    return {
        "formula": "violações esperadas = volume estimado × taxa de entrada em KPI × taxa de violação",
        "p3_modelo": "Abordagem hierárquica (NB02)",
        "p3_ganho_pct": 27.8,
        "p3_status": "Em validação",
        "p2_modelo": "ElasticNet (NB02)",
        "p2_ganho_pct": 3.17,
        "p2_status": "Em validação — resultado inconclusivo",
        "p2": p2,
        "p3": p3,
    }


def copy_nb05_figures():
    """Copia (sem recalcular nada) as figuras OFICIAIS ja geradas pela
    NB05 para dentro de data/processed/figures/, de onde a pagina
    Padroes e Explicabilidade as le com st.image. Simples copia de
    arquivo — nenhum grafico e recriado ou recalculado aqui."""
    src_dir = BASE_DIR / "data" / "reference" / "nb05_figures"
    dst_dir = OUT_DIR / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)

    required = ["03_share_volume_cluster.png", "05_shap_d1_bar.png", "06_coef_triagem.png"]
    copied = []
    for name in required:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def main():
    print(f"Lendo dataset bruto de: {RAW_PATH}")
    df = load_raw()
    print(f"Registros totais: {len(df):,}")

    serie = build_serie_diaria(df)
    serie.to_csv(OUT_DIR / "serie_diaria.csv", index=False)
    print(f"[OK] serie_diaria.csv ({len(serie)} dias)")

    kpis = build_kpis_executivos(df, serie)
    with open(OUT_DIR / "kpis_executivos.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)
    print("[OK] kpis_executivos.json")
    print(json.dumps(kpis, ensure_ascii=False, indent=2))

    top_cat = build_top_categorias(df)
    top_cat.to_csv(OUT_DIR / "top_categorias.csv", index=False)
    print(f"[OK] top_categorias.csv ({len(top_cat)} categorias)")

    try:
        perfil = copy_cluster_profile()
        perfil.to_csv(OUT_DIR / "perfil_clusters.csv", index=False)
        print(f"[OK] perfil_clusters.csv ({len(perfil)} clusters)")
    except FileNotFoundError as e:
        print(f"[AVISO] {e}")

    print("\nGerando artefatos de previsao (D+1 / D+7)...")
    vol = build_daily_series(df)

    previsao_d1, walkforward_d1 = build_previsao_d1(vol)
    with open(OUT_DIR / "previsao_d1.json", "w", encoding="utf-8") as f:
        json.dump(previsao_d1, f, ensure_ascii=False, indent=2)
    print("[OK] previsao_d1.json")
    print(json.dumps(previsao_d1, ensure_ascii=False, indent=2))

    walkforward_d1.to_csv(OUT_DIR / "walkforward_d1.csv", index=False)
    print(f"[OK] walkforward_d1.csv ({len(walkforward_d1)} previsões, reconstrução ilustrativa)")

    previsao_d7 = build_previsao_d7(vol)
    previsao_d7.to_csv(OUT_DIR / "previsao_d7.csv", index=False)
    print(f"[OK] previsao_d7.csv ({len(previsao_d7)} horizontes)")

    previsao_p2p3 = build_previsao_p2p3()
    with open(OUT_DIR / "previsao_p2p3.json", "w", encoding="utf-8") as f:
        json.dump(previsao_p2p3, f, ensure_ascii=False, indent=2)
    print("[OK] previsao_p2p3.json")

    print("\nTreinando e serializando o pipeline de Triagem Inteligente...")
    build_triage_artifacts()
    print("[OK] triage_model.joblib")
    print("[OK] triage_categorias.json")

    print("\nCalculando indicadores de Risco OLA/KPI...")
    risco_kpi = build_risco_kpi(df)
    with open(OUT_DIR / "risco_kpi.json", "w", encoding="utf-8") as f:
        json.dump(risco_kpi, f, ensure_ascii=False, indent=2)
    print("[OK] risco_kpi.json")
    print(json.dumps(risco_kpi, ensure_ascii=False, indent=2))

    print("\nCopiando figuras oficiais da NB05...")
    copiadas = copy_nb05_figures()
    print(f"[OK] {len(copiadas)} figuras copiadas: {copiadas}")

    print("\nArtefatos gerados em:", OUT_DIR)


if __name__ == "__main__":
    main()
