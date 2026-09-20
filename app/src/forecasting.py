"""
Modulo de forecasting do Sentinela — reconstrucao da metodologia
descrita para a NB02 (D+1 RandomForest / D+7 Ridge multi-horizonte).

IMPORTANTE — leia antes de usar:
O notebook NB02 original nao esta disponivel neste ambiente (apenas um
resumo de alto nivel da metodologia ficou documentado em memoria de
projeto). As funcoes abaixo sao uma RECONSTRUCAO best-effort, fiel ao
que foi documentado (RandomForest para D+1, Ridge multi-horizonte
independente por dia para D+7, corte de regime >= 01/09/2025, sem uso
de dados futuros). Elas NAO reproduzem byte-a-byte os resultados
oficiais da Sprint 3 (91 previsoes / 3 janelas / -8,2% MAE) porque o
codigo exato (features, hiperparametros, fronteiras de janela) nao foi
preservado. Por isso:

- Os NUMEROS OFICIAIS da Sprint 3 sao tratados como constantes citadas
  (nunca recalculados aqui) e exibidos como tal na interface.
- O grafico de evidencia historica gerado por este modulo e uma
  RECONSTRUCAO ILUSTRATIVA, rotulada como tal na interface, usada para
  dar visibilidade visual ao comportamento do modelo (nao para
  substituir o numero oficial).
- A previsao "ao vivo" de D+1/D+7 (para o dia seguinte ao ultimo dado
  disponivel) e uma inferencia real, gerada com a mesma logica de
  features, sem leakage.

Todas as funcoes usam apenas informacao conhecida ATE a data de corte
para gerar cada feature/previsao.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGIME_START = pd.Timestamp("2025-09-01")
FEATS_D1 = ["lag_1", "lag_2", "lag_3", "lag_7", "media_7d", "media_14d", "dow", "is_weekend", "trend"]

# Hiperparametros do RandomForest D+1 (reconstrucao). Regularizados para
# reduzir overfitting dado o historico curto (~108 dias uteis).
RF_PARAMS = dict(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
RIDGE_ALPHA = 1.0


def build_daily_series(df_raw: pd.DataFrame) -> pd.Series:
    """Serie diaria de volume, pos-regime, com todos os dias do
    calendario preenchidos (0 onde nao houve incidentes)."""
    dfr = df_raw[df_raw["Aberto"] >= REGIME_START].copy()
    dfr["data"] = dfr["Aberto"].dt.floor("D")
    serie = dfr.groupby("data").size().rename("volume").reset_index().set_index("data").asfreq("D").fillna(0)
    return serie["volume"]


def build_features_d1(vol: pd.Series) -> pd.DataFrame:
    """Features de D+1 indexadas pela DATA ALVO T (nao pela data de
    origem). lag_k(T) = volume(T-k); target(T) = volume(T) (sem shift
    adicional). Isso garante que, ao estender a serie com uma linha
    futura (volume=NaN), a linha resultante ja contem exatamente as
    features corretas para prever aquela data, sem depender de contar
    "off-by-one" manualmente."""
    s = pd.DataFrame(index=vol.index)
    s["volume"] = vol
    s["lag_1"] = vol.shift(1)
    s["lag_2"] = vol.shift(2)
    s["lag_3"] = vol.shift(3)
    s["lag_7"] = vol.shift(7)
    s["media_7d"] = vol.shift(1).rolling(7).mean()
    s["media_14d"] = vol.shift(1).rolling(14).mean()
    s["dow"] = s.index.dayofweek
    s["is_weekend"] = (s["dow"] >= 5).astype(int)
    s["trend"] = np.arange(len(s))
    s["target"] = vol
    return s


@dataclass
class ForecastD1:
    data_prevista: pd.Timestamp
    volume_previsto: float
    ultimo_dia: pd.Timestamp
    ultimo_valor_real: float
    media_movel_7d_recente: float
    modelo: str = "RandomForest"
    status: str = "Validado"


def forecast_d1_live(vol: pd.Series) -> ForecastD1:
    """Treina o RandomForest em TODO o historico disponivel e gera a
    previsao para o dia seguinte ao ultimo dado (sem leakage: a linha
    de previsao usa apenas volumes ja observados como lags)."""
    ultimo_dia = vol.index.max()
    data_prevista = ultimo_dia + pd.Timedelta(days=1)

    vol_ext = pd.concat([vol, pd.Series([np.nan], index=[data_prevista])])
    s = build_features_d1(vol_ext)

    s_train = s.dropna(subset=FEATS_D1 + ["target"])
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(s_train[FEATS_D1], s_train["target"])

    x_forecast = s.loc[[data_prevista], FEATS_D1]
    pred = float(model.predict(x_forecast)[0])
    pred = max(0.0, pred)

    return ForecastD1(
        data_prevista=data_prevista,
        volume_previsto=pred,
        ultimo_dia=ultimo_dia,
        ultimo_valor_real=float(vol.iloc[-1]),
        media_movel_7d_recente=float(vol.iloc[-7:].mean()),
    )


def walkforward_reconstruction_d1(vol: pd.Series, n_test: int = 91, n_windows: int = 3) -> pd.DataFrame:
    """Reconstrucao ILUSTRATIVA da validacao temporal walk-forward
    (nao e o resultado oficial da Sprint 3 - ver aviso no topo do
    modulo). Retorna um DataFrame com real / previsto / baseline por
    dia e por janela, para uso exclusivo em visualizacao."""
    s = build_features_d1(vol)
    s_model = s.dropna(subset=FEATS_D1 + ["target"]).copy()

    n = len(s_model)
    initial_train = n - n_test
    fold_size = n_test // n_windows
    remainder = n_test - fold_size * n_windows
    fold_sizes = [fold_size] * n_windows
    fold_sizes[-1] += remainder

    rows = []
    idx = initial_train
    for w, fs in enumerate(fold_sizes, start=1):
        train = s_model.iloc[:idx]
        test = s_model.iloc[idx: idx + fs]

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(train[FEATS_D1], train["target"])
        pred = model.predict(test[FEATS_D1])

        for date, real, p, base in zip(test.index, test["target"].values, pred, test["media_7d"].values):
            rows.append({"data": date, "janela": w, "real": real, "rf_pred": max(0.0, p), "baseline_pred": base})
        idx += fs

    return pd.DataFrame(rows)


def build_features_d7(vol: pd.Series, h: int) -> pd.DataFrame:
    """Features para o horizonte h (1..7), a partir de uma origem O
    (indice da linha). Os lags/medias sao calculados EM O (incluindo o
    proprio dia O), e o alvo e volume(O+h). dow/trend descrevem a DATA
    ALVO (O+h), nao a origem."""
    n = len(vol)
    idx_arr = np.arange(n)
    feat = pd.DataFrame(index=vol.index)
    feat["lag_1"] = vol
    feat["lag_2"] = vol.shift(1)
    feat["lag_3"] = vol.shift(2)
    feat["lag_7"] = vol.shift(6)
    feat["media_7d"] = vol.rolling(7).mean()
    feat["media_14d"] = vol.rolling(14).mean()
    target_dates = vol.index + pd.Timedelta(days=h)
    feat["dow"] = target_dates.dayofweek
    feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
    feat["trend"] = idx_arr + h
    feat["target"] = vol.shift(-h)
    feat["data_alvo"] = target_dates
    return feat


FEATS_D7 = ["lag_1", "lag_2", "lag_3", "lag_7", "media_7d", "media_14d", "dow", "is_weekend", "trend"]


def forecast_d7_live(vol: pd.Series) -> pd.DataFrame:
    """Sete modelos Ridge independentes (um por horizonte h=1..7),
    cada um treinado em todo o historico disponivel para aquele
    horizonte, prevendo a partir da ultima origem conhecida (ultimo
    dia real da serie)."""
    ultimo_dia = vol.index.max()
    resultados = []

    for h in range(1, 8):
        feat = build_features_d7(vol, h)
        train = feat.dropna(subset=FEATS_D7 + ["target"])

        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA, random_state=42))
        model.fit(train[FEATS_D7], train["target"])

        origem = feat.loc[[ultimo_dia]]
        pred = float(model.predict(origem[FEATS_D7])[0])
        pred = max(0.0, pred)

        resultados.append({
            "horizonte": h,
            "data": ultimo_dia + pd.Timedelta(days=h),
            "volume_previsto": pred,
            "n_treino": len(train),
        })

    return pd.DataFrame(resultados)
