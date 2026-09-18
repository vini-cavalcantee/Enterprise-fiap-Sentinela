# Sentinela — App Streamlit (Sprint 4)

MVP funcional do painel operacional do Projeto Sentinela (FIAP Enterprise
Challenge + Locaweb).

## Estrutura

```
sentinela_app/
├── app.py                        # entrypoint / landing / navegação
├── pages/
│   ├── 1_Visao_Executiva.py      # ✅ implementada
│   ├── 2_Previsao.py             # 🚧 stub (Sprint 4 - próxima etapa)
│   ├── 3_Triagem_Inteligente.py  # 🚧 stub
│   ├── 4_Risco_OLA_KPI.py        # 🚧 stub
│   └── 5_Padroes_e_Explicabilidade.py  # 🚧 stub
├── src/
│   ├── config.py                 # constantes/cores
│   ├── data_loader.py            # leitura cacheada dos artefatos processados
│   └── components.py             # componentes de UI reutilizáveis
├── scripts/
│   └── build_artifacts.py        # gera data/processed/* a partir do dataset bruto
├── data/
│   ├── raw/LWDATASET.xlsx        # dataset bruto (não lido pelo app em runtime)
│   └── processed/                # artefatos consumidos pelo app (CSV/JSON)
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt

# 1) gerar os artefatos de dados (uma vez, ou sempre que o dataset mudar)
python scripts/build_artifacts.py

# 2) rodar o app
streamlit run app.py
```

## Princípios de arquitetura

- **Nenhum modelo é treinado dentro do app.** Todos os números exibidos
  vêm de artefatos pré-processados (`data/processed/`), gerados a partir
  dos notebooks NB01–NB05.
- **Separação clara** entre processamento de dados (`scripts/`,
  `src/data_loader.py`) e interface (`pages/`).
- **Cache do Streamlit** (`@st.cache_data`) em toda leitura de dados.
- Achados **exploratórios** (amostra curta, não validados com o mesmo
  rigor dos modelos oficiais) são sinalizados visualmente em laranja,
  distintos dos achados **validados** (verde).

## Status das páginas

| Página | Status | Fonte |
|---|---|---|
| Visão Executiva | ✅ Implementada | NB01 (série diária) + NB05 (clusters) |
| Previsão (D+1/D+7) | ✅ Implementada | Reconstrução de NB02 (`src/forecasting.py`) — ver aviso abaixo |
| Triagem Inteligente | ✅ Implementada | Reconstrução de NB03 (`src/triage.py`) — pipeline real serializado |
| Risco OLA/KPI | ✅ Implementada | Números oficiais do NB04 + estatística descritiva real (não modelo) |
| Padrões e Explicabilidade | ✅ Implementada | Artefatos oficiais da NB05 (figuras reutilizadas, nada recalculado) |

## ⚠️ Aviso importante sobre a página Previsão

O notebook **NB02 original não está disponível neste ambiente** (apenas
um resumo da metodologia). `src/forecasting.py` é uma **reconstrução
best-effort**, fiel ao que foi documentado (RandomForest para D+1,
Ridge multi-horizonte para D+7, mesmo corte de regime, sem leakage):

- A **previsão "ao vivo"** de D+1/D+7 (para o dia seguinte ao último
  dado do dataset) é uma inferência real, gerada sem uso de dados
  futuros.
- O **gráfico de evidência histórica** (Real × RandomForest ×
  Baseline) é uma **reconstrução ilustrativa** — o ganho agregado que
  ele mostra (~3,6%) é **menor** que o número oficial da Sprint 3
  (-8,2% MAE, 91 previsões, 3 janelas), porque o código exato do NB02
  (features/hiperparâmetros/fronteiras de janela) não foi preservado.
  Os números oficiais continuam exibidos como referência citada, não
  recalculados.
- **Se o arquivo NB02.ipynb (ou as previsões salvas da validação
  original) estiver disponível**, ele deve substituir
  `walkforward_reconstruction_d1()` em `src/forecasting.py` para que o
  gráfico bata exatamente com o número oficial.

## ⚠️ Aviso sobre a página Triagem Inteligente

Mesma situação da página Previsão: o notebook **NB03 original não está
disponível neste ambiente**. `src/triage.py` reutiliza **exatamente a
mesma reconstrução** já usada e apresentada na NB05 (Logistic
Regression, mesmas features, threshold 0,70, sem `Grupo designado` nem
campos pós-fechamento) — não é um modelo novo. O modelo é treinado uma
única vez em `scripts/build_artifacts.py` e serializado em
`data/processed/triage_model.joblib`; a página só carrega e roda
inferência (nunca retreina). Os números oficiais (precisão 95,9% /
cobertura 61,5% / redução de FP ~49%) são citados como constantes,
não recalculados a partir do pipeline local.

## ℹ️ Sobre a página Risco OLA/KPI

Diferente das páginas Previsão e Triagem, esta página **não reconstrói
nenhum modelo**. Os ganhos percentuais (-27,8% P3 / -3,17% P2) são
números oficiais do NB04, citados como constantes. Volume histórico,
% de entrada em KPI e violações observadas são agregações diretas do
`LWDATASET.xlsx` (estatística descritiva real, não previsão) — usadas
apenas como referência histórica, já que os outputs oficiais do NB04
(ex.: gráfico Real × Sentinela × Baseline) não estão disponíveis neste
ambiente. Nenhum gráfico ilustrativo foi reconstruído em seu lugar.

Não implementar PostgreSQL nem API REST nesta etapa (roadmap documentado
para além da Sprint 4).

## ✅ Sprint 4 — MVP completo

As 5 páginas do MVP estão implementadas. `data/reference/nb05_figures/`
guarda as figuras oficiais da NB05 (copiadas, nunca recalculadas) que
alimentam a página Padrões e Explicabilidade.
