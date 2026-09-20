# Sentinela

Projeto acadêmico do **FIAP Enterprise Challenge 2026**, em parceria com a **Locaweb**.

## Problema

A operação da Locaweb enfrenta um volume massivo de chamados, tratado de forma reativa:

- **85,1%** dos tickets são abertos automaticamente via monitoramento, sem triagem prévia.
- **65,6%** são fechados sem intervenção humana — "fadiga de alertas".
- **61%** das soluções aplicadas são apenas paliativas (contorno).
- O **Team14** absorve **75,7%** de todo o volume de incidentes.

O sistema atual não distingue sinal de ruído, gerando risco constante de violação de OLA/SLA.

## Solução

O Sentinela é um motor de inteligência entre o monitoramento e as equipes de suporte, atuando em quatro frentes:

1. **Antecipação de incidentes** — previsão de volume D+1 e D+7.
2. **Triagem inteligente** — identifica tickets com alta chance de fechar sem intervenção.
3. **Projeção de risco de KPI/OLA** — radar de violações futuras por prioridade.
4. **Apoio à decisão operacional** — indica onde agir preventivamente.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app/main.py
```
