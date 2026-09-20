"""Configuracoes e constantes compartilhadas do app Sentinela."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

APP_TITLE = "Sentinela"
APP_ICON = "🛰️"

# Paleta de cores (consistente entre paginas)
COLOR_PRIMARY = "#A6094A"      # vinho Locaweb/FIAP
COLOR_SECONDARY = "#0F1B33"    # azul escuro
COLOR_ACCENT = "#00B3B0"       # teal (usado em graficos SHAP na NB05)
COLOR_WARNING = "#D97706"
COLOR_MUTED = "#6B7280"

STATUS_COLORS = {
    "Validado": "green",
    "Experimental": "orange",
    "Exploratório": "orange",
}
