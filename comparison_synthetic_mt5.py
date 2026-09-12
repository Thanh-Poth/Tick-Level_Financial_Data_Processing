"""
compare_synthetic_vs_mt5.py
=============================

Lance le MÊME pipeline (build_all_bars -> full_diagnostic) sur :
  1. les données synthétiques (data_acquisition.py)
  2. de vraies données MT5 (data_acquisition_mt5.py)
et produit un tableau + un graphique comparant les résultats.

A LANCER EN LOCAL (MT5 ouvert et connecté) — voir data_acquisition_mt5.py
pour l'explication de pourquoi ça ne peut pas tourner dans un sandbox cloud.

Usage :
    python3 compare_synthetic_vs_mt5.py --symbol EURUSD --days 5
"""

import sys
import argparse
import datetime as dt

sys.path.insert(0, "/home/claude/financial_bars")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_acquisition import generate_synthetic_ticks
from run_analysis import build_all_bars, OUT
from bars import bar_returns
from statistical_tests import full_diagnostic

ORDER = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
LABELS = {"time": "Time", "tick": "Tick", "volume": "Volume", "dollar": "Dollar",
          "dollar_imbalance": "Dollar Imb.", "run": "Run"}


def run_diagnostics(ticks: pd.DataFrame, target_n_bars: int = 390) -> pd.DataFrame:
    """Reprend exactement la logique de run_analysis.main() : construit
    les 6 types de bars puis lance la batterie de tests statistiques.
    Réutiliser cette fonction (plutôt que dupliquer le code) garantit
    que la comparaison synthétique/réel porte bien sur la MÊME méthode."""
    bars = build_all_bars(ticks, target_n_bars=target_n_bars)
    results = []
    for name, b in bars.items():
        r = bar_returns(b)
        results.append(full_diagnostic(r, name))
    return pd.DataFrame(results).set_index("bar_type"), bars


def compare(symbol: str, days: int, target_n_bars: int = 390):
    from data_acquisition_mt5 import fetch_mt5_ticks  # import ici : évite un ImportError
                                                          # au chargement du module si MT5
                                                          # n'est pas installé (ex. juste
                                                          # pour relire ce script)

    print(f"1) Données synthétiques ({days} jours)...")
    synth_ticks = generate_synthetic_ticks(n_days=days, seed=42)
    print(f"   -> {len(synth_ticks):,} trades")

    print(f"\n2) Données MT5 réelles ({symbol}, {days} derniers jours)...")
    end = dt.datetime.now()
    start = end - dt.timedelta(days=days)
    real_ticks = fetch_mt5_ticks(symbol, start, end)
    print(f"   -> {len(real_ticks):,} trades")

    print("\n3) Diagnostics sur données synthétiques...")
    synth_results, synth_bars = run_diagnostics(synth_ticks, target_n_bars)

    print("4) Diagnostics sur données MT5 réelles...")
    real_results, real_bars = run_diagnostics(real_ticks, target_n_bars)

    display_cols = ["n_bars", "std", "jb_stat", "skewness", "kurtosis",
                     "lb_pvalue_lag1", "arch_lm_pvalue"]
    print("\n=== Synthétique ===")
    print(synth_results[display_cols].round(4).to_string())
    print("\n=== MT5 réel ===")
    print(real_results[display_cols].round(4).to_string())

    plot_comparison(synth_results, real_results, symbol)
    write_comparison_report(synth_results, real_results, symbol, days)

    return synth_results, real_results


def plot_comparison(synth_results: pd.DataFrame, real_results: pd.DataFrame, symbol: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ORDER))
    width = 0.35

    jb_synth = synth_results.loc[ORDER, "jb_stat"]
    jb_real = real_results.loc[ORDER, "jb_stat"]
    axes[0].bar(x - width / 2, jb_synth.values, width, label="Synthétique", color="#3b6ea5")
    axes[0].bar(x + width / 2, jb_real.values, width, label=f"MT5 ({symbol})", color="#d17a22")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[k] for k in ORDER], rotation=25, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_title("Statistique Jarque-Bera\n(plus bas = plus proche d'une gaussienne)")
    axes[0].legend()

    kurt_synth = synth_results.loc[ORDER, "kurtosis"]
    kurt_real = real_results.loc[ORDER, "kurtosis"]
    axes[1].bar(x - width / 2, kurt_synth.values, width, label="Synthétique", color="#3b6ea5")
    axes[1].bar(x + width / 2, kurt_real.values, width, label=f"MT5 ({symbol})", color="#d17a22")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[k] for k in ORDER], rotation=25, ha="right")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Excess kurtosis\n(0 = gaussien)")
    axes[1].legend()

    fig.suptitle("Comparaison données synthétiques vs. données réelles MT5", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/compare_synthetic_vs_mt5.png", dpi=130)
    plt.close(fig)


def write_comparison_report(synth_results, real_results, symbol, days):
    lines = [f"# Comparaison données synthétiques vs. MT5 ({symbol}, {days}j)\n"]
    lines.append("## Statistique Jarque-Bera par type de bar\n")
    lines.append("| Type de bar | Synthétique | MT5 réel | Écart relatif |")
    lines.append("|---|---|---|---|")
    for k in ORDER:
        s, r = synth_results.loc[k, "jb_stat"], real_results.loc[k, "jb_stat"]
        rel = (r - s) / s * 100 if s != 0 else float("nan")
        lines.append(f"| {k} | {s:.1f} | {r:.1f} | {rel:+.0f}% |")

    lines.append("\n## Excess kurtosis par type de bar\n")
    lines.append("| Type de bar | Synthétique | MT5 réel |")
    lines.append("|---|---|---|")
    for k in ORDER:
        lines.append(f"| {k} | {synth_results.loc[k, 'kurtosis']:.3f} | {real_results.loc[k, 'kurtosis']:.3f} |")

    lines.append(
        "\n**A vérifier en priorité** : est-ce que le CLASSEMENT des bars "
        "(quel type a le JB le plus bas) est le même sur les deux sources ? "
        "Si oui, c'est un bon signe que les données synthétiques capturent "
        "des propriétés de microstructure suffisamment réalistes pour que "
        "les conclusions du projet se généralisent aux vraies données."
    )

    with open(f"{OUT}/COMPARISON_RESULTS.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD", help="Symbole MT5 (voir Market Watch)")
    parser.add_argument("--days", type=int, default=5, help="Nombre de jours d'historique")
    args = parser.parse_args()

    compare(args.symbol, args.days)
    print(f"\nTerminé. Résultats dans {OUT}/COMPARISON_RESULTS.md et compare_synthetic_vs_mt5.png")