"""
run_analysis.py
================
Orchestre tout le pipeline : génère les données, construit les 6 types
de bars, calcule les rendements, lance la batterie de tests
statistiques, et produit un tableau comparatif + des graphiques.
"""

import sys
sys.path.insert(0, "/home/claude/financial_bars")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from data_acquisition import generate_synthetic_ticks
from bars import time_bars, tick_bars, volume_bars, dollar_bars, calibrate_threshold, bar_returns
from imbalance_bars import dollar_imbalance_bars, run_bars
from statistical_tests import full_diagnostic

OUT = "/mnt/user-data/outputs"


def build_all_bars(ticks: pd.DataFrame, target_n_bars: int = 390):
    """Construit les 6 types de bars, calibrés pour produire un nombre
    de bars comparable (sauf les information-driven bars, dont la
    taille variable est justement la propriété étudiée)."""
    bars = {}
    bars["time"] = time_bars(ticks, freq="5min")

    n_target = len(bars["time"])
    tick_thresh = calibrate_threshold(ticks.assign(count=1).rename(columns={"count": "count"}), "count", n_target) \
        if False else calibrate_threshold(ticks.assign(count=1), "count", n_target)
    vol_thresh = calibrate_threshold(ticks, "volume", n_target)
    dol_thresh = calibrate_threshold(ticks, "dollar", n_target)

    bars["tick"] = tick_bars(ticks, ticks_per_bar=int(tick_thresh))
    bars["volume"] = volume_bars(ticks, volume_per_bar=vol_thresh)
    bars["dollar"] = dollar_bars(ticks, dollar_per_bar=dol_thresh)

    # init_expected_ticks aligné sur la taille moyenne des autres bars
    # pour une comparaison raisonnable (mais la taille RÉALISÉE des
    # imbalance/run bars restera hétérogène par construction -> c'est
    # justement ce qu'on veut observer).
    avg_size = len(ticks) // n_target
    bars["dollar_imbalance"] = dollar_imbalance_bars(ticks, init_expected_ticks=avg_size)
    bars["run"] = run_bars(ticks, init_expected_ticks=avg_size)

    return bars


def main():
    print("1) Génération des données tick synthétiques...")
    ticks = generate_synthetic_ticks(n_days=5, seed=42)
    print(f"   -> {len(ticks):,} trades sur {ticks.index.normalize().nunique()} jours")

    print("\n2) Construction des 6 types de bars...")
    bars = build_all_bars(ticks)
    for name, b in bars.items():
        print(f"   {name:20s}: {len(b):6d} bars  (taille moy. {len(ticks)/len(b):7.1f} trades/bar, "
              f"std taille {b['n_trades'].std():7.1f})")

    print("\n3) Calcul des rendements et batterie de tests statistiques...")
    results = []
    returns_by_type = {}
    for name, b in bars.items():
        r = bar_returns(b)
        returns_by_type[name] = r
        results.append(full_diagnostic(r, name))

    results_df = pd.DataFrame(results).set_index("bar_type")
    display_cols = ["n_bars", "std", "jb_stat", "skewness", "kurtosis",
                     "lb_pvalue_lag1", "autocorr_lag1", "arch_lm_pvalue", "adf_pvalue"]
    print("\n" + results_df[display_cols].round(4).to_string())

    results_df.to_csv(f"{OUT}/bars_comparison_results.csv")

    print("\n4) Génération des graphiques...")
    plot_return_distributions(returns_by_type)
    plot_acf_comparison(returns_by_type)
    plot_bar_size_distribution(bars)
    plot_jb_bar_chart(results_df)

    write_report(results_df, bars, ticks)

    print(f"\nTerminé. Fichiers dans {OUT}/")


def plot_return_distributions(returns_by_type: dict):
    order = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
    labels = {"time": "Time bars", "tick": "Tick bars", "volume": "Volume bars",
              "dollar": "Dollar bars", "dollar_imbalance": "Dollar Imbalance bars", "run": "Run bars"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, key in zip(axes.flat, order):
        r = returns_by_type[key].dropna()
        r_std = (r - r.mean()) / r.std()
        ax.hist(r_std, bins=60, density=True, alpha=0.7, color="#3b6ea5", edgecolor="none")
        x = np.linspace(-5, 5, 200)
        ax.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2), "r-", lw=1.5, label="N(0,1)")
        ax.set_title(f"{labels[key]}  (n={len(r)})", fontsize=11)
        ax.set_xlim(-6, 6)
        ax.legend(fontsize=8)
    fig.suptitle("Rendements standardisés par type de bar, vs. loi normale", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/return_distributions.png", dpi=130)
    plt.close(fig)


def plot_acf_comparison(returns_by_type: dict):
    order = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
    labels = {"time": "Time bars", "tick": "Tick bars", "volume": "Volume bars",
              "dollar": "Dollar bars", "dollar_imbalance": "Dollar Imbalance bars", "run": "Run bars"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, key in zip(axes.flat, order):
        r = returns_by_type[key].dropna()
        plot_acf(r, lags=20, ax=ax, title=f"{labels[key]}", zero=False)
    fig.suptitle("Autocorrélation des rendements par type de bar (20 premiers lags)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/acf_comparison.png", dpi=130)
    plt.close(fig)


def plot_bar_size_distribution(bars: dict):
    order = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
    labels = {"time": "Time bars", "tick": "Tick bars", "volume": "Volume bars",
              "dollar": "Dollar bars", "dollar_imbalance": "Dollar Imbalance bars", "run": "Run bars"}
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [bars[k]["n_trades"].values for k in order]
    ax.boxplot(data, labels=[labels[k] for k in order], showfliers=True)
    ax.set_ylabel("Nombre de trades par bar")
    ax.set_yscale("log")
    ax.set_title("Distribution de la taille des bars (nb de trades), échelle log")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bar_size_distribution.png", dpi=130)
    plt.close(fig)


def plot_jb_bar_chart(results_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    order = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
    jb = results_df.loc[order, "jb_stat"]
    axes[0].bar(range(len(order)), jb.values, color="#3b6ea5")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=30, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_title("Statistique de Jarque-Bera (échelle log)\nplus bas = plus proche d'une gaussienne")

    kurt = results_df.loc[order, "kurtosis"]
    axes[1].bar(range(len(order)), kurt.values, color="#a53b3b")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=30, ha="right")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Excess kurtosis\n0 = gaussien, >0 = queues grasses")

    fig.tight_layout()
    fig.savefig(f"{OUT}/jb_and_kurtosis.png", dpi=130)
    plt.close(fig)


def write_report(results_df: pd.DataFrame, bars: dict, ticks: pd.DataFrame):
    order = ["time", "tick", "volume", "dollar", "dollar_imbalance", "run"]
    best_jb = results_df.loc[order, "jb_stat"].idxmin()
    worst_jb = results_df.loc[order, "jb_stat"].idxmax()

    lines = []
    lines.append("# Résultats — Financial Data Structures (bars alternatifs)\n")
    lines.append(f"Données : {len(ticks):,} trades simulés sur {ticks.index.normalize().nunique()} jours.\n")
    lines.append("## Nombre de bars et taille\n")
    lines.append("| Type de bar | Nb bars | Taille moy. (trades) | Std taille |")
    lines.append("|---|---|---|---|")
    for k in order:
        b = bars[k]
        lines.append(f"| {k} | {len(b)} | {len(ticks)/len(b):.1f} | {b['n_trades'].std():.1f} |")

    lines.append("\n## Tests statistiques sur les rendements\n")
    lines.append("| Type de bar | JB stat | Skewness | Excess kurtosis | LB p-val (lag1) | ARCH p-val | ADF p-val |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in order:
        row = results_df.loc[k]
        lines.append(
            f"| {k} | {row.jb_stat:.1f} | {row.skewness:.3f} | {row['kurtosis']:.3f} | "
            f"{row.lb_pvalue_lag1:.4f} | {row.arch_lm_pvalue:.4f} | {row.adf_pvalue:.4f} |"
        )

    lines.append(f"\n**Bar le plus proche d'une distribution gaussienne (JB stat la plus faible) : `{best_jb}`**")
    lines.append(f"**Bar le plus éloigné (JB stat la plus élevée) : `{worst_jb}`**\n")

    with open(f"{OUT}/RESULTS.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
