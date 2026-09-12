"""
statistical_tests.py
=====================

Les tests qu'on applique aux rendements de CHAQUE type de bar, et
POURQUOI on les choisit. C'est le cœur scientifique du projet : on ne
se contente pas de "construire des bars différents", on démontre
QUANTITATIVEMENT lesquels ont de meilleures propriétés statistiques
pour du ML derrière (rendements plus proches i.i.d. Gaussien = hypothèse
de base de beaucoup de modèles, tests d'hypothèses, calcul de Sharpe, etc.)

Pour chaque type de bar, sur la série de log-rendements r_t = log(close_t) - log(close_{t-1}) :

1. JARQUE-BERA (normalité)
   H0 : les rendements suivent une loi normale (skewness=0, kurtosis=3)
   Statistique basée sur skewness et excess kurtosis empiriques.
   p-value faible (< 0.05) -> on REJETTE la normalité.
   Interprétation attendue : TOUS les rendements financiers rejettent
   la normalité (queues grasses), mais le but ici est de comparer
   l'AMPLEUR du rejet (statistique JB) entre types de bars : plus la
   statistique JB est faible, plus la distribution se rapproche d'une
   gaussienne -> les bars "intelligents" (volume/dollar/imbalance)
   doivent avoir un JB significativement plus faible que time bars.

2. LJUNG-BOX (autocorrélation des rendements)
   H0 : les rendements ne sont pas autocorrélés (bruit blanc)
   Testé sur plusieurs lags (ex: 1 à 10).
   Une autocorrélation significative dans les RENDEMENTS (pas les
   rendements au carré) est en général le signe d'une microstructure
   mal échantillonnée (bid-ask bounce, échantillonnage à fréquence
   inadaptée) plutôt que d'une vraie prévisibilité.

3. ARCH-LM (hétéroscédasticité conditionnelle, "volatility clustering")
   H0 : pas d'effet ARCH (variance conditionnelle constante)
   Testé sur les rendements au carré. On s'attend à ce que TOUS les
   types de bars montrent un effet ARCH (la volatility clustering est
   un fait stylisé quasi-universel), mais là encore on compare
   l'AMPLEUR (statistique du test) entre types de bars.

4. Bonus : ADF (stationnarité) sur les rendements — sert de garde-fou :
   si un type de bar donne des rendements non-stationnaires, la
   comparaison des tests ci-dessus n'a pas beaucoup de sens.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def jarque_bera_test(returns: pd.Series) -> dict:
    """Test de normalité. Statistique élevée = distribution plus éloignée d'une gaussienne."""
    jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
    return {
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
        "skewness": stats.skew(returns.dropna()),
        "kurtosis": stats.kurtosis(returns.dropna(), fisher=True),  # excess kurtosis (0 = gaussien)
        "reject_normality_5pct": bool(jb_pvalue < 0.05),
    }


def ljung_box_test(returns: pd.Series, lags: int = 10) -> dict:
    """
    Test d'autocorrélation des RENDEMENTS bruts (pas au carré).
    On regarde le lag 1 en particulier (le plus interprétable) et un
    résumé sur les 10 premiers lags.
    """
    res = acorr_ljungbox(returns.dropna(), lags=[1, 5, lags], return_df=True)
    return {
        "lb_stat_lag1": res.loc[1, "lb_stat"],
        "lb_pvalue_lag1": res.loc[1, "lb_pvalue"],
        "lb_pvalue_lag10": res.loc[lags, "lb_pvalue"],
        "autocorr_lag1": returns.autocorr(lag=1),
    }


def arch_test(returns: pd.Series, lags: int = 5) -> dict:
    """
    Test ARCH-LM d'hétéroscédasticité conditionnelle : régresse r_t^2
    sur ses propres retards. p-value faible -> présence de volatility
    clustering (la variance d'aujourd'hui dépend de la variance récente).
    """
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns.dropna(), nlags=lags)
    return {
        "arch_lm_stat": lm_stat,
        "arch_lm_pvalue": lm_pvalue,
        "has_arch_effect_5pct": bool(lm_pvalue < 0.05),
    }


def stationarity_test(returns: pd.Series) -> dict:
    """ADF : H0 = racine unitaire (non-stationnaire). On VEUT rejeter H0."""
    adf_stat, adf_pvalue, *_ = adfuller(returns.dropna())
    return {"adf_stat": adf_stat, "adf_pvalue": adf_pvalue, "is_stationary_5pct": bool(adf_pvalue < 0.05)}


def full_diagnostic(returns: pd.Series, name: str) -> dict:
    """Lance toute la batterie de tests et renvoie un dict plat (une ligne de tableau)."""
    r = returns.dropna()
    out = {"bar_type": name, "n_bars": len(r), "mean": r.mean(), "std": r.std()}
    out.update(jarque_bera_test(r))
    out.update(ljung_box_test(r))
    out.update(arch_test(r))
    out.update(stationarity_test(r))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/financial_bars")
    from bars import time_bars, tick_bars, volume_bars, dollar_bars, calibrate_threshold, bar_returns

    ticks = pd.read_pickle("/home/claude/financial_bars/ticks.pkl")
    tb = time_bars(ticks, freq="5min")
    n_target = len(tb)
    vb = volume_bars(ticks, calibrate_threshold(ticks, "volume", n_target))

    r_time = bar_returns(tb)
    r_vol = bar_returns(vb)

    print("=== Time bars ===")
    print(full_diagnostic(r_time, "time_bars"))
    print("\n=== Volume bars ===")
    print(full_diagnostic(r_vol, "volume_bars"))
