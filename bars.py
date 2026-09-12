"""
bars.py
=======

Construction des 4 types de "bars" standards à partir d'un flux de trades
(López de Prado, "Advances in Financial Machine Learning", chapitre 2).

IDÉE CENTRALE DU PROJET :
Un "bar" (OHLCV) est juste une façon d'agréger un flux continu de trades
en points discrets. Le choix classique — échantillonner par TEMPS
(1 bar / minute) — est arbitraire et a un défaut fondamental : le marché
ne "produit" pas de l'information à vitesse constante. À l'ouverture ou
lors d'une news, des centaines de trades porteurs d'information arrivent
en quelques secondes ; à midi, quasiment rien ne se passe pendant de
longues minutes. Échantillonner par temps revient donc à :
  - SUR-échantillonner les périodes calmes (bars quasi-vides, bruit pur)
  - SOUS-échantillonner les périodes actives (un seul bar absorbe un
    choc d'information massif -> perte d'info, rendements à queue
    lourde, autocorrélation artificielle)

Les bars alternatifs échantillonnent au contraire quand une certaine
QUANTITÉ D'ACTIVITÉ a été atteinte, pas quand un temps fixe s'est
écoulé. Idée : synchroniser l'échantillonnage sur le rythme de l'
horloge du marché ("volume clock" / "trade clock" / "dollar clock")
plutôt que sur l'horloge murale.

Chaque fonction ci-dessous prend le DataFrame de trades (format défini
dans data_acquisition.py) et renvoie un DataFrame de bars OHLCV avec :
    open, high, low, close, volume, dollar_volume, n_trades, vwap
indexé par le timestamp de FIN du bar.
"""

import numpy as np
import pandas as pd


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    """Agrège un groupe de trades en un bar OHLCV standard."""
    if len(group) == 0:
        return pd.Series(
            {
                "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                "volume": np.nan, "dollar_volume": np.nan, "n_trades": 0,
                "vwap": np.nan, "start_time": pd.NaT, "end_time": pd.NaT,
            }
        )
    return pd.Series(
        {
            "open": group["price"].iloc[0],
            "high": group["price"].max(),
            "low": group["price"].min(),
            "close": group["price"].iloc[-1],
            "volume": group["volume"].sum(),
            "dollar_volume": group["dollar"].sum(),
            "n_trades": len(group),
            "vwap": (group["price"] * group["volume"]).sum() / group["volume"].sum(),
            "start_time": group.index[0],
            "end_time": group.index[-1],
        }
    )


def time_bars(ticks: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    Bars échantillonnées par intervalle de temps fixe (la méthode
    classique, notre baseline de comparaison).

    freq : chaîne pandas ('1min', '5min', '30s', ...)
    """
    grouped = ticks.groupby(pd.Grouper(freq=freq))
    bars = grouped.apply(_aggregate_group, include_groups=False)
    bars = bars.dropna(subset=["open"])
    return bars.set_index("end_time")


def _threshold_bars(ticks: pd.DataFrame, cum_col: str, threshold: float) -> pd.DataFrame:
    """
    Fonction générique : découpe le flux de trades en groupes dès que
    la somme cumulée de `cum_col` (volume ou dollar) dépasse `threshold`.
    tick_bars, volume_bars et dollar_bars ne sont que des cas particuliers
    de cette même logique, seule la colonne cumulée change.
    """
    cum = ticks[cum_col].to_numpy().cumsum()
    # id du bar = combien de fois le seuil a été franchi jusqu'ici
    bar_id = (cum // threshold).astype(int)
    ticks = ticks.copy()
    ticks["_bar_id"] = bar_id
    grouped = ticks.groupby("_bar_id")
    bars = grouped.apply(_aggregate_group, include_groups=False)
    return bars.set_index("end_time")


def tick_bars(ticks: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """Un nouveau bar toutes les N transactions (peu importe leur taille)."""
    tmp = ticks.copy()
    tmp["_unit"] = 1
    return _threshold_bars(tmp.rename(columns={"_unit": "volume_count"}).assign(volume_count=1),
                            "volume_count", ticks_per_bar)


def volume_bars(ticks: pd.DataFrame, volume_per_bar: float) -> pd.DataFrame:
    """Un nouveau bar dès que la somme des VOLUMES échangés atteint le seuil."""
    return _threshold_bars(ticks, "volume", volume_per_bar)


def dollar_bars(ticks: pd.DataFrame, dollar_per_bar: float) -> pd.DataFrame:
    """
    Un nouveau bar dès que la somme des montants en $ échangés (price*volume)
    atteint le seuil. C'est le bar le plus robuste des trois "seuils
    d'activité" car il absorbe naturellement l'effet des variations de
    prix dans le temps : à prix double, un même volume économique
    correspond à un volume physique moitié moindre. Sur un historique
    long (plusieurs années), volume_bars dérive à cause du prix ;
    dollar_bars beaucoup moins.
    """
    return _threshold_bars(ticks, "dollar", dollar_per_bar)


def calibrate_threshold(ticks: pd.DataFrame, col: str, target_n_bars: int) -> float:
    """
    Choisit un seuil tel que le nombre de bars produits soit ~ égal à
    `target_n_bars`, pour pouvoir comparer les bars "à nombre de bars
    égal" avec les time bars (sinon on compare des séries de tailles
    différentes, ce qui biaise les tests statistiques).
    """
    total = ticks[col].sum()
    return total / target_n_bars


def bar_returns(bars: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Log-rendements entre bars consécutifs — c'est CE qu'on va tester statistiquement."""
    return np.log(bars[price_col]).diff().dropna()


if __name__ == "__main__":
    ticks = pd.read_pickle("/home/claude/financial_bars/ticks.pkl")

    tb = time_bars(ticks, freq="5min")
    print(f"time_bars (5min)   : {len(tb)} bars")

    n_target = len(tb)
    tick_thresh = calibrate_threshold(ticks.assign(count=1), "count", n_target)
    vol_thresh = calibrate_threshold(ticks, "volume", n_target)
    dol_thresh = calibrate_threshold(ticks, "dollar", n_target)

    tib = tick_bars(ticks, ticks_per_bar=int(tick_thresh))
    vb = volume_bars(ticks, volume_per_bar=vol_thresh)
    db = dollar_bars(ticks, dollar_per_bar=dol_thresh)

    print(f"tick_bars           : {len(tib)} bars (seuil={int(tick_thresh)} trades)")
    print(f"volume_bars          : {len(vb)} bars (seuil={vol_thresh:,.0f})")
    print(f"dollar_bars          : {len(db)} bars (seuil={dol_thresh:,.0f})")
