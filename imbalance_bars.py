"""
imbalance_bars.py
==================

"Information-driven bars" (López de Prado, AFML chap. 2.3). C'est la
partie la plus avancée et la plus rarement implémentée du projet —
c'est elle qui fait la différence sur un dossier de candidature.

IDÉE DE DÉPART
---------------
Les bars du fichier bars.py (tick/volume/dollar) échantillonnent selon
une quantité d'ACTIVITÉ. Mais toute activité n'est pas porteuse
d'information : un flux de trades équilibré (autant d'achats que de
ventes) peut être très actif sans que le prix ne "sache" dans quelle
direction aller. À l'inverse, un DÉSÉQUILIBRE persistant entre achats
et ventes (imbalance) est le signe qu'un agent informé est en train
d'exécuter une position — c'est justement l'instant où on veut
échantillonner un bar.

DÉFINITIONS (notations AFML)
-----------------------------
Pour chaque trade t, on définit le signe b_t ∈ {-1, +1} (tick rule si
pas de carnet d'ordres — voir data_acquisition.apply_tick_rule).

Le déséquilibre cumulé à l'instant T (depuis le début du bar courant) :
    theta_T = sum_{t=1}^{T} b_t * v_t
où v_t est un "poids" du trade t :
    - imbalance bar EN TICKS   : v_t = 1
    - imbalance bar EN VOLUME  : v_t = volume_t
    - imbalance bar EN DOLLAR  : v_t = dollar_t   <-- celle qu'on implémente

On ferme le bar dès que |theta_T| dépasse un seuil ATTENDU E_0[theta] :

    E_0[theta] = E_0[T] * |2*P[b=+1] - 1| * E_0[v]     (approximation)

Ce seuil n'est PAS fixe comme pour volume_bars — il est ESTIMÉ de façon
ADAPTATIVE par moyenne mobile exponentielle (EWMA) sur les tailles et
imbalances des bars déjà formés :
    - E[T]     : EWMA de la longueur (en # de trades) des N derniers bars
    - E[b]     : EWMA de theta/T des N derniers bars (imbalance moyenne signée)
Le bar suivant se ferme dès que |theta_cumulé| >= E[T] * |E[b]|.

C'est cette adaptativité qui rend les imbalance bars "intelligentes" :
le seuil se resserre automatiquement quand le marché devient
directionnel (bars plus courts = échantillonnage plus fin exactement
quand l'info arrive), et se desserre en marché équilibré.

RUN BARS
--------
Variante : au lieu de sommer les signes (theta = sum b_t*v_t), on
détecte des "runs" — des séquences consécutives de trades du même
signe — et on ferme le bar quand la plus longue série d'achats OU de
ventes consécutives (pondérée) dépasse un seuil attendu, calibré de la
même façon par EWMA. Les runs captent un pattern différent de
l'imbalance : une alternance rapide achat/vente/achat/vente peut avoir
un theta proche de 0 (pas d'imbalance) tout en ayant des runs courts
mais très fréquents — signe d'un carnet d'ordres agité même sans
direction nette.
"""

import numpy as np
import pandas as pd


def _ewma(arr, span):
    """EWMA simple sur un array 1D (dernière valeur = estimateur courant)."""
    if len(arr) == 0:
        return np.nan
    alpha = 2 / (span + 1)
    w = (1 - alpha) ** np.arange(len(arr))[::-1]
    return float(np.sum(w * arr) / np.sum(w))


def dollar_imbalance_bars(
    ticks: pd.DataFrame,
    init_expected_ticks: int = 300,
    ewma_span_T: int = 20,
) -> pd.DataFrame:
    """
    Dollar Imbalance Bars (DIB) avec seuil adaptatif par EWMA.

    ticks doit contenir les colonnes price, volume, dollar, sign.
    init_expected_ticks : sert d'amorçage (avant d'avoir assez
        d'historique) ET de span pour l'EWMA "au niveau tick" de b_t
        et v_t (voir note d'implémentation ci-dessous).

    NOTE D'IMPLÉMENTATION IMPORTANTE
    ---------------------------------
    Une première version naïve estime E[b] (l'imbalance moyenne signée)
    en faisant l'EWMA des theta/T *réalisés* sur les derniers bars
    fermés. Ce choix est INSTABLE en pratique : theta/T sur un bar de
    quelques trades est un estimateur très bruité (un simple hasard de
    séquence de signes peut le faire osciller violemment), et comme il
    ré-alimente directement le seuil du bar suivant, on obtient une
    boucle de rétroaction qui fait soit s'effondrer les bars vers une
    taille minimale, soit au contraire diverger vers des bars géants.

    La version stable (et plus fidèle à l'esprit d'AFML) estime plutôt
    P[b=+1] et E[v] par EWMA calculée en CONTINU, tick par tick, sur
    une fenêtre glissante de taille ~ E[T] : on lisse sur des milliers
    d'observations plutôt que sur une poignée de bars, ce qui rend
    l'estimateur beaucoup moins bruité.
    """
    signed = ticks["sign"].to_numpy().astype(float)
    dollars = ticks["dollar"].to_numpy()
    n = len(ticks)

    span = max(init_expected_ticks, 10)
    alpha = 2 / (span + 1)
    MAX_BAR_LEN = 20 * init_expected_ticks  # garde-fou anti-runaway, cf. note dans le module

    bar_start = 0
    bar_lengths = []
    bar_edges = []

    theta = 0.0
    expected_T = float(init_expected_ticks)
    p_buy_ewma = 0.5   # P[b=+1], initialisé à "flux équilibré"
    v_ewma = float(dollars[:1000].mean()) if n >= 1000 else float(dollars.mean())

    i = 0
    while i < n:
        theta += signed[i] * dollars[i]
        length_so_far = i - bar_start + 1

        # Mise à jour continue (tick par tick) des estimateurs — c'est
        # cette continuité qui stabilise le seuil par rapport à la
        # version "EWMA sur bars fermés".
        b_i = 1.0 if signed[i] > 0 else 0.0
        p_buy_ewma = alpha * b_i + (1 - alpha) * p_buy_ewma
        v_ewma = alpha * dollars[i] + (1 - alpha) * v_ewma

        expected_b_per_trade = abs(2 * p_buy_ewma - 1) * v_ewma
        threshold = expected_T * max(expected_b_per_trade, 1e-8)

        forced_close = length_so_far >= MAX_BAR_LEN

        if (abs(theta) >= threshold and length_so_far >= 10) or forced_close:
            bar_lengths.append(length_so_far)
            bar_edges.append(i)
            raw_expected_T = _ewma(np.array(bar_lengths[-ewma_span_T:]), ewma_span_T)
            # Plancher ET plafond sur expected_T : sans plancher, une
            # première fermeture précoce (déclenchée par du bruit) fait
            # chuter expected_T, ce qui abaisse le seuil suivant, ce qui
            # déclenche une fermeture encore plus précoce -> boucle de
            # rétroaction qui fait s'effondrer TOUS les bars suivants à
            # la taille minimale. Le plancher casse cette boucle.
            expected_T = float(np.clip(raw_expected_T, 0.3 * init_expected_ticks, 5 * init_expected_ticks))

            bar_start = i + 1
            theta = 0.0

        i += 1

    return _bars_from_edges(ticks, bar_edges)


def run_bars(
    ticks: pd.DataFrame,
    init_expected_ticks: int = 300,
    ewma_span_T: int = 20,
) -> pd.DataFrame:
    """
    Dollar Run Bars : ferme le bar quand max(run acheteur cumulé,
    run vendeur cumulé), en $, dépasse un seuil attendu (EWMA).

    Contrairement à l'imbalance bar (qui somme TOUS les signes, achats
    et ventes se compensant), le run bar somme séparément les $ du
    côté acheteur et du côté vendeur, et ne regarde QUE le plus grand
    des deux. Un flux qui alterne achat/vente en gros blocs (buy-buy-buy-
    sell-sell-sell) produit des runs longs même si l'imbalance nette
    sur la fenêtre peut être faible.
    """
    signed = ticks["sign"].to_numpy().astype(float)
    dollars = ticks["dollar"].to_numpy()
    n = len(ticks)

    span = max(init_expected_ticks, 10)
    alpha = 2 / (span + 1)
    MAX_BAR_LEN = 20 * init_expected_ticks  # même garde-fou anti-runaway que pour DIB

    bar_start = 0
    bar_lengths = []
    bar_edges = []

    buy_run, sell_run = 0.0, 0.0
    expected_T = float(init_expected_ticks)
    p_buy_ewma = 0.5
    v_ewma = float(dollars[:1000].mean()) if n >= 1000 else float(dollars.mean())

    i = 0
    while i < n:
        d = signed[i] * dollars[i]
        if d > 0:
            buy_run += d
        else:
            sell_run += -d
        run_max = max(buy_run, sell_run)
        length_so_far = i - bar_start + 1

        b_i = 1.0 if signed[i] > 0 else 0.0
        p_buy_ewma = alpha * b_i + (1 - alpha) * p_buy_ewma
        v_ewma = alpha * dollars[i] + (1 - alpha) * v_ewma

        # "run attendu" par trade = côté dominant (max(P_buy, P_sell)) * montant moyen
        expected_run_per_trade = max(p_buy_ewma, 1 - p_buy_ewma) * v_ewma
        threshold = expected_T * max(expected_run_per_trade, 1e-8)

        forced_close = length_so_far >= MAX_BAR_LEN

        if (run_max >= threshold and length_so_far >= 10) or forced_close:
            bar_lengths.append(length_so_far)
            bar_edges.append(i)

            raw_expected_T = _ewma(np.array(bar_lengths[-ewma_span_T:]), ewma_span_T)
            expected_T = float(np.clip(raw_expected_T, 0.3 * init_expected_ticks, 5 * init_expected_ticks))

            bar_start = i + 1
            buy_run, sell_run = 0.0, 0.0

        i += 1

    return _bars_from_edges(ticks, bar_edges)


def _bars_from_edges(ticks: pd.DataFrame, edges: list) -> pd.DataFrame:
    """Reconstruit un DataFrame OHLCV standard à partir d'une liste d'indices de fin de bar."""
    from bars import _aggregate_group

    rows = []
    start = 0
    for e in edges:
        group = ticks.iloc[start : e + 1]
        rows.append(_aggregate_group(group))
        start = e + 1
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["open"])
    return out.set_index("end_time")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/financial_bars")
    ticks = pd.read_pickle("/home/claude/financial_bars/ticks.pkl")

    dib = dollar_imbalance_bars(ticks)
    print(f"dollar_imbalance_bars : {len(dib)} bars "
          f"(taille moyenne = {len(ticks)/len(dib):.0f} trades/bar, "
          f"std = {dib['n_trades'].std():.0f})")

    rb = run_bars(ticks)
    print(f"run_bars              : {len(rb)} bars "
          f"(taille moyenne = {len(ticks)/len(rb):.0f} trades/bar, "
          f"std = {rb['n_trades'].std():.0f})")
