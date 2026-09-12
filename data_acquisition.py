"""
data_acquisition.py
====================

CE MODULE EST LE SEUL QUI DEVRAIT CHANGER SI TU BRANCHES DE VRAIES DONNÉES.
Tout le reste du projet (bars.py, imbalance_bars.py, statistical_tests.py)
ne dépend que du FORMAT en sortie de ce module, pas de sa source.

Format attendu, partout dans le projet : un DataFrame pandas avec un
DatetimeIndex (timestamp du trade) et les colonnes :
    - price   : prix du trade (float)
    - volume  : quantité échangée (float ou int)
    - dollar  : price * volume (calculé automatiquement)
    - sign    : +1 (achat / trade au ask) ou -1 (vente / trade au bid)
                Si tu as de vraies données sans le sign, utilise
                la tick rule (voir `apply_tick_rule` plus bas) pour
                le reconstruire à partir du seul prix.

Pourquoi des données SIMULÉES ici et pas du vrai tick data Yahoo/IB/etc. ?
---------------------------------------------------------------------
Le sandbox d'exécution n'a pas accès aux APIs de données de marché
(seuls pypi/npm/github sont autorisés en sortie réseau). Pour que tu
puisses lancer et comprendre TOUT le pipeline de bout en bout sans
dépendre d'une API externe, on simule un flux de trades avec une
microstructure volontairement réaliste :

  1. Intensité d'arrivée des trades non-homogène dans le temps
     (courbe en U intra-journalière : forte activité à l'ouverture et
     à la clôture, creux le midi — c'est le pattern observé sur
     quasiment tous les marchés actions).
  2. Un régime caché (Markov 2 états : calme / stressé) qui module
     à la fois l'intensité des arrivées, la taille des volumes et
     la persistance du signe des trades (autocorrélation du flux
     d'ordres — c'est CE mécanisme qui crée l'"information" que les
     bars alternatifs sont censés mieux capturer que les time bars).
  3. Un impact prix : chaque trade signé pousse le prix dans sa
     direction (price impact), ce qui crée un couplage réaliste
     entre volume, signe et rendement — sans ce couplage, les
     dollar/volume/imbalance bars n'auraient statistiquement aucune
     raison de se comporter différemment des time bars, et tout le
     projet perdrait son intérêt pédagogique.

>>> Pour brancher de VRAIES données plus tard :
Remplace `generate_synthetic_ticks()` par une fonction qui charge ton
CSV/parquet de tick data réel (Databento, LOBSTER, Kaggle, export de ton
broker/MT5 en mode tick, etc.) et renvoie un DataFrame au même format.
Rien d'autre à changer dans le reste du code.
"""

import numpy as np
import pandas as pd


def generate_synthetic_ticks(
    n_days: int = 5,
    seed: int = 42,
    base_price: float = 100.0,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    Simule un flux de trades tick-by-tick sur `n_days` journées de
    bourse (09:30-16:00, style actions US).

    Retourne un DataFrame indexé par timestamp avec les colonnes
    price, volume, sign, dollar.
    """
    rng = np.random.default_rng(seed)

    all_frames = []
    price = base_price

    for day in range(n_days):
        day_start = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        session_start = day_start + pd.Timedelta(hours=9, minutes=30)
        session_end = day_start + pd.Timedelta(hours=16)
        session_seconds = (session_end - session_start).total_seconds()

        # ---- 1) Régime caché (chaîne de Markov 2 états) ---------------
        # 0 = calme, 1 = stressé/news. On simule un régime par tranche
        # de ~10 minutes pour avoir des clusters de volatilité/volume.
        n_blocks = int(session_seconds // 600) + 1
        regimes = np.zeros(n_blocks, dtype=int)
        p_stay_calm, p_stay_stress = 0.92, 0.75  # persistance des régimes
        for i in range(1, n_blocks):
            if regimes[i - 1] == 0:
                regimes[i] = rng.random() > p_stay_calm
            else:
                regimes[i] = rng.random() < p_stay_stress
        block_len = 600.0

        # Biais directionnel persistant par bloc "stressé" : modélise un
        # agent informé qui accumule (ou distribue) une position pendant
        # toute la durée du régime — sans ce biais MARGINAL (et pas
        # seulement une auto-corrélation de signe), le déséquilibre net
        # sur une fenêtre reste centré sur 0 en espérance et les
        # imbalance/run bars n'ont rien de systématique à détecter.
        block_bias = np.zeros(n_blocks)  # 0.5 = pas de biais (P[achat]=50%)
        for i in range(n_blocks):
            if regimes[i] == 1:
                direction = rng.choice([-1, 1])
                block_bias[i] = 0.5 + direction * rng.uniform(0.10, 0.25)

        # ---- 2) Générer les timestamps (processus de Poisson) ---------
        # Intensité de base en U (ouverture/clôture actives), modulée
        # par le régime du bloc courant.
        t = 0.0
        timestamps = []
        block_regime_at_t = []
        lambda_calm, lambda_stress = 0.7, 3.5  # trades / seconde
        while t < session_seconds:
            frac = t / session_seconds
            u_shape = 1.0 + 1.8 * ((frac - 0.5) ** 2) * 4  # >=1, pic aux bords
            block_idx = min(int(t // block_len), n_blocks - 1)
            regime = regimes[block_idx]
            lam = (lambda_stress if regime == 1 else lambda_calm) * u_shape
            dt = rng.exponential(1.0 / lam)
            t += dt
            if t >= session_seconds:
                break
            timestamps.append(t)
            block_regime_at_t.append(regime)

        n = len(timestamps)
        block_regime_at_t = np.array(block_regime_at_t)

        # ---- 3) Signe des trades : AR(1) + biais de bloc ----------------
        # Deux mécanismes combinés, comme sur un marché réel :
        #  (a) auto-corrélation à court terme (les gros ordres sont
        #      "slicés" -> plusieurs trades de suite dans le même sens) ;
        #  (b) biais MARGINAL persistant pendant tout un bloc stressé
        #      (block_bias) -> c'est lui qui crée un vrai déséquilibre net,
        #      pas seulement du bruit auto-corrélé qui se compense en
        #      moyenne. Sans (b), les imbalance/run bars n'ont rien de
        #      systématique à détecter (voir note plus haut).
        block_idx_at_t = np.minimum((np.array(timestamps) // block_len).astype(int), n_blocks - 1)
        signs = np.zeros(n, dtype=int)
        s = rng.choice([-1, 1])
        for i in range(n):
            persistence = 0.35 if block_regime_at_t[i] == 1 else 0.05
            if rng.random() < persistence:
                pass  # garde le même signe que le trade précédent
            else:
                bias = block_bias[block_idx_at_t[i]]
                p_buy = bias if bias > 0 else 0.5
                s = 1 if rng.random() < p_buy else -1
            signs[i] = s

        # ---- 4) Volumes : log-normale, plus grosse en régime stressé --
        mu_vol = np.where(block_regime_at_t == 1, 5.3, 4.4)
        volumes = rng.lognormal(mean=mu_vol, sigma=0.6, size=n)
        volumes = np.round(volumes / 10) * 10  # lots ronds
        volumes = np.clip(volumes, 10, None)

        # ---- 5) Prix : marche aléatoire + impact + retour à la moyenne -
        # Le biais de bloc (point 3b) crée un vrai déséquilibre d'ordres,
        # mais un déséquilibre d'ordres ne doit PAS se traduire par une
        # tendance de prix qui explose exponentiellement sur toute une
        # journée : dans la réalité, l'impact prix d'un déséquilibre
        # d'ordres est un phénomène transitoire (le prix "digère" le flux
        # puis se stabilise), sans quoi un simple bloc de 10 minutes
        # biaisé à 60% ferait diverger le prix. On ajoute donc un faible
        # terme de retour à la moyenne (mean-reversion type OU) autour du
        # prix d'ouverture de la journée.
        vol_scale = np.where(block_regime_at_t == 1, 0.006, 0.002)
        noise = rng.normal(0, 1, n) * vol_scale
        impact = 0.02 * signs * (volumes / volumes.mean()) * vol_scale
        kappa = 0.001  # force de rappel vers le log-prix d'ouverture du jour
        log_open = np.log(price)
        log_price = log_open
        prices = np.empty(n)
        for i in range(n):
            log_price += noise[i] + impact[i] - kappa * (log_price - log_open)
            prices[i] = np.exp(log_price)
        prices = np.round(prices / tick_size) * tick_size
        price = prices[-1]

        idx = session_start + pd.to_timedelta(timestamps, unit="s")
        day_df = pd.DataFrame(
            {"price": prices, "volume": volumes, "sign": signs}, index=idx
        )
        all_frames.append(day_df)

    df = pd.concat(all_frames)
    df.index.name = "timestamp"
    df["dollar"] = df["price"] * df["volume"]
    return df


def apply_tick_rule(prices: pd.Series) -> np.ndarray:
    """
    Reconstruit le signe d'un trade à partir du SEUL prix, quand on
    n'a pas le carnet d'ordres (bid/ask) pour savoir si le trade a
    eu lieu au bid ou à l'ask. C'est la "tick rule" classique :

        b_t = b_{t-1}                 si price_t == price_{t-1}
        b_t = sign(price_t - price_{t-1})   sinon

    À utiliser sur de vraies données quand la colonne `sign` n'existe
    pas nativement (c'est le cas le plus fréquent en pratique).
    """
    diff = prices.diff()
    signs = np.sign(diff)
    signs = signs.replace(0, np.nan).ffill().fillna(1)
    return signs.to_numpy()


if __name__ == "__main__":
    ticks = generate_synthetic_ticks(n_days=5)
    print(ticks.head())
    print(f"\n{len(ticks):,} trades simulés sur {ticks.index.normalize().nunique()} jours")
    print(f"Volume total : {ticks['volume'].sum():,.0f}")
    print(f"Dollar volume total : {ticks['dollar'].sum():,.0f}")
    ticks.to_pickle("/home/claude/financial_bars/ticks.pkl")
