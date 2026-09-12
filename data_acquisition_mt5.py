"""
data_acquisition_mt5.py
=========================

Récupère de VRAIES données tick via le terminal MetaTrader5, au même
format que `generate_synthetic_ticks()` (data_acquisition.py) :
DataFrame indexé par timestamp avec les colonnes price, volume, sign, dollar.

IMPORTANT — CE MODULE NE PEUT PAS TOURNER DANS UN SANDBOX CLOUD
-------------------------------------------------------------------
Le package `MetaTrader5` ne fait pas d'appels réseau directs : il
communique en IPC avec le TERMINAL MetaTrader5 installé localement
(Windows, ou Linux via Wine) et connecté à un compte broker. Il n'y a
donc aucun moyen de le faire fonctionner dans un environnement
d'exécution cloud sans MT5 installé et un compte connecté.

-> Lance ce script SUR TA MACHINE, avec MT5 ouvert et connecté à ton
compte (démo ou réel), pas ici.

Installation (une fois) :
    pip install MetaTrader5

Ce module expose une seule fonction publique : `fetch_mt5_ticks`.
"""

import numpy as np
import pandas as pd


def fetch_mt5_ticks(
    symbol: str,
    start,
    end,
    mt5_terminal_path: str | None = None,
) -> pd.DataFrame:
    """
    Récupère les ticks MT5 pour `symbol` entre `start` et `end`
    (datetime, ou string parsable par pandas), et renvoie un DataFrame
    au même format que generate_synthetic_ticks() : price, volume,
    sign, dollar — indexé par timestamp.

    mt5_terminal_path : chemin vers terminal64.exe si MT5 n'est pas
        détecté automatiquement (utile si plusieurs installations
        MT5 coexistent, ex. plusieurs brokers).

    Détail de la reconstruction du signe (sign) :
    MT5 fournit un champ `flags` (bitmask) qui indique, quand le
    broker le transmet, si le tick correspond à un trade agressif
    acheteur ou vendeur (TICK_FLAG_BUY / TICK_FLAG_SELL). C'est plus
    fiable qu'une tick rule reconstruite depuis le seul prix. Quand
    l'info n'est pas disponible (fréquent sur le Forex, où on ne reçoit
    souvent que des cotations bid/ask sans trade "last"), on retombe
    sur la tick rule (voir `apply_tick_rule` dans data_acquisition.py).
    """
    import MetaTrader5 as mt5  # import local : évite un ImportError si le
                                 # module est juste consulté/testé sans MT5 installé

    initialized = mt5.initialize(mt5_terminal_path) if mt5_terminal_path else mt5.initialize()
    if not initialized:
        raise RuntimeError(
            f"Échec de connexion au terminal MT5 (mt5.last_error() = {mt5.last_error()}). "
            "Vérifie que le terminal MT5 est bien ouvert et connecté à un compte."
        )

    try:
        start_ts = pd.Timestamp(start).to_pydatetime()
        end_ts = pd.Timestamp(end).to_pydatetime()

        raw = mt5.copy_ticks_range(symbol, start_ts, end_ts, mt5.COPY_TICKS_ALL)
        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"Aucun tick renvoyé pour {symbol} entre {start} et {end} "
                f"(mt5.last_error() = {mt5.last_error()}). Vérifie le nom du "
                "symbole (visible dans le Market Watch de MT5) et que "
                "l'historique demandé est disponible chez ton broker."
            )

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["time_msc"], unit="ms")
        df = df.set_index("timestamp").sort_index()

        # --- Prix : trade 'last' si dispo (actions/CFD), sinon mid bid/ask (Forex) ---
        has_last = "last" in df.columns and (df["last"] > 0).any()
        if has_last:
            price = df["last"].where(df["last"] > 0, (df["bid"] + df["ask"]) / 2)
        else:
            price = (df["bid"] + df["ask"]) / 2

        # --- Volume : volume_real (lots fractionnaires) si dispo, sinon volume (entier) ---
        if "volume_real" in df.columns and (df["volume_real"] > 0).any():
            volume = df["volume_real"].where(df["volume_real"] > 0, df["volume"])
        else:
            volume = df["volume"].replace(0, 1)  # cotations pures Forex : pas de volume trade -> poids uniforme

        # --- Signe : flags MT5 (TICK_FLAG_BUY=0x20, TICK_FLAG_SELL=0x40) si dispo ---
        flags = df["flags"].to_numpy()
        buy_flag = (flags & mt5.TICK_FLAG_BUY) != 0
        sell_flag = (flags & mt5.TICK_FLAG_SELL) != 0
        has_flag = buy_flag | sell_flag

        sign = np.where(buy_flag, 1, np.where(sell_flag, -1, np.nan)).astype(float)

        if not has_flag.all():
            # Fallback tick rule pour les ticks sans flag buy/sell explicite
            from data_acquisition import apply_tick_rule
            tick_rule_sign = apply_tick_rule(price)
            sign = np.where(has_flag, sign, tick_rule_sign)

        out = pd.DataFrame(
            {"price": price.to_numpy(), "volume": volume.to_numpy(), "sign": sign.astype(int)},
            index=df.index,
        )
        out.index.name = "timestamp"
        out["dollar"] = out["price"] * out["volume"]
        return out

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    # Exemple d'utilisation — adapte symbol/dates à ton broker.
    # A lancer localement, MT5 ouvert et connecté.
    import datetime as dt

    end = dt.datetime.now()
    start = end - dt.timedelta(days=5)

    ticks = fetch_mt5_ticks("EURUSD", start, end)
    print(ticks.head())
    print(f"\n{len(ticks):,} ticks récupérés")
    ticks.to_pickle("/home/claude/financial_bars/ticks_mt5.pkl")