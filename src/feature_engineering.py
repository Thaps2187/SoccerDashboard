import numpy as np
import pandas as pd
from collections import defaultdict
from sqlalchemy import text
from pathlib import Path
import math

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / \
    "final_K16_HA60_SR30_xg0_L101_rest1_cc0_lr0.04_it300_l20.0_calisotonic"
FEATURES_TXT = MODEL_DIR / "features.txt"
EXPECTED = [ln.strip() for ln in open(FEATURES_TXT) if ln.strip()]

FEATURE_COLUMNS = [ln.strip() for ln in open(MODEL_DIR / "features.txt") if ln.strip()]

class FeatureEngineer:
    def __init__(self, engine, elo_cfg=None, form_n=5, h2h_n=5):
        self.engine = engine
        self.form_n = int(form_n)
        self.h2h_n = int(h2h_n)
        self.elo_cfg = elo_cfg or {
            "start": 1500.0,
            "home_adv": 60.0,
            "K": 16.0,
            "season_regress": 0.3,
        }
        self._elo_cache = {}

    def build(self, upcoming_df):
        
        self._check_columns(
            upcoming_df,
            {"fixture_id","league_id","season_year","match_date","home_team_id","away_team_id"},
        )

        df = upcoming_df.copy()
        
        df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")


        rows = []
        for _, r in df.iterrows():
            fixture_id = int(r["fixture_id"])
            league_id  = int(r["league_id"])
            home_id    = int(r["home_team_id"])
            away_id    = int(r["away_team_id"])
            cutoff     = pd.Timestamp(r["match_date"])

            # Elo features
            elo_h, elo_a, elo_diff = self._elo_pair(league_id, home_id, away_id, cutoff)


            gf_h5, ga_h5   = self._last_n_gf_ga(home_id, cutoff, 5)
            gf_a5, ga_a5   = self._last_n_gf_ga(away_id, cutoff, 5)
            gf_h10, ga_h10 = self._last_n_gf_ga(home_id, cutoff, 10)
            gf_a10, ga_a10 = self._last_n_gf_ga(away_id, cutoff, 10)

            # 1) Team form/rest
            home_form = self._team_form_stats(home_id, cutoff)
            away_form = self._team_form_stats(away_id, cutoff)

            # 2) H2H
            h2h = self._h2h_stats(home_id, away_id, cutoff)

            # 3) Assemble feature row
            row = {
                "fixture_id": fixture_id,
                "rest_home_days": home_form["days_since"],
                "rest_away_days": away_form["days_since"],

                "form_home_pts": home_form["pts_sum"],
                "form_away_pts": away_form["pts_sum"],
                "form_home_gf":  home_form["gf_avg"],
                "form_home_ga":  home_form["ga_avg"],
                "form_away_gf":  away_form["gf_avg"],
                "form_away_ga":  away_form["ga_avg"],

                # diffs (home - away).
                "form_pts_diff": home_form["pts_avg"] - away_form["pts_avg"],
                "form_gf_diff":  home_form["gf_avg"]  - away_form["gf_avg"],
                "form_ga_diff":  away_form["ga_avg"]  - home_form["ga_avg"],

                "h2h_home_winrate": h2h["home_winrate"],
                "h2h_gd_avg":       h2h["gd_avg"],

                "gf_home_L5":  gf_h5,  "ga_home_L5":  ga_h5,
                "gf_away_L5":  gf_a5,  "ga_away_L5":  ga_a5,
                "gf_home_L10": gf_h10, "ga_home_L10": ga_h10,
                "gf_away_L10": gf_a10, "ga_away_L10": ga_a10,

                # Elo
                "elo_home": elo_h,
                "elo_away": elo_a,
                "elo_diff": elo_diff,

                "h2h_home_winrate": self._h2h_winrate(home_id, away_id, cutoff)
    
            }
            rows.append(row)

        X = pd.DataFrame(rows).set_index("fixture_id")

        # 4) Final clean-up:
        X = self._finalize(X)
        X = self._enforce_model_schema(X, upcoming_df)
        # Ensure exact column order expected by the model

        X = X.reindex(columns=FEATURE_COLUMNS).fillna(0.0)
        return X

    def _enforce_model_schema(self, X: pd.DataFrame, upcoming: pd.DataFrame):
        X = X.copy()

        # 1) rename to match training names
        rename_map = {
            "rest_home_days": "rest_days_home",
            "rest_away_days": "rest_days_away",
        }
        X.rename(columns=rename_map, inplace=True)

        # 2) easy derived fields many pipelines expect
        if "rest_diff" not in X and {"rest_days_home","rest_days_away"} <= set(X.columns):
            X["rest_diff"] = X["rest_days_home"] - X["rest_days_away"]

        # month / round_number / league_id come from upcoming
        md = pd.to_datetime(upcoming["match_date"], errors="coerce", utc=True)
        if "month" not in X:
            X["month"] = md.dt.month.fillna(0).astype(int)
        if "round_number" not in X:
            if "match_day" in upcoming.columns:
                rn = pd.to_numeric(
                    upcoming["match_day"].astype(str).str.replace("MD","",regex=False),
                    errors="coerce"
                ).fillna(0).astype(int)
                X["round_number"] = rn.values
            else:
                X["round_number"] = 0
        if "league_id" not in X and "league_id" in upcoming.columns:
            X["league_id"] = pd.to_numeric(upcoming["league_id"], errors="coerce").fillna(0).astype(int)

        # home_adv 
        if "home_adv" not in X:
            X["home_adv"] = 1.0

        # 3) neutral defaults for features you don’t compute yet
        defaults = {
            # Elo related
            "elo_home": 1500.0, "elo_away": 1500.0, "elo_diff": 0.0,
            # rolling GF/GA
            "gf_home_L5": 0.0, "ga_home_L5": 0.0, "gf_away_L5": 0.0, "ga_away_L5": 0.0,
            "gf_home_L10": 0.0, "ga_home_L10": 0.0, "gf_away_L10": 0.0, "ga_away_L10": 0.0,
            # shots & shots-on-target
            "sh_home_L5": 0.0, "sh_away_L5": 0.0, "sh_home_L10": 0.0, "sh_away_L10": 0.0,
            "sot_home_L5": 0.0, "sot_away_L5": 0.0,
        }
        for col in EXPECTED:
            if col not in X.columns:
                X[col] = defaults.get(col, 0.0)

        # 4) enforce numeric types & column order the model expects
        for col in EXPECTED:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        return X.reindex(columns=EXPECTED)

    # --------------------- helpers ---------------------

    def _last_n_gf_ga(self, team_id: int, cutoff_ts, n: int) -> tuple[float,float]:
        with self.engine.begin() as conn:
            df = pd.read_sql_query(text(f"""
                SELECT home_team_id, away_team_id, home_goals, away_goals
                FROM matches
                WHERE status='FT'
                AND (home_team_id=:tid OR away_team_id=:tid)
                AND match_date < :cutoff
                ORDER BY match_date DESC
                LIMIT {n}
            """), conn, params={"tid": team_id, "cutoff": pd.Timestamp(cutoff_ts).isoformat()})
        if df.empty:
            return 0.0, 0.0
        is_home = df["home_team_id"].eq(team_id)
        gf = np.where(is_home, df["home_goals"], df["away_goals"]).astype(float)
        ga = np.where(is_home, df["away_goals"], df["home_goals"]).astype(float)
        return float(gf.mean()), float(ga.mean())
    
    def _h2h_winrate(self, home_id: int, away_id: int, cutoff_ts, n: int | None = None) -> float:
        """Win rate for the *upcoming home team* vs opponent over last n H2H before cutoff."""
        n = n or self.h2h_n
        with self.engine.begin() as conn:
            df = pd.read_sql_query(text(f"""
                SELECT home_team_id, away_team_id, home_goals, away_goals
                FROM matches
                WHERE status='FT' AND match_date < :cutoff
                  AND (
                    (home_team_id=:h AND away_team_id=:a) OR
                    (home_team_id=:a AND away_team_id=:h)
                  )
                ORDER BY match_date DESC
                LIMIT {n}
            """), conn, params={"h": home_id, "a": away_id,
                                "cutoff": pd.Timestamp(cutoff_ts).isoformat()})
        if df.empty:
            return 0.5  # neutral prior when no H2H history

        is_home_side = df["home_team_id"].eq(home_id)
        gf = np.where(is_home_side, df["home_goals"], df["away_goals"]).astype(float)
        ga = np.where(is_home_side, df["away_goals"], df["home_goals"]).astype(float)
        return float((gf > ga).mean())

    def _team_form_stats(self, team_id, cutoff_ts):
        
        sql = text(f"""
            SELECT match_date, home_team_id, away_team_id, home_goals, away_goals
            FROM matches
            WHERE status = 'FT'
              AND (home_team_id = :tid OR away_team_id = :tid)
              AND match_date < :cutoff
            ORDER BY match_date DESC
            LIMIT {self.form_n}
        """)

        with self.engine.begin() as conn:
            df = pd.read_sql_query(sql, conn, params={"tid": team_id, "cutoff": cutoff_ts.isoformat()})

        if df.empty:
            return {"days_since": np.nan, "pts_sum": 0.0, "pts_avg": 0.0, "gf_avg": 0.0, "ga_avg": 0.0}

        df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")

        is_home = df["home_team_id"].eq(team_id)
        gf = np.where(is_home, df["home_goals"], df["away_goals"]).astype(float)
        ga = np.where(is_home, df["away_goals"], df["home_goals"]).astype(float)

        # Points per match
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))

        last_date  = df["match_date"].max()
        days_since = float((cutoff_ts - last_date).days)

        return {
            "days_since": days_since,
            "pts_sum": float(pts.sum()),
            "pts_avg": float(pts.mean()),
            "gf_avg":  float(gf.mean()),
            "ga_avg":  float(ga.mean()),
        }

    def _h2h_stats(self, home_id, away_id, cutoff_ts):

        sql = text(f"""
            SELECT match_date, home_team_id, away_team_id, home_goals, away_goals
            FROM matches
            WHERE status = 'FT'
              AND match_date < :cutoff
              AND (
                (home_team_id = :h AND away_team_id = :a) OR
                (home_team_id = :a AND away_team_id = :h)
              )
            ORDER BY match_date DESC
            LIMIT {self.h2h_n}
        """)
        with self.engine.begin() as conn:
            df = pd.read_sql_query(sql, conn, params={"h": home_id, "a": away_id, "cutoff": cutoff_ts.isoformat()})

        if df.empty:
            return {"home_winrate": 0.0, "gd_avg": 0.0}

        is_home_side = df["home_team_id"].eq(home_id)
        gf = np.where(is_home_side, df["home_goals"], df["away_goals"]).astype(float)
        ga = np.where(is_home_side, df["away_goals"], df["home_goals"]).astype(float)

        wins = float((gf > ga).sum())
        total = float(len(df))
        winrate = wins / total if total else 0.0
        gd_avg  = float((gf - ga).mean()) if total else 0.0

        return {"home_winrate": winrate, "gd_avg": gd_avg}

    def _finalize(self, X):

        X = X.copy().replace([np.inf, -np.inf], np.nan)

        X.rename(columns={
                    "rest_home_days": "rest_days_home",
                    "rest_away_days": "rest_days_away",
                }, inplace=True)

        # Neutral defaults
        defaults = {
        "rest_days_home": 7.0, "rest_days_away": 7.0, "rest_diff": 0.0,
        "h2h_home_winrate": 0.5,
        "elo_home": 1500.0, "elo_away": 1500.0, "elo_diff": 0.0,
        "gf_home_L5": 0.0, "ga_home_L5": 0.0, "gf_away_L5": 0.0, "ga_away_L5": 0.0,
        "gf_home_L10": 0.0, "ga_home_L10": 0.0, "gf_away_L10": 0.0, "ga_away_L10": 0.0,
        "sh_home_L5": 0.0, "sh_away_L5": 0.0, "sh_home_L10": 0.0, "sh_away_L10": 0.0,
        "sot_home_L5": 0.0, "sot_away_L5": 0.0,
        "league_id": 0, "month": 0, "round_number": 0, "home_adv": 1.0,
        }

        # keep exactly what the model expects (use your features.txt -> FEATURE_COLUMNS)
        X = X.reindex(columns=FEATURE_COLUMNS, fill_value=np.nan).fillna(defaults)

        # clip ranges
        if "rest_days_home" in X:
            X["rest_days_home"] = pd.to_numeric(X["rest_days_home"], errors="coerce").fillna(7.0).clip(0, 60)
        if "rest_days_away" in X:
            X["rest_days_away"] = pd.to_numeric(X["rest_days_away"], errors="coerce").fillna(7.0).clip(0, 60)
        if "h2h_home_winrate" in X:
            X["h2h_home_winrate"] = pd.to_numeric(X["h2h_home_winrate"], errors="coerce").fillna(0.5).clip(0, 1)

        return X

    # --- ELO core -------------------------------------------------------------

    def _elo_expected(self, Ra, Rb):
        """Return expected score for A vs B using chess-style logistic (400 scale)."""
        return 1.0 / (1.0 + 10 ** (-(Ra - Rb) / 400.0))

    def _elo_regress(self, R):
        """Pull rating toward start by season_regress."""
        start = self.elo_cfg["start"]
        lam = self.elo_cfg["season_regress"]
        return start + (R - start) * (1.0 - lam)

    def _ensure_elo_until(self, league_id: int, until_ts):
        """
        Compute/update Elo ratings for a league up to 'until_ts' using FT matches.
        Uses a simple forward-only cache.
        """
        until_ts = pd.to_datetime(until_ts, utc=True)

        state = self._elo_cache.get(league_id)
        if state is None:
            state = {
                "until": pd.Timestamp.min.tz_localize("UTC"),
                "ratings": defaultdict(lambda: float(self.elo_cfg["start"])),
                "last_season": {},  # team_id -> season_year we last saw
            }
            self._elo_cache[league_id] = state

        # nothing to do if we’re already past this cutoff
        if state["until"] >= until_ts:
            return

        # pull matches strictly before the new cutoff, and after what we've processed
        with self.engine.begin() as conn:
            df = pd.read_sql_query(
                text("""
                    SELECT season_year, match_date,
                           home_team_id, away_team_id,
                           home_goals, away_goals
                    FROM matches
                    WHERE league_id = :lg
                      AND status = 'FT'
                      AND match_date > :from_dt
                      AND match_date < :to_dt
                    ORDER BY match_date ASC
                """),
                conn,
                params={
                    "lg": league_id,
                    "from_dt": state["until"].isoformat() if state["until"] != pd.Timestamp.min.tz_localize("UTC") else "1900-01-01T00:00:00Z",
                    "to_dt": until_ts.isoformat(),
                },
            )

        K = float(self.elo_cfg["K"])
        H = float(self.elo_cfg["home_adv"])

        for _, r in df.iterrows():
            ssn = int(r["season_year"])
            h = int(r["home_team_id"])
            a = int(r["away_team_id"])
            hg = int(r["home_goals"])
            ag = int(r["away_goals"])

            # season regression when a team first appears in a new season
            for t in (h, a):
                last = state["last_season"].get(t)
                if last is None or last != ssn:
                    state["ratings"][t] = self._elo_regress(state["ratings"][t])
                    state["last_season"][t] = ssn

            Rh = state["ratings"][h]
            Ra = state["ratings"][a]

            # expected for home with home advantage
            Eh = self._elo_expected(Rh + H, Ra)
            # realized score (win=1, draw=0.5, loss=0)
            if hg > ag:
                Sh = 1.0
            elif hg == ag:
                Sh = 0.5
            else:
                Sh = 0.0

            # update
            Rh_new = Rh + K * (Sh - Eh)
            Ra_new = Ra + K * ((1.0 - Sh) - (1.0 - Eh))  # symmetric update

            state["ratings"][h] = Rh_new
            state["ratings"][a] = Ra_new

        # advance the watermark
        state["until"] = until_ts

    def _elo_pair(self, league_id: int, home_id: int, away_id: int, cutoff_ts):
        """Return (elo_home, elo_away, elo_diff) as of cutoff."""
        self._ensure_elo_until(league_id, cutoff_ts)
        ratings = self._elo_cache[league_id]["ratings"]
        Rh = float(ratings.get(int(home_id), self.elo_cfg["start"]))
        Ra = float(ratings.get(int(away_id), self.elo_cfg["start"]))
        return Rh, Ra, Rh - Ra
    

    @staticmethod
    def _check_columns(df, required):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in upcoming_df: {sorted(missing)}")
