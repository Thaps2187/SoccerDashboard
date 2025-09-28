#!/usr/bin/env python3
"""
load_db.py — Copy tables from SQLite (soccer.db) to PostgreSQL safely.

Usage (from your project root, with venv activated):
  # 1) Make sure POSTGRES_URL is set (include ?sslmode=require for Render/Neon):
  # export POSTGRES_URL='postgresql+psycopg2://USER:PASS@host:5432/DB?sslmode=require'
  # optional (defaults to ./soccer.db):
  # export SQLITE_URL='sqlite:///path/to/soccer.db'

  # 2) Run:
  python src/load_db.py
  # overwrite tables instead of skipping:
  python src/load_db.py --truncate
  # just copy a couple of tables:
  python src/load_db.py --only leagues,teams
  # skip specific tables:
  python src/load_db.py --skip fixture_predictions
  # tweak batch size:
  python src/load_db.py --chunk 10000

Requires:
  SQLAlchemy>=2.0, psycopg2-binary>=2.9, (optional) python-dotenv>=1.0
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# SQLAlchemy imports
from sqlalchemy import (
    MetaData, create_engine, inspect, select, text
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import func


# -------------------------
# Config / CLI
# -------------------------
DEFAULT_LOAD_ORDER = [
    # parents → children
    "leagues",
    "teams",
    "players",
    "player_seasons",
    "matches",
    "match_stats",
    "standings",
    "squad_membership",
    "fixture_predictions"  # handled separately; see notes below
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Copy tables from SQLite to Postgres safely.")
    ap.add_argument("--truncate", action="store_true",
                    help="Truncate each table in Postgres before copying.")
    ap.add_argument("--chunk", type=int, default=5000,
                    help="Batch size for inserts (default: 5000).")
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated list of tables to copy. If omitted, uses default order + leftovers.")
    ap.add_argument("--skip", type=str, default="fixture_predictions",
                    help="Comma-separated list of tables to skip (default: fixture_predictions).")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose logging for first batch per table.")
    return ap.parse_args()


# -------------------------
# Env / Engines
# -------------------------
def get_urls() -> Tuple[str, str]:
    # Try to load .env if present, overriding stale environment (helps in notebooks)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    default_sqlite_path = Path.cwd() / "soccer.db"
    sqlite_url = os.getenv("SQLITE_URL", f"sqlite:///{default_sqlite_path}")
    pg_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not pg_url:
        raise SystemExit("Set POSTGRES_URL (or DATABASE_URL) first.")
    return sqlite_url, pg_url


def make_engines(sqlite_url: str, pg_url: str) -> Tuple[Engine, Engine]:
    s_eng = create_engine(sqlite_url)
    p_eng = create_engine(pg_url, pool_pre_ping=True)
    # quick connectivity check
    with s_eng.connect() as c:
        assert c.exec_driver_sql("SELECT 1").scalar() == 1, "SQLite not reachable"
    with p_eng.connect() as c:
        assert c.exec_driver_sql("SELECT 1").scalar() == 1, "Postgres not reachable"
    print("SQLite OK -> 1")
    print("Postgres OK -> 1")
    return s_eng, p_eng


# -------------------------
# Helpers
# -------------------------
def list_sqlite_tables(s_eng: Engine) -> List[str]:
    ins = inspect(s_eng)
    return ins.get_table_names()


def create_pg_table_from_sqlite(s_eng: Engine, p_eng: Engine, table_name: str) -> None:
    """
    Reflect exactly one table from SQLite and create it in Postgres if missing.
    """
    md = MetaData()
    md.reflect(bind=s_eng, only=[table_name])
    if table_name not in md.tables:
        print(f"  (missing in SQLite; skip create) {table_name}")
        return
    md.create_all(p_eng)


def truncate_table(p_eng: Engine, table_name: str) -> None:
    with p_eng.begin() as conn:
        conn.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE'))


def count_rows(eng: Engine, table_name: str) -> int:
    md = MetaData()
    md.reflect(bind=eng, only=[table_name])
    tbl = md.tables[table_name]
    with eng.connect() as conn:
        return conn.execute(select(func.count()).select_from(tbl)).scalar_one()


def get_pg_table_and_allowed_cols(p_eng: Engine, table_name: str):
    """
    Reflect the PG table object and compute allowed column set = all columns - generated ALWAYS columns.
    """
    md_pg = MetaData()
    md_pg.reflect(bind=p_eng, only=[table_name])
    pg_table = md_pg.tables[table_name]

    with p_eng.connect() as c:
        gen_cols = {
            r[0]
            for r in c.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public'
                      AND table_name = :t
                      AND is_generated = 'ALWAYS'
                """),
                {"t": table_name},
            )
        }
    allowed = {col.name for col in pg_table.columns} - gen_cols
    return pg_table, allowed, gen_cols


# "186 cm" → 186, "80 kg" → 80
_DIGITS = re.compile(r"\d+")


def _to_int_or_none(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    m = _DIGITS.search(str(v))
    return int(m.group()) if m else None


def clean_row_for_table(table_name: str, d: dict) -> dict:
    """
    Table-specific cleaning & key dropping.
    - players: coerce height/weight strings to integers
    - standings: drop generated columns so PG computes them
    - fixture_predictions: (if created with generated cols in PG) drop label cols
    """
    if table_name == "players":
        for k in ("height_cm", "height", "height_in_cm"):
            if k in d:
                d[k] = _to_int_or_none(d[k])
        for k in ("weight_kg", "weight", "weight_in_kg"):
            if k in d:
                d[k] = _to_int_or_none(d[k])

    if table_name == "standings":
        for k in ("goal_diff", "points", "match_played"):
            d.pop(k, None)


    return d


def copy_table(s_eng: Engine, p_eng: Engine, table_name: str, chunk: int, verbose: bool = False) -> int:
    """
    Copy rows from SQLite -> PG for a single table.
    - Reads via SQLite-reflected table
    - Inserts via Postgres-reflected table (with generated cols filtered out)
    """
    # reflect source (SQLite)
    md_src = MetaData()
    md_src.reflect(bind=s_eng, only=[table_name])
    if table_name not in md_src.tables:
        print("  (missing in SQLite; skipping)")
        return 0
    src_table = md_src.tables[table_name]

    # reflect target (Postgres)
    pg_table, allowed, gen_cols = get_pg_table_and_allowed_cols(p_eng, table_name)

    total = 0
    with s_eng.connect() as s_conn, p_eng.begin() as p_conn:
        res = s_conn.execute(select(src_table))
        first_batch = True
        while True:
            batch = res.fetchmany(chunk)
            if not batch:
                break
            rows = []
            for r in batch:
                d = dict(r._mapping)
                d = clean_row_for_table(table_name, d)
                d = {k: v for k, v in d.items() if k in allowed}  # drop generated & unknown cols
                rows.append(d)

            if rows:
                if verbose and first_batch:
                    print(f"  [debug] generated cols in PG: {sorted(gen_cols)}")
                    print(f"  [debug] inserting keys: {sorted(rows[0].keys())}")
                    first_batch = False

                p_conn.execute(pg_table.insert(), rows)
                total += len(rows)
    return total


def build_order(s_eng: Engine, base: List[str], skip: Set[str]) -> List[str]:
    """
    Take DEFAULT_LOAD_ORDER, keep what exists in SQLite and not skipped, then
    append leftovers.
    """
    existing = set(list_sqlite_tables(s_eng))
    order = [t for t in base if t in existing and t not in skip]
    leftovers = [t for t in sorted(existing) if t not in set(order) | skip]
    return order + leftovers


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    sqlite_url, pg_url = get_urls()
    s_eng, p_eng = make_engines(sqlite_url, pg_url)

    # Resolve SKIP/ONLY lists
    skip = {t.strip() for t in args.skip.split(",")} if args.skip else set()
    only = [t.strip() for t in args.only.split(",") if t.strip()] if args.only else []

    # Decide load order
    base_order = only if only else DEFAULT_LOAD_ORDER
    order = build_order(s_eng, base_order, skip)

    print("SQLite tables:", list_sqlite_tables(s_eng))
    print("Load order:", order)
    if not order:
        print("Nothing to load. Exiting.")
        return

    summary: List[Tuple[str, int, int, str]] = []

    for t in order:
        print(f"\n→ {t}")

        # IMPORTANT: Do NOT auto-create fixture_predictions from SQLite.
        if t != "fixture_predictions":
            create_pg_table_from_sqlite(s_eng, p_eng, t)

        # Skip or truncate logic
        if args.truncate:
            print("  truncating…")
            truncate_table(p_eng, t)
        else:
            # skip if PG already has rows (avoid duplicates)
            try:
                existing = count_rows(p_eng, t)
                if existing > 0:
                    print(f"  Postgres already has {existing} rows; skipping (use --truncate to overwrite).")
                    summary.append((t, existing, existing, "skipped"))
                    continue
            except Exception:
                pass  # table might be new

        # Copy rows
        try:
            inserted = copy_table(s_eng, p_eng, t, args.chunk, verbose=args.verbose)
            # Optional: bump identity on simple integer PK tables
            try:
                advance_identity(p_eng, t)
            except Exception:
                pass
            s_count = count_rows(s_eng, t)
            p_count = count_rows(p_eng, t)
            print(f"  inserted={inserted}  sqlite={s_count}  postgres={p_count}")
            summary.append((t, s_count, p_count, "copied"))
        except SQLAlchemyError as e:
            print(f"  ERROR copying {t}: {e}")
            summary.append((t, -1, -1, f"error: {e}"))

    print("\nSummary:")
    for t, s_c, p_c, status in summary:
        print(f"{t:20s} sqlite={s_c:7d}  postgres={p_c:7d}  {status}")

    if "fixture_predictions" in skip or "fixture_predictions" not in order:
        print("\nNote: 'fixture_predictions' was not auto-created from SQLite.")
        print("      Create it in Postgres with valid DDL (plain columns OR generated expressions),")
        print("      then re-run this script with --only fixture_predictions to load it (dropping pred/actual labels if generated).")


def advance_identity(p_eng: Engine, table_name: str):
    """
    Try to advance identity/serial to max(id)+1 for single-column PKs.
    Safe to skip on failure.
    """
    try:
        with p_eng.begin() as conn:
            q_pk = text("""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = :tbl::regclass
                AND    i.indisprimary
            """)
            pk_rows = conn.execute(q_pk, {"tbl": table_name}).fetchall()
            if len(pk_rows) != 1:
                return
            pk_col = pk_rows[0][0]
            next_val = conn.execute(
                text(f'SELECT COALESCE(MAX("{pk_col}"),0)+1 FROM "{table_name}"')
            ).scalar_one()
            try:
                conn.execute(
                    text(f'ALTER TABLE "{table_name}" ALTER COLUMN "{pk_col}" RESTART WITH :n'),
                    {"n": int(next_val)},
                )
                return
            except SQLAlchemyError:
                seq = conn.execute(
                    text("SELECT pg_get_serial_sequence(:tbl,:col)"),
                    {"tbl": table_name, "col": pk_col},
                ).scalar()
                if seq:
                    conn.execute(
                        text("SELECT setval(:seq, :n, false)"),
                        {"seq": seq, "n": int(next_val)},
                    )
    except SQLAlchemyError:
        return


if __name__ == "__main__":
    main()
