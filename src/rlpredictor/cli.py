"""`rlpredict` command-line interface."""

from __future__ import annotations
from datetime import date
from typing import Optional
from rlpredictor import config

import json
import logging
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Explainable Rocket League esports match predictions.")

def _conn():
    from rlpredictor.data import db
    return db.connect()

def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)

@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

@app.command()
def ingest(
    seasons: list[str] = typer.Option(["2024", "2025", "2026"], "--season", "-s",
                                      help="RLCS seasons to ingest (repeatable)."),
    force: bool = typer.Option(False, help="Re-fetch every page, not just changed ones."),
    bc_budget: int = typer.Option(0, help="Also fetch up to N Ballchasing groups (needs API key)."),
):
    """Pull RLCS results and rosters from the Liquipedia API (incremental)."""
    from rlpredictor.data import ingest as ing
    from rlpredictor.data.liquipedia import Liquipedia

    try:
        conn = _conn()
        rep = ing.ingest(conn, Liquipedia(conn), seasons, force=force, progress=typer.echo)
    except config.ConfigError as e:
        _fail(str(e))
    n_unres = len({r for _, r in rep.unresolved})
    typer.echo(f"Pages listed {rep.pages_listed}, parsed {rep.pages_fetched}, "
               f"series upserted {rep.series_upserted}, rosters {rep.rosters}, "
               f"opponents not in event rosters {n_unres}")
    if bc_budget:
        enrich(budget=bc_budget, team=[])
    _status(conn)

@app.command()
def enrich(
    budget: int = typer.Option(400, help="Max Ballchasing API calls this run."),
    team: list[str] = typer.Option([], help="Prioritise series involving these teams."),
):
    """Fetch per-player series stats from Ballchasing (player-value factor)."""
    from rlpredictor.data import db
    from rlpredictor.data.ballchasing import Ballchasing, enrich as run

    conn = _conn()
    try:
        bc = Ballchasing(conn)
    except config.ConfigError as e:
        _fail(str(e))
    ids = [r["id"] for t in team for r in db.find_team(conn, t)[:1]]
    rep = run(conn, bc, budget=budget, team_ids=ids or None, progress=typer.echo)
    typer.echo(f"Ballchasing: attempted {rep.attempted}, ok {rep.ok}, missing {rep.missing}, "
               f"errors {rep.errors}, unmatched players {rep.unmatched_players}")

def _status(conn) -> None:
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    typer.echo(
        f"DB: {q('SELECT COUNT(*) FROM series WHERE finished=1')} finished series "
        f"({q('SELECT MIN(date) FROM series')} → {q('SELECT MAX(date) FROM series WHERE finished=1')}), "
        f"{q('SELECT COUNT(*) FROM series WHERE finished=0')} upcoming, "
        f"{q('SELECT COUNT(*) FROM teams')} teams, {q('SELECT COUNT(*) FROM players')} players, "
        f"{q('SELECT COUNT(DISTINCT series_id) FROM player_series_stats')} series with Ballchasing stats")

@app.command()
def status():
    """Show what's in the local database and model."""
    conn = _conn()
    _status(conn)
    if config.MODEL_PATH.exists():
        from rlpredictor.model.train import GameModel
        m = GameModel.load()
        typer.echo(f"Model: trained through {m.trained_through} on {m.n_games} games; "
                   f"factors: {', '.join(m.metrics.get('kept_groups', []))}")
    else:
        typer.echo("Model: not trained yet (run `rlpredict train`).")

@app.command()
def backtest():
    """Walk-forward evaluation vs coin flip and Elo-only, with factor ablations."""
    from rlpredictor.model import backtest as bt
    from rlpredictor.model.features import build

    ds, _ = build(_conn())
    typer.echo(f"{len(ds.series)} series / {len(ds.y)} games")
    res, keep = bt.run(ds)
    typer.echo(bt.HEADER)
    for r in res.values():
        typer.echo(r.row())
    typer.echo(f"\nFactors that improve log-loss (kept): {', '.join(keep)}")

@app.command()
def train(select: bool = typer.Option(True, help="Drop factors that don't help in backtest.")):
    """Train the model on all finished series and save it."""
    from rlpredictor.model import backtest as bt
    from rlpredictor.model.features import FACTOR_GROUPS, FEATURES, build
    from rlpredictor.model.train import fit

    ds, hist = build(_conn())
    if len(ds.y) < 500:
        _fail(f"Only {len(ds.y)} games in the database; run `rlpredict ingest` first.")
    metrics: dict = {}
    keep = list(FACTOR_GROUPS)
    if select:
        typer.echo("Running walk-forward backtest to select factors...")
        res, keep = bt.run(ds)
        typer.echo(bt.HEADER)
        for r in res.values():
            typer.echo(r.row())
        metrics["backtest"] = {k: vars(v) for k, v in res.items()}
    metrics["kept_groups"] = keep
    active = [f for g in keep for f in FACTOR_GROUPS[g]]
    model = fit(ds.X, ds.y, [f for f in FEATURES if f in active])
    model.trained_through = str(hist.last_date)
    model.metrics = metrics
    path = model.save()
    typer.echo(f"\nSaved model to {path} (factors: {', '.join(keep)})")
    for f, w in zip(FEATURES, model.weights):
        if w:
            typer.echo(f"  {f:<16} {w:+.4f}")

def _bar(p: float, width: int = 30) -> str:
    n = round(p * width)
    return "█" * n + "░" * (width - n)

def render(pred) -> str:
    d = pred
    lines = []
    a, b = d.team1, d.team2
    lines.append(f"\n{a.name}  vs  {b.name}   (Bo{d.best_of}, as of {d.as_of})")
    for side in (a, b):
        roster = ", ".join(side.roster) if side.roster else "roster not listed"
        src = f"  [{side.roster_source}]" if side.roster_source else ""
        lines.append(f"  {side.name}: {roster}{src}")
    lines.append("")
    hi, lo = max(d.predicted_score), min(d.predicted_score)
    lines.append(f"Prediction: {d.winner} wins  ({d.winner_prob:.0%})"
                 f"   most likely {hi}-{lo}"
                 + ("   ⚠ upset watch" if d.upset else ""))
    lines.append(f"  {a.name:<24} {_bar(d.p_series)} {d.p_series:.0%}")
    lines.append(f"  {b.name:<24} {_bar(1 - d.p_series)} {1 - d.p_series:.0%}")
    lines.append(f"  per-game: {a.name} {d.p_game:.0%}")
    lines.append("")
    lines.append("Series length: " + "  ".join(f"{k} {v:.0%}" for k, v in d.length.items()))
    top = list(d.score_distribution.items())[:4]
    lines.append("Top scores:    " + "  ".join(f"{k} {v:.0%}" for k, v in top))
    lines.append("")
    lines.append("Why:")
    for f in d.factors:
        if not f.in_model:
            continue
        who = f"+{f.contribution_pp:.1f} pts → {f.favors}" if f.favors else "neutral"
        lines.append(f"  • {f.name:<17} {who}")
        lines.append(f"      {f.evidence}")
    ctx = [f for f in d.factors if not f.in_model]
    if ctx:
        lines.append("Context (not weighted by the model):")
        for f in ctx:
            lines.append(f"  • {f.name}: {f.evidence}")
    for n in d.notes:
        lines.append(f"Note: {n}")
    lines.append(f"\n{d.attribution}")
    return "\n".join(lines)

def _predictor():
    from rlpredictor.model.train import GameModel
    from rlpredictor.predictor import Predictor

    if not config.MODEL_PATH.exists():
        _fail("No trained model. Run `rlpredict ingest` then `rlpredict train`.")
    return Predictor(_conn(), GameModel.load())

@app.command()
def predict(
    team1: str,
    team2: str,
    bo: int = typer.Option(5, "--bo", help="Best-of (1, 3, 5, 7)."),
    on: Optional[str] = typer.Option(None, help="Predict as of date YYYY-MM-DD (default today)."),
    roster1: Optional[str] = typer.Option(None, help="Override team1 lineup: 'a,b,c'."),
    roster2: Optional[str] = typer.Option(None, help="Override team2 lineup: 'a,b,c'."),
    as_json: bool = typer.Option(False, "--json"),
):
    """Who wins TEAM1 vs TEAM2, and why."""
    from rlpredictor.predictor import PredictError

    p = _predictor()
    try:
        pred = p.predict(team1, team2, best_of=bo,
                         on=date.fromisoformat(on) if on else None,
                         roster1=roster1.split(",") if roster1 else None,
                         roster2=roster2.split(",") if roster2 else None)
    except PredictError as e:
        _fail(str(e))
    typer.echo(json.dumps(pred.to_dict(), indent=2, default=str) if as_json else render(pred))

@app.command()
def event(page: str, as_json: bool = typer.Option(False, "--json")):
    """Predict all upcoming matches on a Liquipedia event page (title or URL)."""
    p = _predictor()
    rows = p.upcoming(page)
    if as_json:
        typer.echo(json.dumps(rows, indent=2, default=str))
        return
    if not rows:
        typer.echo("No upcoming matches found for that page (run `rlpredict ingest` to refresh).")
        return
    for r in rows:
        if "error" in r:
            typer.echo(f"{r['date']}  {r['error']}")
            continue
        d = r["prediction"]
        top = next((f for f in d["factors"] if f["in_model"] and f["favors"]), None)
        why = f" — {top['name'].lower()}" if top else ""
        typer.echo(f"{r['date'] or '?':<11} {d['team1']['name']:>24} vs {d['team2']['name']:<24} "
                   f"→ {d['winner']} {d['winner_prob']:.0%} "
                   f"({max(d['predicted_score'])}-{min(d['predicted_score'])}){why}"
                   + ("  ⚠" if d["upset"] else ""))

@app.command()
def teams(query: str):
    """Search teams in the local database."""
    from rlpredictor.data import db

    rows = db.find_team(_conn(), query)
    if not rows:
        typer.echo("No matches.")
    for r in rows:
        extra = f"  last played {r['last_played']}" if "last_played" in r.keys() and r["last_played"] else ""
        typer.echo(f"{r['name']}  ({r['region'] or '?'}){extra}")

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the JSON API."""
    import uvicorn
    uvicorn.run("rlpredictor.api:app", host=host, port=port)

if __name__ == "__main__":
    app()
