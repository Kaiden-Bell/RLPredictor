"""Prediction service shared by the CLI and the API."""

from __future__ import annotations
from rlpredictor import config
from rlpredictor.data import db, reddit
from rlpredictor.data.wikitext import player_key
from rlpredictor.model import features as F
from rlpredictor.model.confidence import Confidence, assess
from rlpredictor.model.explain import Factor, explain
from rlpredictor.model.series import length_buckets, score_distribution, series_win_prob
from rlpredictor.model.train import GameModel
from dataclasses import dataclass, field
from datetime import date

import sqlite3



class PredictError(ValueError):
    def __init__(self, msg: str, candidates: list[str] | None = None):
        super().__init__(msg)
        self.candidates = candidates or []


@dataclass
class TeamSide:
    id: int
    name: str
    roster: list[str]          # display names
    roster_ids: tuple[str, ...] | None
    roster_source: str | None  # event the roster was listed at, or "override"


@dataclass
class Prediction:
    team1: TeamSide
    team2: TeamSide
    best_of: int
    p_game: float              # team1 per-game win probability
    p_series: float            # team1 series win probability
    winner: str
    winner_prob: float
    predicted_score: tuple[int, int]
    score_distribution: dict[str, float]
    length: dict[str, float]
    factors: list[Factor]
    upset: bool
    as_of: str
    confidence: Confidence | None = None
    attribution: str = config.ATTRIBUTION
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "team1": vars(self.team1) | {"roster_ids": list(self.team1.roster_ids or [])},
            "team2": vars(self.team2) | {"roster_ids": list(self.team2.roster_ids or [])},
            "best_of": self.best_of,
            "p_game": round(self.p_game, 4),
            "p_series": round(self.p_series, 4),
            "winner": self.winner,
            "winner_prob": round(self.winner_prob, 4),
            "predicted_score": list(self.predicted_score),
            "score_distribution": {k: round(v, 4) for k, v in self.score_distribution.items()},
            "length": {k: round(v, 4) for k, v in self.length.items()},
            "factors": [f.to_dict() for f in self.factors],
            "upset": self.upset,
            "as_of": self.as_of,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "attribution": self.attribution,
            "notes": self.notes,
        }


class Predictor:
    """Builds the point-in-time history once and answers many predictions."""

    def __init__(self, conn: sqlite3.Connection, model: GameModel | None = None):
        self.conn = conn
        self.model = model or GameModel.load()
        _, self.history = F.build(conn)
        self.player_names = {r[0]: r[1] for r in conn.execute("SELECT id, name FROM players")}

    def resolve_team(self, query: str | int) -> sqlite3.Row:
        if isinstance(query, int):
            row = self.conn.execute("SELECT * FROM teams WHERE id = ?", (query,)).fetchone()
            if not row:
                raise PredictError(f"No team with id {query}")
            return row
        rows = db.find_team(self.conn, query)
        if not rows:
            raise PredictError(f"No team matching '{query}'. Try `rlpredict teams {query}`.")
        if len(rows) > 1:
            exact = [r for r in rows if r["name"].casefold() == query.casefold()]
            if len(exact) == 1:
                return exact[0]
            if "last_played" in rows[0].keys():
                return rows[0] 
        return rows[0]

    def event_roster(self, team_id: int, on: date) -> tuple[tuple[str, ...] | None, str | None, date | None]:
        """Roster from the most recent event (with a listed roster) that started by `on`."""
        row = self.conn.execute(
            """SELECT er.event_root, p.start_date FROM event_rosters er
               JOIN pages p ON p.title = er.event_root
               WHERE er.team_id = ? AND (p.start_date IS NULL OR p.start_date <= ?)
               GROUP BY er.event_root
               ORDER BY p.start_date IS NULL, p.start_date DESC LIMIT 1""",
            (team_id, str(on))).fetchone()
        if not row:
            return None, None, None
        players = self.conn.execute(
            "SELECT player_id FROM event_rosters WHERE event_root = ? AND team_id = ? ORDER BY slot",
            (row[0], team_id)).fetchall()
        start = date.fromisoformat(row[1]) if row[1] else None
        return tuple(p[0] for p in players), row[0], start

    def squad(self, team_id: int) -> tuple[tuple[str, ...], date | None]:
        """Active squad from the team page, and the date it took effect (latest starter
        join date, or when it was fetched if Liquipedia lists no join dates)."""
        rows = self.conn.execute(
            "SELECT player_id, slot, join_date FROM team_squads WHERE team_id = ? ORDER BY slot",
            (team_id,)).fetchall()
        if not rows:
            return (), None
        joins = [r[2] for r in rows if r[1] <= 3 and r[2]]
        if joins:
            since = date.fromisoformat(max(joins))
        else:
            f = self.conn.execute("SELECT fetched_at FROM team_squad_fetch WHERE team_id = ?",
                                  (team_id,)).fetchone()
            since = date.fromtimestamp(f[0]) if f else None
        return tuple(r[0] for r in rows), since

    def latest_roster(self, team_id: int, on: date | None = None
                      ) -> tuple[tuple[str, ...] | None, str | None, str | None]:
        """(player ids, source label, roster-change note) for a team as of `on`.

        The team-page squad wins when it took effect after the latest event roster (and by
        `on`); otherwise the event roster is the better point-in-time record."""
        on = on or date.today()
        ev_ids, ev_root, ev_start = self.event_roster(team_id, on)
        sq_ids, sq_since = self.squad(team_id)
        use_squad = (len(sq_ids) >= 3 and sq_since is not None and sq_since <= on
                     and (ev_ids is None or ev_start is None or sq_since > ev_start))
        if not use_squad:
            return ev_ids, ev_root, None
        note = None
        if ev_ids and set(ev_ids[:3]) != set(sq_ids[:3]):
            name = lambda p: self.player_names.get(p, p)  # noqa: E731
            added = [name(p) for p in sq_ids[:3] if p not in ev_ids[:3]]
            dropped = [name(p) for p in ev_ids[:3] if p not in sq_ids[:3]]
            note = (f"roster changed since {ev_root}: +{', '.join(added)}; "
                    f"−{', '.join(dropped)} (Liquipedia team page)")
        return sq_ids, f"Liquipedia team page, since {sq_since}", note

    def _side(self, query: str | int, override: list[str] | None,
              on: date | None = None) -> tuple[TeamSide, str | None]:
        t = self.resolve_team(query)
        note = None
        if override:
            ids = tuple(player_key(n) for n in override)
            src = "override"
        else:
            ids, src, note = self.latest_roster(t["id"], on)
        names = [self.player_names.get(p, p) for p in (ids or ())[:3]]
        return TeamSide(t["id"], t["name"], names, ids, src), note

    def predict(self, team1: str | int, team2: str | int, best_of: int = 5, on: date | None = None,
                roster1: list[str] | None = None, roster2: list[str] | None = None) -> Prediction:

        if best_of < 1 or best_of % 2 == 0:
            raise PredictError("best_of must be an odd number (1, 3, 5, 7)")
        on = on or date.today()
        (a, note_a), (b, note_b) = self._side(team1, roster1, on), self._side(team2, roster2, on)
        if a.id == b.id:
            raise PredictError(f"Both names resolve to the same team ({a.name}).")

        x, ev = self.history.features(a.id, b.id, a.roster_ids, b.roster_ids, on)
        pname = lambda p: self.player_names.get(p, p)  # noqa: E731
        u, conf = assess(ev, self.model.shrink, self.model.metrics.get("confidence"),
                         a.name, b.name, pname)
        p_game = float(self.model.p_game(x, u))
        dist = score_distribution(p_game, best_of)
        p_series = series_win_prob(p_game, best_of)
        top_score = max(dist, key=dist.get)
        winner, wp = (a.name, p_series) if p_series >= 0.5 else (b.name, 1 - p_series)
        r1, r2 = ev["rating"]
        favored_by_rating = a.name if r1 >= r2 else b.name
        upset = winner != favored_by_rating or wp < 0.60

        notes = []
        for side, change in ((a, note_a), (b, note_b)):
            if change:
                notes.append(f"{side.name} {change}.")
            if not side.roster_ids:
                notes.append(f"No listed roster for {side.name}; using team-level rating only.")
        if self.history.last_date and (on - self.history.last_date).days > 45:
            notes.append(f"Latest result in the database is {self.history.last_date}; "
                         "run `rlpredict ingest` to refresh.")

        factors = explain(self.model, x, ev, best_of, a.name, b.name, pname=pname, u=u)
        if reddit.has_data(self.conn):
            factors.append(Factor("Community sentiment", 0.0, None,
                                  reddit.evidence(self.conn, (a.id, a.name), (b.id, b.name), on),
                                  in_model=False))

        return Prediction(
            team1=a, team2=b, best_of=best_of, p_game=p_game, p_series=p_series,
            winner=winner, winner_prob=wp, predicted_score=top_score,
            score_distribution={f"{x1}-{x2}": v for (x1, x2), v in
                                sorted(dist.items(), key=lambda kv: -kv[1])},
            length=length_buckets(dist, best_of), factors=factors, upset=upset,
            as_of=str(on), confidence=conf, notes=notes,
        )

    def upcoming(self, page_query: str) -> list[dict]:
        """Unfinished series on an event page (or any of its subpages)."""
        title = page_query.split("liquipedia.net/rocketleague/")[-1].replace("_", " ").strip("/")
        rows = self.conn.execute(
            """SELECT s.*, t1.name AS n1, t2.name AS n2 FROM series s
               JOIN teams t1 ON t1.id = s.team1_id JOIN teams t2 ON t2.id = s.team2_id
               WHERE s.finished = 0 AND (s.page = ? OR s.page LIKE ? OR s.event_root = ?)
               ORDER BY s.date, s.id""", (title, title + "/%", title)).fetchall()
        out = []
        for r in rows:
            bo = r["best_of"] or 5
            try:
                p = self.predict(r["team1_id"], r["team2_id"], best_of=bo,
                                 on=date.fromisoformat(r["date"]) if r["date"] else None)
                out.append({"series_id": r["id"], "date": r["date"], "section": r["section"],
                            "prediction": p.to_dict()})
            except PredictError as e:
                out.append({"series_id": r["id"], "date": r["date"], "error": str(e)})
                
        return out
