from pathlib import Path

from rlpredictor.data import db
from rlpredictor.data.ballchasing import store_group
from rlpredictor.data.identity import match_player, normalize_player_name
from rlpredictor.data.ingest import IngestReport, parse_page, store_pages

FIX = Path(__file__).parent / "fixtures"
MAJOR = "Rocket League Championship Series/2025/Raleigh Major"


def _ingest_fixture(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    pages = [parse_page(MAJOR, 1, (FIX / "raleigh_major.wiki").read_text())]
    rep = IngestReport()
    store_pages(conn, pages, lp=None, report=rep)
    return conn, rep


def test_store_pages(tmp_path):
    conn, rep = _ingest_fixture(tmp_path)
    n = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    assert n == 42  # 43 matches minus the casters' showmatch
    row = conn.execute(
        """SELECT t1.name, s.score1, s.score2, t2.name, s.tier, s.region, s.bc_group FROM series s
           JOIN teams t1 ON t1.id=s.team1_id JOIN teams t2 ON t2.id=s.team2_id
           WHERE s.bc_group='kc-vs-twis-hxxdbc25jy'""").fetchone()
    assert tuple(row) == ("Karmine Corp", 3, 0, "Twisted Minds", 1, "International",
                          "kc-vs-twis-hxxdbc25jy")
    vit = db.find_team(conn, "team vitality")[0]
    players = [r[0] for r in conn.execute(
        "SELECT player_id FROM event_rosters WHERE team_id=? ORDER BY slot", (vit["id"],))]
    assert players == ["zen", "m0nkey m00n", "exotiik"]
    # idempotent
    store_pages(conn, [parse_page(MAJOR, 2, (FIX / "raleigh_major.wiki").read_text())], None, rep)
    assert conn.execute("SELECT COUNT(*) FROM series").fetchone()[0] == n


def test_ballchasing_matching(tmp_path):
    conn, _ = _ingest_fixture(tmp_path)
    assert normalize_player_name("KC Vatira") == "vatira"
    cands = {"zen": "zen", "m0nkey m00n": "M0nkey M00n", "exotiik": "ExoTiiK"}
    assert match_player(conn, None, "M0NKEY M00N", cands) == "m0nkey m00n"
    assert match_player(conn, None, "Exotiik.", cands) == "exotiik"
    assert match_player(conn, None, "someoneelse", cands) is None

    sid = conn.execute("SELECT id FROM series WHERE bc_group='kc-vs-twis-hxxdbc25jy'").fetchone()[0]
    kc = db.find_team(conn, "karmine corp")[0]["id"]
    group = {"players": [
        {"platform": "steam", "id": "1", "name": "Vatira",
         "cumulative": {"games": 3, "core": {"goals": 4, "assists": 2, "saves": 5, "shots": 9, "score": 1500},
                        "demo": {"inflicted": 3}}},
        {"platform": "epic", "id": "2", "name": "randomsub", "cumulative": {"games": 3, "core": {}}},
    ]}
    m, u = store_group(conn, sid, group, {kc: {"vatira": "Vatira", "atow.": "Atow.", "dralii": "dralii"}})
    assert (m, u) == (1, 1)
    row = conn.execute("SELECT player_id, games, goals, score FROM player_series_stats").fetchone()
    assert tuple(row) == ("vatira", 3, 4, 1500)
    # platform id remembered for next time
    assert conn.execute("SELECT player_id FROM player_platform_ids WHERE platform_id='steam:1'").fetchone()[0] == "vatira"
