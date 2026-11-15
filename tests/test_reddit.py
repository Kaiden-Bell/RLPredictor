from datetime import date, datetime, timezone

import pytest

from rlpredictor.data import db, reddit
from rlpredictor.data.ingest import store_squad
from rlpredictor.data.wikitext import SquadMember

pytest.importorskip("vaderSentiment")


def ts(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "r.db")
    gm = db.get_or_create_team(c, "gentle mates", "Gentle Mates", "Europe")
    sr = db.get_or_create_team(c, "shopify rebellion", "Shopify Rebellion", "North America")
    store_squad(c, gm, [SquadMember(n, n, "", date(2026, 10, 28)) for n in ("Atow", "Vatira", "zen")])
    store_squad(c, sr, [SquadMember(n, n, "", date(2026, 10, 28))
                        for n in ("Firstkiller", "Atomic", "Daniel")])
    c.commit()
    return c, gm, sr


class FakeReddit:
    def __init__(self, posts, comments):
        self.posts, self._comments, self.queries = posts, comments, []

    def search(self, q, t="week", limit=100):
        self.queries.append(q)
        return self.posts

    def comments(self, pid, limit=100):
        return self._comments.get(pid, [])


def test_scoring_and_terms(conn):
    c, gm, _ = conn
    assert reddit.compound("Vatira is cracked") > 0.2       # esports slang override
    assert reddit.compound("they are washed and choked") < -0.2
    terms = reddit.team_terms(c, gm)
    assert terms == ["Gentle Mates", "Atow", "Vatira"]      # "zen" too short to match safely
    assert reddit.matched_terms("atow and VATIRA popped off", terms) == ["Atow", "Vatira"]
    assert reddit.matched_terms("Atowski is someone else", terms) == []


def test_collect_and_point_in_time(conn):
    c, gm, sr = conn
    posts = [
        {"id": "a", "title": "Gentle Mates look cracked with Vatira", "selftext": "",
         "created_utc": ts(2026, 11, 10), "num_comments": 50, "permalink": "/a"},
        {"id": "b", "title": "Shopify Rebellion washed already?", "selftext": "",
         "created_utc": ts(2026, 11, 12), "num_comments": 5, "permalink": "/b"},
        {"id": "c", "title": "Unrelated post about the meta", "selftext": "",
         "created_utc": ts(2026, 11, 12), "num_comments": 0, "permalink": "/c"},
    ]
    comments = {"a": [{"id": "x", "body": "Atomic is goated", "created_utc": ts(2026, 11, 13)},
                      {"id": "y", "body": "nice", "created_utc": ts(2026, 11, 13)}]}
    fake = FakeReddit(posts, comments)
    rep = reddit.collect(c, fake, [gm, sr])
    assert '"Gentle Mates" OR "Atow" OR "Vatira"' == fake.queries[0]
    assert rep.stored == 3 and rep.comments_fetched == 1     # unrelated post + "nice" skipped
    # idempotent
    assert reddit.collect(c, fake, [gm, sr]).stored == 0

    s = reddit.sentiment_for(c, gm, date(2026, 11, 14))
    assert s["n"] == 1 and s["mean"] > 0
    sr_s = reddit.sentiment_for(c, sr, date(2026, 11, 14))
    assert sr_s["n"] == 2                                     # the post + the Atomic comment
    # Point-in-time: text written on/after the match day is never used.
    assert reddit.sentiment_for(c, gm, date(2026, 11, 10)) is None
    assert reddit.sentiment_for(c, sr, date(2026, 11, 13))["n"] == 1
    assert reddit.sentiment_for(c, gm, date(2026, 1, 1)) is None
    ev = reddit.evidence(c, (gm, "Gentle Mates"), (sr, "Shopify Rebellion"), date(2026, 11, 14))
    assert "Gentle Mates +" in ev and "over 1" in ev


def test_credentials_required(monkeypatch):
    from rlpredictor import config

    for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(config.ConfigError, match="prefs/apps"):
        reddit.Reddit(None)
