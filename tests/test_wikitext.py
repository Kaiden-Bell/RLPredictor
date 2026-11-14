from datetime import date
from pathlib import Path

from rlpredictor.data.wikitext import (
    parse_matches,
    parse_page_info,
    parse_rosters,
    region_from_title,
    team_key,
)

FIX = Path(__file__).parent / "fixtures"


def test_swiss_matches():
    text = (FIX / "swiss_group_a.wiki").read_text()
    info = parse_page_info("x", text)
    assert info.tier == 2 and info.page_date == date(2025, 4, 27) and not info.has_infobox
    ms = parse_matches(text, info.page_date)
    assert len(ms) == 33
    assert len({m.match_key for m in ms}) == len(ms)
    m = ms[0]
    assert (m.team1, m.score1, m.score2, m.team2) == ("Karmine Corp", 1, 3, "Ninjas in Pyjamas")
    assert m.best_of == 5 and m.finished and m.date == date(2025, 4, 27)
    assert m.match_key == "cUgVCfgq1X:M1"
    assert [g.winner for g in m.games] == [2, 1, 2, 2]
    assert m.games[2].ot is True
    assert m.section == "Round 1"


def test_major_rosters_and_bracket():
    text = (FIX / "raleigh_major.wiki").read_text()
    info = parse_page_info("x", text)
    assert info.has_infobox and info.tier == 1
    rosters = {r.team_name: r for r in parse_rosters(text)}
    assert rosters["Team Vitality"].players == ["zen", "M0nkey M00n", "ExoTiiK"]
    # staff (coach) excluded, link used for disambiguated players
    assert rosters["Gentle Mates Alpine"].players == ["Seikoo", "Juicy (French Player)", "yujin"]
    ms = parse_matches(text)
    first = ms[0]
    assert (first.team1, first.score1, first.score2, first.team2) == ("Karmine Corp", 3, 0, "Twisted Minds")
    assert first.ballchasing == "kc-vs-twis-hxxdbc25jy"
    final = [m for m in ms if m.match_key == "Pke14cOtmZ:R4M1"][0]
    assert (final.team1, final.score1, final.score2, final.team2) == ("Dignitas", 1, 4, "Team Falcons")
    assert final.best_of == 7


def test_walkover_tbd_and_upcoming():
    text = """
{{Matchlist|id=abc
|M1={{Match
  |opponent1={{TeamOpponent|alpha|score=W}}
  |opponent2={{TeamOpponent|beta|score=FF}}
  |finished=true
}}
|M2={{Match
  |opponent1={{TeamOpponent|tbd}}
  |opponent2={{TeamOpponent|beta}}
}}
|M3={{Match
  |opponent1={{TeamOpponent|alpha}}
  |opponent2={{TeamOpponent|gamma}}
  |bestof=7
  |date=March 3, 2026 - 18:00 {{Abbr/CET}}
}}
}}"""
    ms = {m.match_key: m for m in parse_matches(text)}
    assert ms["abc:M1"].walkover
    assert "abc:M2" not in ms
    up = ms["abc:M3"]
    assert not up.finished and up.best_of == 7 and up.date == date(2026, 3, 3)


def test_helpers():
    assert team_key("  Team_Vitality ") == "team vitality"
    assert region_from_title("RLCS/2025/Raleigh Major/Europe/Open 4") == "Europe"
    assert region_from_title("RLCS/2025/Raleigh Major") == "International"
