"""Parsers for Liquipedia Rocket League wikitext (match2 templates).

Pure functions: wikitext in, plain dataclasses out. No network or DB access, so they
are easy to test against saved fixtures.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

import mwparserfromhell as mwp

REGIONS = [
    "Europe",
    "North America",
    "South America",
    "Middle East and North Africa",
    "Asia-Pacific",
    "Oceania",
    "Sub-Saharan Africa",
    "Asia-Pacific North",
    "Asia-Pacific South",
]
TBD_NAMES = {"", "tbd", "tba", "bye", "&nbsp;", "definitions"}


def team_key(name: str) -> str:
    """Case/whitespace-insensitive key for a team template or TeamCard name."""
    n = unicodedata.normalize("NFKC", name or "").replace("​", "")
    n = n.replace("_", " ")
    return " ".join(n.split()).casefold()


def player_key(name: str) -> str:
    n = unicodedata.normalize("NFKC", name or "").replace("_", " ")
    return " ".join(n.split()).casefold()


@dataclass
class PageInfo:
    title: str
    has_infobox: bool = False
    name: str | None = None
    tier: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    page_date: date | None = None
    mode: str | None = None


@dataclass
class RosterEntry:
    team_name: str  # display name (TeamCard `team`)
    team_link: str  # template/link name used for matching opponents
    players: list[str]  # starters (p1..p3), Liquipedia names as written
    subs: list[str] = field(default_factory=list)


@dataclass
class GameResult:
    n: int
    score1: int | None
    score2: int | None
    winner: int | None  # 1 or 2
    ot: bool = False


@dataclass
class SeriesResult:
    match_key: str  # unique within the page
    team1: str
    team2: str
    score1: int | None
    score2: int | None
    finished: bool
    best_of: int | None
    date: date | None
    games: list[GameResult]
    ballchasing: str | None
    section: str | None
    walkover: bool = False

    @property
    def upcoming(self) -> bool:
        return not self.finished


# ---------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_DATE_RE = re.compile(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})")
_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    text = str(text)
    m = _ISO_RE.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        if mo == 0 or d == 0:
            return None
        try:
            return date(y, mo, d)
        except ValueError:
            return None
    m = _DATE_RE.search(text)
    if m and m.group(1).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    return None


def _val(tpl, name: str) -> str | None:
    if tpl.has(name):
        v = str(tpl.get(name).value).strip()
        # drop html comments and refs
        v = re.sub(r"<!--.*?-->", "", v, flags=re.S)
        v = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", v, flags=re.S).strip()
        return v or None
    return None


def _int(v: str | None) -> int | None:
    if v is None:
        return None
    v = v.strip()
    return int(v) if re.fullmatch(r"-?\d+", v) else None


def _tname(tpl) -> str:
    return str(tpl.name).strip().lower()


# ---------------------------------------------------------------------------
def parse_page_info(title: str, text: str) -> PageInfo:
    code = mwp.parse(text)
    info = PageInfo(title=title)
    for tpl in code.filter_templates(recursive=False):
        n = _tname(tpl)
        if n in ("infobox league", "hiddendatabox"):
            info.has_infobox = info.has_infobox or n == "infobox league"
            info.name = info.name or _val(tpl, "name")
            info.tier = info.tier or _int(_val(tpl, "liquipediatier"))
            info.start_date = info.start_date or parse_date(_val(tpl, "sdate"))
            info.end_date = info.end_date or parse_date(_val(tpl, "edate"))
            info.page_date = info.page_date or parse_date(_val(tpl, "date"))
            info.mode = info.mode or _val(tpl, "mode")
    return info


def region_from_title(title: str) -> str:
    parts = title.split("/")
    for p in parts:
        if p in REGIONS:
            return p
    return "International"


def parse_rosters(text: str) -> list[RosterEntry]:
    out: list[RosterEntry] = []
    for tpl in mwp.parse(text).filter_templates(recursive=True):
        if _tname(tpl) != "teamcard":
            continue
        team = _val(tpl, "team")
        if not team:
            continue
        link = _val(tpl, "link") or team
        players: list[tuple[int, str]] = []
        subs: list[str] = []
        for p in tpl.params:
            k = str(p.name).strip()
            m = re.fullmatch(r"([ps])(\d+)", k)
            if not m:
                continue
            raw = _val(tpl, k)
            if not raw:
                continue
            name = _val(tpl, f"{k}link") or raw
            name = re.sub(r"\[\[([^|\]]+)(?:\|[^\]]*)?\]\]", r"\1", name).strip()
            if m.group(1) == "p":
                players.append((int(m.group(2)), name))
            else:
                subs.append(name)
        players.sort()
        starters = [n for _, n in players[:3]]
        subs = [n for _, n in players[3:]] + subs
        out.append(RosterEntry(team_name=team, team_link=link, players=starters, subs=subs))

    # Newer format: {{TeamParticipants |{{Opponent|Team|players={{Persons|{{Person|x}}...}}}}}}
    for tp in mwp.parse(text).filter_templates(recursive=True):
        if _tname(tp) != "teamparticipants":
            continue
        for p in tp.params:
            for op in p.value.filter_templates(recursive=False):
                if _tname(op) != "opponent" or not op.params or op.params[0].showkey:
                    continue
                team = str(op.params[0].value).strip()
                if not team or not op.has("players"):
                    continue
                starters, subs = [], []
                for person in op.get("players").value.filter_templates(recursive=True):
                    if _tname(person) != "person" or not person.params:
                        continue
                    if (_val(person, "type") or "").lower() == "staff":
                        continue
                    pname = _val(person, "link") or str(person.params[0].value).strip()
                    role = (_val(person, "role") or "").lower()
                    if role.startswith("sub") or (_val(person, "type") or "").lower() == "sub":
                        subs.append(pname)
                    else:
                        starters.append(pname)
                subs = starters[3:] + subs
                out.append(RosterEntry(team_name=team, team_link=team,
                                       players=starters[:3], subs=subs))
    return out


def _opponent(tpl) -> tuple[str | None, int | None, bool]:
    """Return (team name, score, walkover-flag) from an `opponentN` value."""
    inner = [t for t in tpl.value.filter_templates(recursive=False)]
    if not inner:
        return None, None, False
    op = inner[0]
    if _tname(op) not in ("teamopponent", "opponent"):
        return None, None, False
    name = str(op.params[0].value).strip() if op.params and not op.params[0].showkey else None
    if name is None:
        name = _val(op, "template") or _val(op, "1")
    raw_score = _val(op, "score")
    wo = raw_score is not None and raw_score.upper() in {"W", "FF", "L", "DQ", "-1"}
    return name, _int(raw_score), wo


def _parse_map(tpl, n: int) -> GameResult | None:
    s1, s2 = _int(_val(tpl, "score1")), _int(_val(tpl, "score2"))
    winner = _int(_val(tpl, "winner"))
    if winner not in (1, 2):
        winner = None
        if s1 is not None and s2 is not None and s1 != s2:
            winner = 1 if s1 > s2 else 2
    if winner is None and s1 is None and s2 is None:
        return None
    ot = (_val(tpl, "ot") or "").lower() in {"true", "t", "1", "yes", "y"}
    return GameResult(n=n, score1=s1, score2=s2, winner=winner, ot=ot)


def parse_matches(text: str, page_date: date | None = None) -> list[SeriesResult]:
    code = mwp.parse(text)
    out: list[SeriesResult] = []
    section: str | None = None
    counter = 0

    # Stable keys: "<bracket/matchlist id>:<param name>" (e.g. "wwRwcsAQ8R:R1M1").
    keys: dict[int, str] = {}
    for parent in code.filter_templates(recursive=True):
        if _tname(parent) == "match":
            continue
        pid = _val(parent, "id") or _tname(parent)
        for p in parent.params:
            for child in p.value.filter_templates(recursive=False):
                if _tname(child) == "match":
                    keys[id(child)] = f"{pid}:{str(p.name).strip()}"

    for tpl in code.filter_templates(recursive=True):
        name = _tname(tpl)
        if name == "matchsection" and tpl.params:
            section = str(tpl.params[0].value).strip()
            continue
        if name != "match":
            continue
        counter += 1
        if not (tpl.has("opponent1") and tpl.has("opponent2")):
            continue

        t1, s1, wo1 = _opponent(tpl.get("opponent1"))
        t2, s2, wo2 = _opponent(tpl.get("opponent2"))
        if not t1 or not t2 or team_key(t1) in TBD_NAMES or team_key(t2) in TBD_NAMES:
            continue

        games: list[GameResult] = []
        for i in range(1, 10):
            if not tpl.has(f"map{i}"):
                continue
            inner = tpl.get(f"map{i}").value.filter_templates(recursive=False)
            if inner and _tname(inner[0]) == "map":
                g = _parse_map(inner[0], i)
                if g:
                    games.append(g)

        # Series score: explicit, else count map winners.
        if s1 is None and s2 is None and games:
            s1 = sum(1 for g in games if g.winner == 1)
            s2 = sum(1 for g in games if g.winner == 2)

        finished_flag = (_val(tpl, "finished") or "").lower() in {"true", "t", "1", "yes"}
        walkover = wo1 or wo2 or bool(_val(tpl, "walkover"))
        bestof = _int(_val(tpl, "bestof"))
        if bestof is None and s1 is not None and s2 is not None and max(s1, s2) > 0:
            bestof = 2 * max(s1, s2) - 1
        finished = finished_flag or (
            s1 is not None and s2 is not None and bestof is not None
            and max(s1, s2) == (bestof + 1) // 2 and s1 != s2
        )

        mkey = keys.get(id(tpl), f"m{counter}")

        out.append(SeriesResult(
            match_key=mkey,
            team1=t1,
            team2=t2,
            score1=s1,
            score2=s2,
            finished=finished,
            best_of=bestof,
            date=parse_date(_val(tpl, "date")) or page_date,
            games=games,
            ballchasing=_val(tpl, "ballchasing"),
            section=section,
            walkover=walkover,
        ))
    return out


# ---------------------------------------------------------------------------
# Active squads. Team pages only contain `{{ActiveSquadAuto}}`, which Liquipedia fills from
# its transfer database, so the squad is read from the template's *expanded* output: an
# HTML table (ID | Name | [role] | Join Date) whose ID cell holds a `[[Page|Display]]` link.
STARTER_ROLES = {"", "loan", "uncontracted", "trial"}
SUB_ROLES = {"substitute", "stand-in", "standin"}


@dataclass
class SquadMember:
    name: str              # display name
    link: str              # Liquipedia page name (the player id source)
    role: str              # "" for a regular player
    join_date: date | None

    @property
    def is_sub(self) -> bool:
        return self.role.lower() in SUB_ROLES


_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t([hd])[^>]*>(.*?)</t\1>", re.S)
_LINK_RE = re.compile(r"\[\[(?!File:)([^|\]]+)(?:\|([^\]]*))?\]\]", re.I)


def _strip(html: str) -> str:
    html = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", html, flags=re.S)
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ").strip()


def parse_squad(expanded: str) -> list[SquadMember]:
    """Players (no staff) from an expanded `{{ActiveSquadAuto}}` table."""
    out: list[SquadMember] = []
    cols: list[str] | None = None
    for row in _ROW_RE.findall(expanded):
        cells = _CELL_RE.findall(row)
        if cells and all(kind == "h" for kind, _ in cells):
            cols = [_strip(c).lower() for _, c in cells]
            continue
        if not cols or not cells:
            continue
        vals = [c for _, c in cells]
        by = dict(zip(cols, vals))
        m = _LINK_RE.search(vals[0])
        if not m:
            continue
        link = m.group(1).strip()
        name = (m.group(2) or link).strip()
        # The role column has an empty header; it is omitted when nobody has a role.
        role = _strip(by[""]) if "" in by else ""
        if role.lower() not in STARTER_ROLES | SUB_ROLES:
            continue  # coaches, analysts, managers, ...
        out.append(SquadMember(name, link, role, parse_date(_strip(by.get("join date", "")))))
    return out


def split_expanded(expanded: str) -> dict[int, str]:
    """Split a batched expansion (`§0§...§1§...`) back into its numbered parts."""
    parts = re.split(r"§(\d+)§", expanded)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
