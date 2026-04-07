from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd
import requests

from formatting import clean_text, coerce_float, coerce_int

BASE_URL = 'https://statsapi.mlb.com/api/v1'
STATCAST_URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
TIMEOUT = 25
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json,text/csv,*/*',
}

DEFAULT_TEAM_ID = 118  # Kansas City Royals

# ---------------------------------------------------------------------------
# Canonical team metadata for all 30 MLB teams
# ---------------------------------------------------------------------------

TEAM_META: dict[int, dict[str, Any]] = {
    108: {'id': 108, 'abbreviation': 'LAA', 'full_name': 'Los Angeles Angels',     'league': 'AL', 'division': 'AL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/108.svg'},
    109: {'id': 109, 'abbreviation': 'ARI', 'full_name': 'Arizona Diamondbacks',   'league': 'NL', 'division': 'NL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/109.svg'},
    110: {'id': 110, 'abbreviation': 'BAL', 'full_name': 'Baltimore Orioles',      'league': 'AL', 'division': 'AL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/110.svg'},
    111: {'id': 111, 'abbreviation': 'BOS', 'full_name': 'Boston Red Sox',         'league': 'AL', 'division': 'AL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/111.svg'},
    112: {'id': 112, 'abbreviation': 'CHC', 'full_name': 'Chicago Cubs',           'league': 'NL', 'division': 'NL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/112.svg'},
    113: {'id': 113, 'abbreviation': 'CIN', 'full_name': 'Cincinnati Reds',        'league': 'NL', 'division': 'NL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/113.svg'},
    114: {'id': 114, 'abbreviation': 'CLE', 'full_name': 'Cleveland Guardians',    'league': 'AL', 'division': 'AL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/114.svg'},
    115: {'id': 115, 'abbreviation': 'COL', 'full_name': 'Colorado Rockies',       'league': 'NL', 'division': 'NL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/115.svg'},
    116: {'id': 116, 'abbreviation': 'DET', 'full_name': 'Detroit Tigers',         'league': 'AL', 'division': 'AL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/116.svg'},
    117: {'id': 117, 'abbreviation': 'HOU', 'full_name': 'Houston Astros',         'league': 'AL', 'division': 'AL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/117.svg'},
    118: {'id': 118, 'abbreviation': 'KC',  'full_name': 'Kansas City Royals',     'league': 'AL', 'division': 'AL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/118.svg'},
    119: {'id': 119, 'abbreviation': 'LAD', 'full_name': 'Los Angeles Dodgers',    'league': 'NL', 'division': 'NL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/119.svg'},
    120: {'id': 120, 'abbreviation': 'WSH', 'full_name': 'Washington Nationals',   'league': 'NL', 'division': 'NL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/120.svg'},
    121: {'id': 121, 'abbreviation': 'NYM', 'full_name': 'New York Mets',          'league': 'NL', 'division': 'NL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/121.svg'},
    133: {'id': 133, 'abbreviation': 'ATH', 'full_name': 'Athletics',              'league': 'AL', 'division': 'AL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/133.svg'},
    134: {'id': 134, 'abbreviation': 'PIT', 'full_name': 'Pittsburgh Pirates',     'league': 'NL', 'division': 'NL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/134.svg'},
    135: {'id': 135, 'abbreviation': 'SD',  'full_name': 'San Diego Padres',       'league': 'NL', 'division': 'NL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/135.svg'},
    136: {'id': 136, 'abbreviation': 'SEA', 'full_name': 'Seattle Mariners',       'league': 'AL', 'division': 'AL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/136.svg'},
    137: {'id': 137, 'abbreviation': 'SF',  'full_name': 'San Francisco Giants',   'league': 'NL', 'division': 'NL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/137.svg'},
    138: {'id': 138, 'abbreviation': 'STL', 'full_name': 'St. Louis Cardinals',    'league': 'NL', 'division': 'NL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/138.svg'},
    139: {'id': 139, 'abbreviation': 'TB',  'full_name': 'Tampa Bay Rays',         'league': 'AL', 'division': 'AL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/139.svg'},
    140: {'id': 140, 'abbreviation': 'TEX', 'full_name': 'Texas Rangers',          'league': 'AL', 'division': 'AL West',    'logo_url': 'https://www.mlbstatic.com/team-logos/140.svg'},
    141: {'id': 141, 'abbreviation': 'TOR', 'full_name': 'Toronto Blue Jays',      'league': 'AL', 'division': 'AL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/141.svg'},
    142: {'id': 142, 'abbreviation': 'MIN', 'full_name': 'Minnesota Twins',        'league': 'AL', 'division': 'AL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/142.svg'},
    143: {'id': 143, 'abbreviation': 'PHI', 'full_name': 'Philadelphia Phillies',  'league': 'NL', 'division': 'NL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/143.svg'},
    144: {'id': 144, 'abbreviation': 'ATL', 'full_name': 'Atlanta Braves',         'league': 'NL', 'division': 'NL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/144.svg'},
    145: {'id': 145, 'abbreviation': 'CWS', 'full_name': 'Chicago White Sox',      'league': 'AL', 'division': 'AL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/145.svg'},
    146: {'id': 146, 'abbreviation': 'MIA', 'full_name': 'Miami Marlins',          'league': 'NL', 'division': 'NL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/146.svg'},
    147: {'id': 147, 'abbreviation': 'NYY', 'full_name': 'New York Yankees',       'league': 'AL', 'division': 'AL East',    'logo_url': 'https://www.mlbstatic.com/team-logos/147.svg'},
    158: {'id': 158, 'abbreviation': 'MIL', 'full_name': 'Milwaukee Brewers',      'league': 'NL', 'division': 'NL Central', 'logo_url': 'https://www.mlbstatic.com/team-logos/158.svg'},
}

# Pre-built lookup indexes (built once at import time)
_ABBR_INDEX: dict[str, int] = {meta['abbreviation'].upper(): tid for tid, meta in TEAM_META.items()}
_NAME_INDEX: dict[str, int] = {meta['full_name'].lower(): tid for tid, meta in TEAM_META.items()}

# ---------------------------------------------------------------------------
# Division / league ID mappings (MLB Stats API values)
# ---------------------------------------------------------------------------

DIVISION_LEAGUE_IDS: dict[str, int] = {
    'AL East':    103,
    'AL Central': 103,
    'AL West':    103,
    'NL East':    104,
    'NL Central': 104,
    'NL West':    104,
}

DIVISION_IDS: dict[str, int] = {
    'AL East':    201,
    'AL Central': 202,
    'AL West':    200,
    'NL East':    204,
    'NL Central': 205,
    'NL West':    203,
}

# Convenience: all known division names
ALL_DIVISIONS: list[str] = list(DIVISION_IDS.keys())

# FALLBACK_TEAMS kept for backward-compat with load_teams()
FALLBACK_TEAMS = [
    {'id': meta['id'], 'name': meta['full_name'], 'abbreviation': meta['abbreviation'], 'division': meta['division']}
    for meta in TEAM_META.values()
]


# ---------------------------------------------------------------------------
# Team-meta helper functions
# ---------------------------------------------------------------------------

def get_team_meta(team_id_or_abbr_or_name: int | str) -> dict[str, Any] | None:
    """Return canonical team metadata dict or None if not found.

    Accepts team_id (int), abbreviation (str, case-insensitive), or
    full_name (str, case-insensitive).
    """
    key = team_id_or_abbr_or_name
    if isinstance(key, int):
        return TEAM_META.get(key)
    if isinstance(key, str):
        # Try as numeric id first
        try:
            return TEAM_META.get(int(key))
        except (ValueError, TypeError):
            pass
        upper = key.strip().upper()
        if upper in _ABBR_INDEX:
            return TEAM_META[_ABBR_INDEX[upper]]
        lower = key.strip().lower()
        if lower in _NAME_INDEX:
            return TEAM_META[_NAME_INDEX[lower]]
    return None


def get_team_logo(team_id_or_abbr_or_name: int | str) -> str:
    """Return the SVG logo URL for a team, or empty string if not found."""
    meta = get_team_meta(team_id_or_abbr_or_name)
    return meta['logo_url'] if meta else ''


def get_team_division(team_id_or_abbr_or_name: int | str) -> str:
    """Return division name for a team, or empty string if not found."""
    meta = get_team_meta(team_id_or_abbr_or_name)
    return meta['division'] if meta else ''


def get_teams_in_division(division_name: str) -> list[dict[str, Any]]:
    """Return list of team metadata dicts for all teams in the given division."""
    return [meta for meta in TEAM_META.values() if meta['division'] == division_name]


# ---------------------------------------------------------------------------
# Core MLB API error type
# ---------------------------------------------------------------------------

class MLBApiError(Exception):
    pass


# ---------------------------------------------------------------------------
# MLB API client
# ---------------------------------------------------------------------------

@dataclass
class MLBClient:
    base_url: str = BASE_URL
    timeout: int = TIMEOUT

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f'{self.base_url}{path}'
        response = requests.get(url, params=params or {}, timeout=self.timeout, headers=HEADERS)
        if not response.ok:
            detail = response.text[:250] if response.text else ''
            raise MLBApiError(f'MLB API HTTP {response.status_code} for {path}. {detail}')
        return response.json()

    def get_teams(self) -> list[dict[str, Any]]:
        return self._get('/teams', {'sportId': 1}).get('teams', [])

    def get_schedule(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        data = self._get('/schedule', params)
        rows: list[dict[str, Any]] = []
        for day in data.get('dates', []):
            rows.extend(day.get('games', []))
        return rows

    def get_live_feed(self, game_pk: int) -> dict[str, Any]:
        return self._get(f'/game/{game_pk}/feed/live')

    def get_standings(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._get('/standings', params)

    def get_statcast(self, params: dict[str, Any]) -> pd.DataFrame:
        response = requests.get(STATCAST_URL, params=params, timeout=self.timeout, headers=HEADERS)
        if not response.ok:
            detail = response.text[:250] if response.text else ''
            raise MLBApiError(f'Statcast HTTP {response.status_code}. {detail}')
        text = response.text or ''
        if not text.strip() or text.lstrip().startswith('<'):
            raise MLBApiError('Statcast did not return CSV data.')
        try:
            return pd.read_csv(StringIO(text))
        except Exception as exc:
            raise MLBApiError(f'Statcast CSV parse failed. {exc}') from exc

    # ------------------------------------------------------------------
    # Standings helpers
    # ------------------------------------------------------------------

    def get_standings_for_division(self, division_name: str, season: int) -> tuple[pd.DataFrame, str | None]:
        """Return standings DataFrame for a single division.

        Columns: team_id, team_name, wins, losses, gb, win_pct, division_rank
        """
        _empty = pd.DataFrame(columns=['team_id', 'team_name', 'wins', 'losses', 'gb', 'win_pct', 'division_rank'])
        division_id = DIVISION_IDS.get(division_name)
        league_id = DIVISION_LEAGUE_IDS.get(division_name)
        if division_id is None or league_id is None:
            return _empty, f'Unknown division: {division_name!r}'
        try:
            data = self.get_standings({
                'leagueId': league_id,
                'standingsTypes': 'regularSeason',
                'season': season,
            })
            rows: list[dict[str, Any]] = []
            for record in data.get('records', []):
                div_info = record.get('division') or {}
                if coerce_int(div_info.get('id'), -1) != division_id:
                    continue
                for tr in record.get('teamRecords', []):
                    team_info = tr.get('team') or {}
                    gb_raw = tr.get('gamesBack', '--')
                    gb_str = '--' if str(gb_raw).strip() in ('', '-', 'None', 'null') else str(gb_raw).strip()
                    wins = coerce_int(tr.get('wins'), 0)
                    losses = coerce_int(tr.get('losses'), 0)
                    played = wins + losses
                    win_pct = round(wins / played, 3) if played > 0 else 0.0
                    rows.append({
                        'team_id': coerce_int(team_info.get('id'), 0),
                        'team_name': clean_text(team_info.get('name')),
                        'wins': wins,
                        'losses': losses,
                        'gb': gb_str,
                        'win_pct': win_pct,
                        'division_rank': coerce_int(tr.get('divisionRank'), 0),
                    })
            if not rows:
                return _empty, f'No standings data returned for {division_name} season {season}.'
            df = pd.DataFrame(rows).sort_values('division_rank').reset_index(drop=True)
            df['team_id'] = df['team_id'].astype(int)
            df['wins'] = df['wins'].astype(int)
            df['losses'] = df['losses'].astype(int)
            df['win_pct'] = df['win_pct'].astype(float)
            df['division_rank'] = df['division_rank'].astype(int)
            df['gb'] = df['gb'].astype(str)
            df['team_name'] = df['team_name'].astype(str)
            return df, None
        except MLBApiError as exc:
            return _empty, str(exc)
        except Exception as exc:
            return _empty, f'Unexpected error fetching division standings: {exc}'

    def get_wildcard_standings_by_league(self, league: str, season: int) -> tuple[pd.DataFrame, str | None]:
        """Return wild-card standings for one league ('AL' or 'NL').

        Columns: team_id, team_name, wins, losses, gb, win_pct, wc_rank
        """
        _empty = pd.DataFrame(columns=['team_id', 'team_name', 'wins', 'losses', 'gb', 'win_pct', 'wc_rank'])
        league_upper = league.strip().upper()
        league_id_map = {'AL': 103, 'NL': 104}
        league_id = league_id_map.get(league_upper)
        if league_id is None:
            return _empty, f'Unknown league: {league!r}. Use "AL" or "NL".'
        try:
            data = self.get_standings({
                'leagueId': league_id,
                'standingsTypes': 'wildCard',
                'season': season,
            })
            rows: list[dict[str, Any]] = []
            for record in data.get('records', []):
                for tr in record.get('teamRecords', []):
                    team_info = tr.get('team') or {}
                    gb_raw = tr.get('wildCardGamesBack', tr.get('gamesBack', '--'))
                    gb_str = '--' if str(gb_raw).strip() in ('', '-', 'None', 'null') else str(gb_raw).strip()
                    wins = coerce_int(tr.get('wins'), 0)
                    losses = coerce_int(tr.get('losses'), 0)
                    played = wins + losses
                    win_pct = round(wins / played, 3) if played > 0 else 0.0
                    rows.append({
                        'team_id': coerce_int(team_info.get('id'), 0),
                        'team_name': clean_text(team_info.get('name')),
                        'wins': wins,
                        'losses': losses,
                        'gb': gb_str,
                        'win_pct': win_pct,
                        'wc_rank': coerce_int(tr.get('wildCardRank'), 0),
                    })
            if not rows:
                return _empty, f'No wild-card standings returned for {league_upper} season {season}.'
            df = pd.DataFrame(rows).sort_values('wc_rank').reset_index(drop=True)
            df['team_id'] = df['team_id'].astype(int)
            df['wins'] = df['wins'].astype(int)
            df['losses'] = df['losses'].astype(int)
            df['win_pct'] = df['win_pct'].astype(float)
            df['wc_rank'] = df['wc_rank'].astype(int)
            df['gb'] = df['gb'].astype(str)
            df['team_name'] = df['team_name'].astype(str)
            return df, None
        except MLBApiError as exc:
            return _empty, str(exc)
        except Exception as exc:
            return _empty, f'Unexpected error fetching wild-card standings: {exc}'

    def get_team_schedule_range(self, team_id: int, start_date: str, end_date: str) -> tuple[pd.DataFrame, str | None]:
        """Return completed-game schedule for a team over a date range.

        Columns match _games_to_df output; only games with abstract_status=='Final' are returned.
        """
        try:
            games = self.get_schedule({
                'sportId': 1,
                'teamId': team_id,
                'startDate': start_date,
                'endDate': end_date,
                'gameType': 'R',
                'hydrate': 'team',
            })
            df = _games_to_df(games)
            if df.empty:
                return df, None
            completed = df[df['abstract_status'] == 'Final'].reset_index(drop=True)
            return completed, None
        except Exception as exc:
            return _games_to_df([]), str(exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _games_to_df(games: list[dict[str, Any]]) -> pd.DataFrame:
    _cols = ['gamePk', 'gameDate', 'officialDate', 'away', 'home',
             'away_id', 'home_id', 'away_score', 'home_score', 'status', 'abstract_status']
    rows: list[dict[str, Any]] = []
    for g in games:
        teams = g.get('teams', {})
        away_team = (teams.get('away') or {}).get('team') or {}
        home_team = (teams.get('home') or {}).get('team') or {}
        status_obj = g.get('status') or {}
        rows.append({
            'gamePk': coerce_int(g.get('gamePk'), 0),
            'gameDate': clean_text(g.get('gameDate'), ''),
            'officialDate': clean_text(g.get('officialDate'), ''),
            'away': clean_text(away_team.get('name')),
            'home': clean_text(home_team.get('name')),
            'away_id': coerce_int(away_team.get('id'), 0),
            'home_id': coerce_int(home_team.get('id'), 0),
            'away_score': coerce_int((teams.get('away') or {}).get('score'), 0),
            'home_score': coerce_int((teams.get('home') or {}).get('score'), 0),
            'status': clean_text(status_obj.get('detailedState') or status_obj.get('abstractGameState'), 'Unknown'),
            'abstract_status': clean_text(status_obj.get('abstractGameState'), 'Unknown'),
        })
    if not rows:
        return pd.DataFrame(columns=_cols)
    df = pd.DataFrame(rows)
    return df.sort_values(['officialDate', 'gameDate', 'gamePk']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level API functions
# ---------------------------------------------------------------------------

def load_teams() -> tuple[pd.DataFrame, str | None]:
    client = MLBClient()
    try:
        teams = client.get_teams()
        rows = []
        for t in teams:
            rows.append({
                'id': coerce_int(t.get('id'), 0),
                'name': clean_text(t.get('name')),
                'abbreviation': clean_text(t.get('abbreviation')),
                'division': clean_text(((t.get('division') or {}).get('name'))),
            })
        df = pd.DataFrame(rows).sort_values('name').reset_index(drop=True)
        return df, None
    except Exception as exc:
        fallback = pd.DataFrame(FALLBACK_TEAMS).sort_values('name').reset_index(drop=True)
        return fallback, f'Live MLB teams call failed. Using built-in team list instead. {exc}'


def build_schedule_df(team_id: int, target_date: str) -> tuple[pd.DataFrame, str | None]:
    client = MLBClient()
    try:
        games = client.get_schedule({'sportId': 1, 'date': target_date, 'teamId': team_id})
        return _games_to_df(games), None
    except Exception as exc:
        return _games_to_df([]), str(exc)


def build_season_df(team_id: int, season: int, end_date: str) -> tuple[pd.DataFrame, str | None]:
    client = MLBClient()
    try:
        games = client.get_schedule({
            'sportId': 1,
            'teamId': team_id,
            'startDate': f'{season}-01-01',
            'endDate': end_date,
            'gameType': 'R',
            'hydrate': 'team',
        })
        return _games_to_df(games), None
    except Exception as exc:
        return _games_to_df([]), str(exc)


def choose_live_game_pk(schedule_df: pd.DataFrame) -> int | None:
    if schedule_df.empty:
        return None
    live_statuses = {'In Progress', 'Manager Challenge', 'Delayed Start', 'Delayed', 'Suspended'}
    live_candidates = schedule_df[schedule_df['status'].isin(live_statuses)].copy()
    if live_candidates.empty:
        return None
    for _, row in live_candidates.iterrows():
        game_pk = coerce_int(row.get('gamePk'), 0)
        if game_pk > 0:
            return game_pk
    return None


def get_live_summary(game_pk: int | None) -> tuple[dict[str, Any], str | None]:
    if not game_pk:
        return {}, None
    client = MLBClient()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            data = client.get_live_feed(game_pk)
            game_data = data.get('gameData', {})
            live_data = data.get('liveData', {})
            linescore = live_data.get('linescore', {})
            teams = game_data.get('teams', {})
            away = teams.get('away', {})
            home = teams.get('home', {})
            return {
                'gamePk': game_pk,
                'away_team': clean_text(away.get('name')),
                'home_team': clean_text(home.get('name')),
                'status': clean_text((game_data.get('status') or {}).get('detailedState')),
                'inning': clean_text(linescore.get('currentInningOrdinal'), '-'),
                'inning_state': clean_text(linescore.get('inningState'), '-'),
                'away_runs': coerce_int((linescore.get('teams') or {}).get('away', {}).get('runs'), 0),
                'home_runs': coerce_int((linescore.get('teams') or {}).get('home', {}).get('runs'), 0),
                'balls': coerce_int(linescore.get('balls'), 0),
                'strikes': coerce_int(linescore.get('strikes'), 0),
                'outs': coerce_int(linescore.get('outs'), 0),
            }, None
        except Exception as exc:
            if attempt == max_retries - 1:
                return {}, str(exc)
    return {}, 'Live feed unavailable'


def get_wildcard_standings_combined(season: int) -> tuple[pd.DataFrame, str | None]:
    """Wild-card standings for both leagues combined (AL + NL).

    Columns: team_id, team_name, wildcard_rank, wins, losses
    """
    client = MLBClient()
    try:
        data = client.get_standings({
            'leagueId': '103,104',
            'standingsTypes': 'wildCard',
            'season': season,
        })
        rows: list[dict[str, Any]] = []
        for division_data in data.get('records', []):
            for team_record in division_data.get('teamRecords', []):
                team_info = team_record.get('team', {})
                wc_rank = team_record.get('wildCardRank')
                rows.append({
                    'team_id': coerce_int(team_info.get('id'), 0),
                    'team_name': clean_text(team_info.get('name')),
                    'wildcard_rank': wc_rank if wc_rank else None,
                    'wins': coerce_int(team_record.get('wins'), 0),
                    'losses': coerce_int(team_record.get('losses'), 0),
                })
        df = pd.DataFrame(rows)
        return df, None
    except Exception as exc:
        return pd.DataFrame(columns=['team_id', 'team_name', 'wildcard_rank', 'wins', 'losses']), str(exc)


# Backward-compat alias for code that previously called get_wildcard_standings
get_wildcard_standings = get_wildcard_standings_combined


def get_division_standings(division_name: str, season: int) -> tuple[pd.DataFrame, str | None]:
    """Return standings for a single division.

    Delegates to MLBClient.get_standings_for_division.
    Columns: team_id, team_name, wins, losses, gb, win_pct, division_rank
    """
    return MLBClient().get_standings_for_division(division_name, season)


def get_wildcard_standings_league(league: str, season: int) -> tuple[pd.DataFrame, str | None]:
    """Return wild-card standings for one league ('AL' or 'NL').

    Delegates to MLBClient.get_wildcard_standings_by_league.
    Columns: team_id, team_name, wins, losses, gb, win_pct, wc_rank
    """
    return MLBClient().get_wildcard_standings_by_league(league, season)


def get_upcoming_schedule(team_id: int, from_date: str, num_games: int = 10) -> tuple[pd.DataFrame, str | None]:
    """Return the next *num_games* scheduled (not yet played) games for a team.

    Games are fetched starting from *from_date* for a 60-day window and
    filtered to those that have not been completed (abstract_status != 'Final').

    Columns: gamePk, gameDate, officialDate, away, home, away_id, home_id,
             away_score, home_score, status, abstract_status
    """
    _empty = _games_to_df([])
    try:
        from datetime import date, timedelta
        try:
            start = date.fromisoformat(from_date)
        except ValueError:
            return _empty, f'Invalid from_date: {from_date!r}. Expected YYYY-MM-DD.'
        end = start + timedelta(days=60)
        client = MLBClient()
        games = client.get_schedule({
            'sportId': 1,
            'teamId': team_id,
            'startDate': start.isoformat(),
            'endDate': end.isoformat(),
            'gameType': 'R',
            'hydrate': 'team',
        })
        df = _games_to_df(games)
        if df.empty:
            return df, None
        upcoming = df[df['abstract_status'] != 'Final'].head(num_games).reset_index(drop=True)
        return upcoming, None
    except Exception as exc:
        return _empty, str(exc)


def get_statcast_team_df(team_abbr: str, start_date: str, end_date: str, player_type: str = 'batter') -> tuple[pd.DataFrame, str | None]:
    team_code = clean_text(team_abbr, '').upper()
    if not team_code:
        return pd.DataFrame(), 'Team abbreviation missing for Statcast lookup.'

    season = start_date[:4]
    params = {
        'all': 'true',
        'player_type': player_type,
        'game_date_gt': start_date,
        'game_date_lt': end_date,
        'team': team_code,
        'hfGT': 'R|',
        'type': 'details',
        'min_pitches': '0',
        'min_results': '0',
        'group_by': 'name',
        'sort_col': 'pitches',
        'sort_order': 'desc',
        'hfSea': f'{season}|',
    }
    client = MLBClient()
    try:
        df = client.get_statcast(params)
        if df is None or df.empty:
            return pd.DataFrame(), 'Statcast returned no rows for the selected window.'
        out = df.copy()
        for col in ['launch_speed', 'launch_angle', 'hit_distance_sc', 'release_speed',
                    'release_spin_rate', 'estimated_woba_using_speedangle', 'woba_value', 'hc_x', 'hc_y']:
            if col in out.columns:
                out[col] = out[col].apply(coerce_float)
        return out, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ---------------------------------------------------------------------------
# Live game helpers (Live Feed)
# ---------------------------------------------------------------------------

def get_current_team_game(team_id: int) -> tuple[dict[str, Any], str | None]:
    """Find the current live game, most recent completed game, or next scheduled
    game for a team.

    Returns a dict with keys: gamePk, gameDate, officialDate, away, home,
    away_id, home_id, away_score, home_score, status, abstract_status,
    and a 'game_phase' key ('live', 'final', 'scheduled', or 'none').
    """
    from datetime import date, timedelta
    today = date.today().isoformat()
    client = MLBClient()
    empty: dict[str, Any] = {}

    # 1. Check today's schedule
    try:
        games_today = client.get_schedule({
            'sportId': 1, 'date': today, 'teamId': team_id, 'hydrate': 'team',
        })
        df_today = _games_to_df(games_today)
        if not df_today.empty:
            live_statuses = {'In Progress', 'Manager Challenge', 'Delayed', 'Delayed Start', 'Suspended'}
            live_rows = df_today[df_today['status'].isin(live_statuses)]
            if not live_rows.empty:
                row = live_rows.iloc[0].to_dict()
                row['game_phase'] = 'live'
                return row, None
            final_rows = df_today[df_today['abstract_status'] == 'Final']
            if not final_rows.empty:
                row = final_rows.iloc[-1].to_dict()
                row['game_phase'] = 'final'
                return row, None
            sched_rows = df_today[df_today['abstract_status'] == 'Preview']
            if not sched_rows.empty:
                row = sched_rows.iloc[0].to_dict()
                row['game_phase'] = 'scheduled'
                return row, None
            # Any game today
            row = df_today.iloc[0].to_dict()
            row['game_phase'] = 'scheduled'
            return row, None
    except Exception:
        pass

    # 2. Fallback: most recent final game (last 7 days)
    try:
        start_lookback = (date.fromisoformat(today) - timedelta(days=7)).isoformat()
        games_recent = client.get_schedule({
            'sportId': 1, 'teamId': team_id,
            'startDate': start_lookback, 'endDate': today,
            'gameType': 'R', 'hydrate': 'team',
        })
        df_recent = _games_to_df(games_recent)
        if not df_recent.empty:
            finals = df_recent[df_recent['abstract_status'] == 'Final']
            if not finals.empty:
                row = finals.iloc[-1].to_dict()
                row['game_phase'] = 'final'
                return row, None
    except Exception:
        pass

    # 3. Next scheduled game (7 days ahead)
    try:
        end_look = (date.fromisoformat(today) + timedelta(days=7)).isoformat()
        games_next = client.get_schedule({
            'sportId': 1, 'teamId': team_id,
            'startDate': today, 'endDate': end_look,
            'gameType': 'R', 'hydrate': 'team',
        })
        df_next = _games_to_df(games_next)
        if not df_next.empty:
            sched = df_next[df_next['abstract_status'] != 'Final']
            if not sched.empty:
                row = sched.iloc[0].to_dict()
                row['game_phase'] = 'scheduled'
                return row, None
    except Exception:
        pass

    return empty, 'No current game found for team.'


def get_game_status(game_pk: int) -> tuple[str, str | None]:
    """Return (detailedState, error_or_None) for a game."""
    if not game_pk:
        return 'Unknown', 'No gamePk provided.'
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        state = (data.get('gameData', {}).get('status') or {}).get('detailedState', 'Unknown')
        return clean_text(state, 'Unknown'), None
    except Exception as exc:
        return 'Unknown', str(exc)


def get_live_scorecard(game_pk: int) -> tuple[dict[str, Any], str | None]:
    """Return a full parsed scorecard dict from the live feed.

    Returned keys:
        gamePk, gameDate, venue, away_team, home_team, away_abbr, home_abbr,
        away_id, home_id, status, abstract_status, inning_state,
        current_inning, innings (list of {inning, away, home}),
        away_runs, away_hits, away_errors,
        home_runs, home_hits, home_errors,
        balls, strikes, outs,
        on_1b, on_2b, on_3b,
        current_batter, current_pitcher,
        recent_plays (list of str)
    """
    if not game_pk:
        return {}, 'No gamePk provided.'
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        game_data = data.get('gameData') or {}
        live_data = data.get('liveData') or {}
        linescore = live_data.get('linescore') or {}
        teams_gd = game_data.get('teams') or {}
        away_gd = teams_gd.get('away') or {}
        home_gd = teams_gd.get('home') or {}
        status_obj = game_data.get('status') or {}
        venue_obj = game_data.get('venue') or {}
        datetime_obj = game_data.get('datetime') or {}

        # Innings
        innings_raw = linescore.get('innings') or []
        innings: list[dict[str, Any]] = []
        for inn in innings_raw:
            away_inn = inn.get('away') or {}
            home_inn = inn.get('home') or {}
            innings.append({
                'inning': coerce_int(inn.get('num'), 0),
                'away': away_inn.get('runs') if away_inn.get('runs') is not None else '-',
                'home': home_inn.get('runs') if home_inn.get('runs') is not None else '-',
            })

        # Team totals
        ls_teams = linescore.get('teams') or {}
        ls_away = ls_teams.get('away') or {}
        ls_home = ls_teams.get('home') or {}

        # Baserunners
        offense = linescore.get('offense') or {}
        on_1b = bool(offense.get('first'))
        on_2b = bool(offense.get('second'))
        on_3b = bool(offense.get('third'))
        cur_batter = (offense.get('batter') or {}).get('fullName', '')
        cur_pitcher = (linescore.get('defense') or {}).get('pitcher', {}) if isinstance(linescore.get('defense'), dict) else {}
        cur_pitcher_name = cur_pitcher.get('fullName', '') if isinstance(cur_pitcher, dict) else ''

        # Recent plays
        plays_obj = live_data.get('plays') or {}
        all_plays_raw = plays_obj.get('allPlays') or []
        recent_plays: list[str] = []
        for p in all_plays_raw[-10:]:
            result = (p.get('result') or {})
            desc = clean_text(result.get('description'), '')
            if desc:
                recent_plays.append(desc)

        scorecard: dict[str, Any] = {
            'gamePk': game_pk,
            'gameDate': clean_text(datetime_obj.get('officialDate') or game_data.get('game', {}).get('id', ''), ''),
            'venue': clean_text(venue_obj.get('name'), 'Unknown Venue'),
            'away_team': clean_text(away_gd.get('name'), 'Away'),
            'home_team': clean_text(home_gd.get('name'), 'Home'),
            'away_abbr': clean_text((away_gd.get('teamCode') or away_gd.get('abbreviation')), 'AWY'),
            'home_abbr': clean_text((home_gd.get('teamCode') or home_gd.get('abbreviation')), 'HME'),
            'away_id': coerce_int(away_gd.get('id'), 0),
            'home_id': coerce_int(home_gd.get('id'), 0),
            'status': clean_text(status_obj.get('detailedState'), 'Unknown'),
            'abstract_status': clean_text(status_obj.get('abstractGameState'), 'Unknown'),
            'inning_state': clean_text(linescore.get('inningState'), ''),
            'current_inning': coerce_int(linescore.get('currentInning'), 0),
            'current_inning_ordinal': clean_text(linescore.get('currentInningOrdinal'), ''),
            'innings': innings,
            'away_runs': coerce_int(ls_away.get('runs'), 0),
            'away_hits': coerce_int(ls_away.get('hits'), 0),
            'away_errors': coerce_int(ls_away.get('errors'), 0),
            'home_runs': coerce_int(ls_home.get('runs'), 0),
            'home_hits': coerce_int(ls_home.get('hits'), 0),
            'home_errors': coerce_int(ls_home.get('errors'), 0),
            'balls': coerce_int(linescore.get('balls'), 0),
            'strikes': coerce_int(linescore.get('strikes'), 0),
            'outs': coerce_int(linescore.get('outs'), 0),
            'on_1b': on_1b,
            'on_2b': on_2b,
            'on_3b': on_3b,
            'current_batter': clean_text(cur_batter, ''),
            'current_pitcher': clean_text(cur_pitcher_name, ''),
            'recent_plays': recent_plays,
        }
        return scorecard, None
    except Exception as exc:
        return {}, str(exc)


def get_game_plays(game_pk: int) -> tuple[list[dict[str, Any]], str | None]:
    """Return parsed play-by-play list from live feed.

    Each play dict has keys: idx, inning, half_inning, outs_before, outs_after,
    balls, strikes, batter_id, batter_name, pitcher_id, pitcher_name,
    event, description, rbi, score_change, away_score_after, home_score_after,
    wpa (float or None), start_runners, end_runners
    """
    if not game_pk:
        return [], 'No gamePk provided.'
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        live_data = data.get('liveData') or {}
        plays_obj = live_data.get('plays') or {}
        all_plays_raw = plays_obj.get('allPlays') or []
        plays: list[dict[str, Any]] = []
        for idx, p in enumerate(all_plays_raw):
            about = p.get('about') or {}
            result = p.get('result') or {}
            matchup = p.get('matchup') or {}
            count = p.get('count') or {}
            runners = p.get('runners') or []
            batter = matchup.get('batter') or {}
            pitcher = matchup.get('pitcher') or {}

            # Compute which runners are on base before/after
            start_runners = set()
            end_runners = set()
            for r in runners:
                mv = r.get('movement') or {}
                det = r.get('details') or {}
                start = mv.get('originBase') or ''
                end_b = mv.get('end') or mv.get('endBase') or ''
                if start:
                    start_runners.add(start)
                if end_b and end_b not in ('score', 'Score'):
                    end_runners.add(end_b)

            play_dict: dict[str, Any] = {
                'idx': idx,
                'inning': coerce_int(about.get('inning'), 0),
                'half_inning': clean_text(about.get('halfInning'), 'top'),
                'outs_before': coerce_int(about.get('outs'), 0),
                'outs_after': coerce_int(count.get('outs'), 0),
                'balls': coerce_int(count.get('balls'), 0),
                'strikes': coerce_int(count.get('strikes'), 0),
                'batter_id': coerce_int(batter.get('id'), 0),
                'batter_name': clean_text(batter.get('fullName'), 'Unknown'),
                'pitcher_id': coerce_int(pitcher.get('id'), 0),
                'pitcher_name': clean_text(pitcher.get('fullName'), 'Unknown'),
                'event': clean_text(result.get('event'), ''),
                'event_type': clean_text(result.get('eventType'), ''),
                'description': clean_text(result.get('description'), ''),
                'rbi': coerce_int(result.get('rbi'), 0),
                'away_score_after': coerce_int(result.get('awayScore'), 0),
                'home_score_after': coerce_int(result.get('homeScore'), 0),
                'start_runners': sorted(start_runners),
                'end_runners': sorted(end_runners),
                'is_scoring_play': bool(about.get('isScoringPlay')),
                'wpa': None,  # populated by data_helpers
                'win_probability_added': coerce_float(
                    (p.get('winProbability') or [{}])[-1].get('homeTeamWinProbabilityAdded') if p.get('winProbability') else None
                ),
            }
            plays.append(play_dict)
        return plays, None
    except Exception as exc:
        return [], str(exc)


def get_game_boxscore(game_pk: int) -> tuple[dict[str, Any], str | None]:
    """Return parsed boxscore dict from the live feed.

    Returned keys:
        away_batters: list of {id, name, ab, r, h, rbi, bb, k, sb, hr, tb, obp, slg}
        home_batters: list of {...}
        away_pitchers: list of {id, name, ip, h, r, er, bb, k, hr, pitches, strikes, era}
        home_pitchers: list of {...}
        away_notes: str  (W/L/S if applicable)
        home_notes: str
    """
    if not game_pk:
        return {}, 'No gamePk provided.'
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        live_data = data.get('liveData') or {}
        boxscore = live_data.get('boxscore') or {}
        box_teams = boxscore.get('teams') or {}

        def _parse_batters(team_box: dict) -> list[dict[str, Any]]:
            batters_order = team_box.get('batters') or []
            players = team_box.get('players') or {}
            result = []
            for pid in batters_order:
                key = f'ID{pid}'
                p = players.get(key) or {}
                person = p.get('person') or {}
                stats = (p.get('stats') or {}).get('batting') or {}
                season_stats = (p.get('seasonStats') or {}).get('batting') or {}
                tb = (
                    coerce_int(stats.get('singles'), 0) * 1 +
                    coerce_int(stats.get('doubles'), 0) * 2 +
                    coerce_int(stats.get('triples'), 0) * 3 +
                    coerce_int(stats.get('homeRuns'), 0) * 4
                )
                if tb == 0:
                    tb = coerce_int(stats.get('totalBases'), 0)
                result.append({
                    'id': coerce_int(person.get('id'), pid),
                    'name': clean_text(person.get('fullName'), 'Unknown'),
                    'position': clean_text((p.get('position') or {}).get('abbreviation'), ''),
                    'ab': coerce_int(stats.get('atBats'), 0),
                    'r': coerce_int(stats.get('runs'), 0),
                    'h': coerce_int(stats.get('hits'), 0),
                    'rbi': coerce_int(stats.get('rbi'), 0),
                    'bb': coerce_int(stats.get('baseOnBalls'), 0),
                    'k': coerce_int(stats.get('strikeOuts'), 0),
                    'sb': coerce_int(stats.get('stolenBases'), 0),
                    'hr': coerce_int(stats.get('homeRuns'), 0),
                    'tb': tb,
                    'avg': clean_text(season_stats.get('avg'), '.---'),
                    'obp': clean_text(season_stats.get('obp'), '.---'),
                    'slg': clean_text(season_stats.get('slg'), '.---'),
                })
            return result

        def _parse_pitchers(team_box: dict) -> list[dict[str, Any]]:
            pitchers_order = team_box.get('pitchers') or []
            players = team_box.get('players') or {}
            result = []
            for pid in pitchers_order:
                key = f'ID{pid}'
                p = players.get(key) or {}
                person = p.get('person') or {}
                stats = (p.get('stats') or {}).get('pitching') or {}
                season_stats = (p.get('seasonStats') or {}).get('pitching') or {}
                result.append({
                    'id': coerce_int(person.get('id'), pid),
                    'name': clean_text(person.get('fullName'), 'Unknown'),
                    'ip': clean_text(stats.get('inningsPitched'), '0.0'),
                    'h': coerce_int(stats.get('hits'), 0),
                    'r': coerce_int(stats.get('runs'), 0),
                    'er': coerce_int(stats.get('earnedRuns'), 0),
                    'bb': coerce_int(stats.get('baseOnBalls'), 0),
                    'k': coerce_int(stats.get('strikeOuts'), 0),
                    'hr': coerce_int(stats.get('homeRuns'), 0),
                    'pitches': coerce_int(stats.get('pitchesThrown'), 0),
                    'strikes': coerce_int(stats.get('strikes'), 0),
                    'era': clean_text(season_stats.get('era'), '-.--'),
                    'game_score': None,
                })
            return result

        away_box = box_teams.get('away') or {}
        home_box = box_teams.get('home') or {}

        # Decisions (W/L/S)
        decisions = boxscore.get('info') or []
        away_notes = ''
        home_notes = ''

        return {
            'away_batters': _parse_batters(away_box),
            'home_batters': _parse_batters(home_box),
            'away_pitchers': _parse_pitchers(away_box),
            'home_pitchers': _parse_pitchers(home_box),
            'away_notes': away_notes,
            'home_notes': home_notes,
        }, None
    except Exception as exc:
        return {}, str(exc)