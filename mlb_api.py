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

    def get_boxscore(self, game_pk: int) -> dict[str, Any]:
        return self._get(f'/game/{game_pk}/boxscore')

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
# Live feed with plays (full detail)
# ---------------------------------------------------------------------------

def get_live_feed_full(game_pk: int | None) -> tuple[dict[str, Any], str | None]:
    """Return full live feed data including all plays for a game."""
    if not game_pk:
        return {}, None
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        return data, None
    except Exception as exc:
        return {}, str(exc)


def get_boxscore_data(game_pk: int | None) -> tuple[dict[str, Any], str | None]:
    """Return boxscore data for a game."""
    if not game_pk:
        return {}, None
    client = MLBClient()
    try:
        data = client.get_boxscore(game_pk)
        return data, None
    except Exception as exc:
        return {}, str(exc)


# ---------------------------------------------------------------------------
# Playoff seed generation from standings
# ---------------------------------------------------------------------------

def build_playoff_seeds(season: int) -> tuple[pd.DataFrame, str | None]:
    """Build a DataFrame of playoff seeds (1-6) for each league from current standings.

    Columns: league, seed, team_id, team_name, wins, losses, win_pct, logo_url, division
    Seeds:
      1-3: division leaders sorted by win_pct (1 = best record, bye)
      4-6: top 3 non-division-leaders by win_pct (wild cards)
    """
    _cols = ['league', 'seed', 'team_id', 'team_name', 'wins', 'losses', 'win_pct',
             'logo_url', 'division', 'run_diff_per_game']
    client = MLBClient()
    all_rows: list[dict[str, Any]] = []

    for league, league_id in [('AL', 103), ('NL', 104)]:
        try:
            data = client.get_standings({
                'leagueId': league_id,
                'standingsTypes': 'regularSeason',
                'season': season,
            })
        except Exception as exc:
            return pd.DataFrame(columns=_cols), str(exc)

        division_leaders: list[dict[str, Any]] = []
        non_leaders: list[dict[str, Any]] = []

        for record in data.get('records', []):
            div_info = record.get('division') or {}
            div_id = coerce_int(div_info.get('id'), 0)
            div_name = ''
            for dname, did in DIVISION_IDS.items():
                if did == div_id:
                    div_name = dname
                    break

            team_records = record.get('teamRecords', [])
            sorted_records = sorted(
                team_records,
                key=lambda tr: (coerce_int(tr.get('divisionRank'), 99),),
            )
            for i, tr in enumerate(sorted_records):
                team_info = tr.get('team') or {}
                tid = coerce_int(team_info.get('id'), 0)
                wins = coerce_int(tr.get('wins'), 0)
                losses = coerce_int(tr.get('losses'), 0)
                played = wins + losses
                wp = round(wins / played, 3) if played > 0 else 0.0
                meta = TEAM_META.get(tid, {})
                rd = tr.get('runDifferential', 0)
                rd_pg = round(coerce_int(rd, 0) / played, 2) if played > 0 else 0.0
                row = {
                    'league': league,
                    'team_id': tid,
                    'team_name': clean_text(team_info.get('name')),
                    'wins': wins,
                    'losses': losses,
                    'win_pct': wp,
                    'logo_url': meta.get('logo_url', ''),
                    'division': div_name,
                    'run_diff_per_game': rd_pg,
                }
                if i == 0:
                    division_leaders.append(row)
                else:
                    non_leaders.append(row)

        # Sort division leaders by win_pct desc (seeds 1-3)
        division_leaders.sort(key=lambda r: r['win_pct'], reverse=True)
        for seed_idx, row in enumerate(division_leaders[:3], start=1):
            row['seed'] = seed_idx
            all_rows.append(row)

        # Sort non-leaders by win_pct desc (seeds 4-6)
        non_leaders.sort(key=lambda r: r['win_pct'], reverse=True)
        for seed_idx, row in enumerate(non_leaders[:3], start=4):
            row['seed'] = seed_idx
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(columns=_cols), 'No standings data available to build playoff seeds.'
    return pd.DataFrame(all_rows)[_cols].reset_index(drop=True), None


def get_completed_game_pk(schedule_df: pd.DataFrame) -> int | None:
    """Return gamePk of the most recently completed game, or None."""
    if schedule_df.empty:
        return None
    finals = schedule_df[schedule_df['abstract_status'] == 'Final'].copy()
    if finals.empty:
        return None
    return coerce_int(finals.iloc[-1].get('gamePk'), 0) or None


# ---------------------------------------------------------------------------
# Live scorecard helpers
# ---------------------------------------------------------------------------

def get_game_status(game_pk: int | None) -> tuple[dict[str, Any], str | None]:
    """Return game status info: abstractGameState, detailedState, statusCode."""
    if not game_pk:
        return {}, None
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        gd = data.get('gameData', {})
        status = gd.get('status', {})
        return {
            'abstractGameState': status.get('abstractGameState', ''),
            'detailedState': status.get('detailedState', ''),
            'statusCode': status.get('statusCode', ''),
            'startTimeTBD': status.get('startTimeTBD', False),
        }, None
    except Exception as exc:
        return {}, str(exc)


def get_live_scorecard(game_pk: int | None) -> tuple[dict[str, Any], str | None]:
    """Return a structured scorecard dict parsed from the live feed linescore.

    Returns a dict with keys:
        gamePk, game_date, venue, status, detailed_status,
        away_team, away_id, away_abbr, away_logo, away_record,
        home_team, home_id, home_abbr, home_logo, home_record,
        innings (list of {inning, away_runs, home_runs}),
        away_runs, home_runs, away_hits, home_hits, away_errors, home_errors,
        current_inning, inning_state, inning_ordinal,
        balls, strikes, outs,
        runners (dict with first, second, third booleans),
        last_play, scoring_plays
    """
    if not game_pk:
        return {}, None
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
    except Exception as exc:
        return {}, str(exc)

    gd = data.get('gameData', {})
    ld = data.get('liveData', {})
    linescore = ld.get('linescore', {})
    status = gd.get('status', {})
    teams = gd.get('teams', {})
    venue_info = gd.get('venue', {})
    datetime_info = gd.get('datetime', {})

    away = teams.get('away', {})
    home = teams.get('home', {})
    away_rec = away.get('record', {})
    home_rec = home.get('record', {})
    away_id = coerce_int(away.get('id'), 0)
    home_id = coerce_int(home.get('id'), 0)
    away_meta = TEAM_META.get(away_id, {})
    home_meta = TEAM_META.get(home_id, {})

    # Parse innings from linescore
    innings_data = linescore.get('innings', [])
    innings = []
    for inn in innings_data:
        innings.append({
            'inning': coerce_int(inn.get('num'), 0),
            'away_runs': coerce_int(inn.get('away', {}).get('runs'), None),
            'home_runs': coerce_int(inn.get('home', {}).get('runs'), None),
        })

    ls_teams = linescore.get('teams', {})
    ls_away = ls_teams.get('away', {})
    ls_home = ls_teams.get('home', {})

    # Defense / offense for runners
    defense = linescore.get('defense', {})
    offense = linescore.get('offense', {})

    runners = {
        'first': bool(offense.get('first')),
        'second': bool(offense.get('second')),
        'third': bool(offense.get('third')),
    }

    # Last play
    plays = ld.get('plays', {})
    current_play = plays.get('currentPlay', {})
    last_play_desc = ''
    if current_play:
        last_play_desc = current_play.get('result', {}).get('description', '')

    # Scoring plays
    scoring_play_indices = plays.get('scoringPlays', [])
    all_plays = plays.get('allPlays', [])
    scoring_plays = []
    for idx in scoring_play_indices:
        if 0 <= idx < len(all_plays):
            sp = all_plays[idx]
            sp_about = sp.get('about', {})
            sp_result = sp.get('result', {})
            scoring_plays.append({
                'inning': coerce_int(sp_about.get('inning'), 0),
                'halfInning': sp_about.get('halfInning', ''),
                'description': sp_result.get('description', ''),
                'event': sp_result.get('event', ''),
                'awayScore': coerce_int(sp_result.get('awayScore'), 0),
                'homeScore': coerce_int(sp_result.get('homeScore'), 0),
            })

    abstract_state = status.get('abstractGameState', '')
    detailed_state = status.get('detailedState', '')

    # Extract game date with fallback
    raw_dt = datetime_info.get('dateTime', '')
    fallback_date = raw_dt[:10] if raw_dt else ''
    game_date = datetime_info.get('officialDate', fallback_date)

    scorecard = {
        'gamePk': game_pk,
        'game_date': game_date,
        'venue': venue_info.get('name', ''),
        'status': abstract_state,
        'detailed_status': detailed_state,
        'away_team': away.get('name', ''),
        'away_id': away_id,
        'away_abbr': away_meta.get('abbreviation', away.get('abbreviation', '')),
        'away_logo': away_meta.get('logo_url', ''),
        'away_record': f"{coerce_int(away_rec.get('wins'), 0)}-{coerce_int(away_rec.get('losses'), 0)}",
        'home_team': home.get('name', ''),
        'home_id': home_id,
        'home_abbr': home_meta.get('abbreviation', home.get('abbreviation', '')),
        'home_logo': home_meta.get('logo_url', ''),
        'home_record': f"{coerce_int(home_rec.get('wins'), 0)}-{coerce_int(home_rec.get('losses'), 0)}",
        'innings': innings,
        'away_runs': coerce_int(ls_away.get('runs'), 0),
        'home_runs': coerce_int(ls_home.get('runs'), 0),
        'away_hits': coerce_int(ls_away.get('hits'), 0),
        'home_hits': coerce_int(ls_home.get('hits'), 0),
        'away_errors': coerce_int(ls_away.get('errors'), 0),
        'home_errors': coerce_int(ls_home.get('errors'), 0),
        'current_inning': coerce_int(linescore.get('currentInning'), 0),
        'inning_state': linescore.get('inningState', ''),
        'inning_ordinal': linescore.get('currentInningOrdinal', ''),
        'balls': coerce_int(linescore.get('balls'), 0),
        'strikes': coerce_int(linescore.get('strikes'), 0),
        'outs': coerce_int(linescore.get('outs'), 0),
        'runners': runners,
        'last_play': last_play_desc,
        'scoring_plays': scoring_plays,
    }
    return scorecard, None


def get_game_plays(game_pk: int | None) -> tuple[list[dict[str, Any]], str | None]:
    """Return all plays from a game's live feed."""
    if not game_pk:
        return [], None
    client = MLBClient()
    try:
        data = client.get_live_feed(game_pk)
        all_plays = data.get('liveData', {}).get('plays', {}).get('allPlays', [])
        return all_plays, None
    except Exception as exc:
        return [], str(exc)


def get_current_team_game(team_id: int, target_date: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Find the current or most recent game for a team on a given date.

    Returns a dict with gamePk, status, abstract_status, away, home, etc.
    Preference order: Live > Pre-Game/Scheduled > Final (most recent).
    """
    from datetime import date as dt_date
    if target_date is None:
        target_date = dt_date.today().isoformat()
    try:
        schedule_df, err = build_schedule_df(team_id, target_date)
        if schedule_df.empty:
            return None, err
        # Prefer live games
        live_statuses = {'In Progress', 'Manager Challenge', 'Delayed Start', 'Delayed', 'Suspended'}
        live = schedule_df[schedule_df['status'].isin(live_statuses)]
        if not live.empty:
            row = live.iloc[0]
            return row.to_dict(), None
        # Then pre-game
        pre_statuses = {'Pre-Game', 'Warmup', 'Scheduled'}
        pre = schedule_df[schedule_df['status'].isin(pre_statuses)]
        if not pre.empty:
            row = pre.iloc[0]
            return row.to_dict(), None
        # Then final
        final = schedule_df[schedule_df['abstract_status'] == 'Final']
        if not final.empty:
            row = final.iloc[-1]
            return row.to_dict(), None
        # Fallback: first game
        row = schedule_df.iloc[0]
        return row.to_dict(), None
    except Exception as exc:
        return None, str(exc)