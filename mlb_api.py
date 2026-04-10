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


class MLBApiError(Exception):
    pass


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


FALLBACK_TEAMS = [
    {'id': 108, 'name': 'Los Angeles Angels', 'abbreviation': 'LAA', 'division': 'AL West'},
    {'id': 109, 'name': 'Arizona Diamondbacks', 'abbreviation': 'ARI', 'division': 'NL West'},
    {'id': 110, 'name': 'Baltimore Orioles', 'abbreviation': 'BAL', 'division': 'AL East'},
    {'id': 111, 'name': 'Boston Red Sox', 'abbreviation': 'BOS', 'division': 'AL East'},
    {'id': 112, 'name': 'Chicago Cubs', 'abbreviation': 'CHC', 'division': 'NL Central'},
    {'id': 113, 'name': 'Cincinnati Reds', 'abbreviation': 'CIN', 'division': 'NL Central'},
    {'id': 114, 'name': 'Cleveland Guardians', 'abbreviation': 'CLE', 'division': 'AL Central'},
    {'id': 115, 'name': 'Colorado Rockies', 'abbreviation': 'COL', 'division': 'NL West'},
    {'id': 116, 'name': 'Detroit Tigers', 'abbreviation': 'DET', 'division': 'AL Central'},
    {'id': 117, 'name': 'Houston Astros', 'abbreviation': 'HOU', 'division': 'AL West'},
    {'id': 118, 'name': 'Kansas City Royals', 'abbreviation': 'KC', 'division': 'AL Central'},
    {'id': 119, 'name': 'Los Angeles Dodgers', 'abbreviation': 'LAD', 'division': 'NL West'},
    {'id': 120, 'name': 'Washington Nationals', 'abbreviation': 'WSH', 'division': 'NL East'},
    {'id': 121, 'name': 'New York Mets', 'abbreviation': 'NYM', 'division': 'NL East'},
    {'id': 133, 'name': 'Athletics', 'abbreviation': 'ATH', 'division': 'AL West'},
    {'id': 134, 'name': 'Pittsburgh Pirates', 'abbreviation': 'PIT', 'division': 'NL Central'},
    {'id': 135, 'name': 'San Diego Padres', 'abbreviation': 'SD', 'division': 'NL West'},
    {'id': 136, 'name': 'Seattle Mariners', 'abbreviation': 'SEA', 'division': 'AL West'},
    {'id': 137, 'name': 'San Francisco Giants', 'abbreviation': 'SF', 'division': 'NL West'},
    {'id': 138, 'name': 'St. Louis Cardinals', 'abbreviation': 'STL', 'division': 'NL Central'},
    {'id': 139, 'name': 'Tampa Bay Rays', 'abbreviation': 'TB', 'division': 'AL East'},
    {'id': 140, 'name': 'Texas Rangers', 'abbreviation': 'TEX', 'division': 'AL West'},
    {'id': 141, 'name': 'Toronto Blue Jays', 'abbreviation': 'TOR', 'division': 'AL East'},
    {'id': 142, 'name': 'Minnesota Twins', 'abbreviation': 'MIN', 'division': 'AL Central'},
    {'id': 143, 'name': 'Philadelphia Phillies', 'abbreviation': 'PHI', 'division': 'NL East'},
    {'id': 144, 'name': 'Atlanta Braves', 'abbreviation': 'ATL', 'division': 'NL East'},
    {'id': 145, 'name': 'Chicago White Sox', 'abbreviation': 'CWS', 'division': 'AL Central'},
    {'id': 146, 'name': 'Miami Marlins', 'abbreviation': 'MIA', 'division': 'NL East'},
    {'id': 147, 'name': 'New York Yankees', 'abbreviation': 'NYY', 'division': 'AL East'},
    {'id': 158, 'name': 'Milwaukee Brewers', 'abbreviation': 'MIL', 'division': 'NL Central'},
]


def _games_to_df(games: list[dict[str, Any]]) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['gamePk', 'gameDate', 'officialDate', 'away', 'home', 'away_id', 'home_id', 'away_score', 'home_score', 'status', 'abstract_status'])
    return df.sort_values(['officialDate', 'gameDate', 'gamePk']).reset_index(drop=True)


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
        return pd.DataFrame(FALLBACK_TEAMS).sort_values('name').reset_index(drop=True), f'Live MLB teams call failed. Using built-in team list instead. {exc}'


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


def get_wildcard_standings(season: int) -> tuple[pd.DataFrame, str | None]:
    client = MLBClient()
    try:
        data = client.get_standings({
            'leagueId': '103,104',
            'standingsTypes': 'wildCard',
            'season': season,
        })
        rows = []
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
        for col in ['launch_speed', 'launch_angle', 'hit_distance_sc', 'release_speed', 'release_spin_rate', 'estimated_woba_using_speedangle', 'woba_value', 'hc_x', 'hc_y']:
            if col in out.columns:
                out[col] = out[col].apply(coerce_float)
        return out, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ---------------------------------------------------------------------------
# League-wide team stat helpers
# ---------------------------------------------------------------------------


def get_league_team_hitting_mlb(season: int) -> tuple[pd.DataFrame, str | None]:
    """All-team season hitting stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'sportId': 1, 'stats': 'season', 'group': 'hitting', 'season': season,
        })
        rows: list[dict[str, Any]] = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {})
                s = split.get('stat', {})
                g = max(coerce_int(s.get('gamesPlayed'), 1), 1)
                pa = max(coerce_int(s.get('plateAppearances'), 1), 1)
                runs = coerce_int(s.get('runs'), 0)
                bb = coerce_int(s.get('baseOnBalls'), 0)
                so = coerce_int(s.get('strikeOuts'), 0)
                rows.append({
                    'Team': clean_text(team.get('name')),
                    'G': g, 'PA': pa,
                    'R': runs, 'R/G': round(runs / g, 2),
                    'HR': coerce_int(s.get('homeRuns'), 0),
                    'SB': coerce_int(s.get('stolenBases'), 0),
                    'AVG': coerce_float(s.get('avg'), 0.0),
                    'OBP': coerce_float(s.get('obp'), 0.0),
                    'SLG': coerce_float(s.get('slg'), 0.0),
                    'OPS': coerce_float(s.get('ops'), 0.0),
                    'BB%': round(bb / pa * 100, 1),
                    'K%': round(so / pa * 100, 1),
                })
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_league_team_pitching_mlb(season: int) -> tuple[pd.DataFrame, str | None]:
    """All-team season pitching stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'sportId': 1, 'stats': 'season', 'group': 'pitching', 'season': season,
        })
        rows: list[dict[str, Any]] = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {})
                s = split.get('stat', {})
                rows.append({
                    'Team': clean_text(team.get('name')),
                    'ERA': coerce_float(s.get('era'), 0.0),
                    'WHIP': coerce_float(s.get('whip'), 0.0),
                    'IP': coerce_float(s.get('inningsPitched'), 0.0),
                    'SO': coerce_int(s.get('strikeOuts'), 0),
                    'BB': coerce_int(s.get('baseOnBalls'), 0),
                    'HR': coerce_int(s.get('homeRuns'), 0),
                    'H': coerce_int(s.get('hits'), 0),
                    'ER': coerce_int(s.get('earnedRuns'), 0),
                    'TBF': coerce_int(s.get('battersFaced'), 0),
                    'Opp AVG': coerce_float(s.get('avg'), 0.0),
                    'SV': coerce_int(s.get('saves'), 0),
                    'HLD': coerce_int(s.get('holds'), 0),
                    'BS': coerce_int(s.get('blownSaves'), 0),
                    'GS': coerce_int(s.get('gamesStarted'), 0),
                    'HBP': coerce_int(s.get('hitByPitch'), 0),
                })
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_league_team_fielding_mlb(season: int) -> tuple[pd.DataFrame, str | None]:
    """All-team season fielding stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'sportId': 1, 'stats': 'season', 'group': 'fielding', 'season': season,
        })
        rows: list[dict[str, Any]] = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {})
                s = split.get('stat', {})
                rows.append({
                    'Team': clean_text(team.get('name')),
                    'E': coerce_int(s.get('errors'), 0),
                    'Fielding %': coerce_float(s.get('fielding'), 0.0),
                    'DP': coerce_int(s.get('doublePlays'), 0),
                    'A': coerce_int(s.get('assists'), 0),
                    'PO': coerce_int(s.get('putOuts'), 0),
                    'TC': coerce_int(s.get('chances'), 0),
                })
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


# ---------------------------------------------------------------------------
# FanGraphs helpers (via pybaseball)
# ---------------------------------------------------------------------------

_FG_IMPORT_ERR = 'pybaseball is not installed; FanGraphs data unavailable.'


def _parse_fg_pct(val: Any) -> float:
    """Parse a FanGraphs percentage value (float or ``'8.5 %'`` string)."""
    if isinstance(val, (int, float)):
        return 0.0 if pd.isna(val) else round(float(val), 1)
    s = str(val).strip().rstrip('%').strip()
    try:
        return round(float(s), 1)
    except (ValueError, TypeError):
        return 0.0


def get_fg_team_batting(season: int) -> tuple[pd.DataFrame, str | None]:
    """FanGraphs team batting (wOBA, wRC+, BB%, K%, etc.)."""
    try:
        from pybaseball import team_batting
        df = team_batting(season)
        if df is None or df.empty:
            return pd.DataFrame(), 'FanGraphs returned no team batting data.'
        return df, None
    except ImportError:
        return pd.DataFrame(), _FG_IMPORT_ERR
    except Exception as exc:
        return pd.DataFrame(), f'FanGraphs team batting unavailable: {exc}'


def get_fg_team_pitching(season: int) -> tuple[pd.DataFrame, str | None]:
    """FanGraphs team pitching (FIP, K-BB%, etc.)."""
    try:
        from pybaseball import team_pitching
        df = team_pitching(season)
        if df is None or df.empty:
            return pd.DataFrame(), 'FanGraphs returned no team pitching data.'
        return df, None
    except ImportError:
        return pd.DataFrame(), _FG_IMPORT_ERR
    except Exception as exc:
        return pd.DataFrame(), f'FanGraphs team pitching unavailable: {exc}'


def get_fg_team_fielding(season: int) -> tuple[pd.DataFrame, str | None]:
    """FanGraphs team fielding (DRS, UZR, Def if available)."""
    try:
        from pybaseball import team_fielding
        df = team_fielding(season)
        if df is None or df.empty:
            return pd.DataFrame(), 'FanGraphs returned no team fielding data.'
        return df, None
    except ImportError:
        return pd.DataFrame(), _FG_IMPORT_ERR
    except Exception as exc:
        return pd.DataFrame(), f'FanGraphs team fielding unavailable: {exc}'


def get_fg_pitching_individual(season: int, qual: int = 0) -> tuple[pd.DataFrame, str | None]:
    """Individual pitcher stats from FanGraphs for starter/reliever split."""
    try:
        from pybaseball import pitching_stats
        df = pitching_stats(season, qual=qual)
        if df is None or df.empty:
            return pd.DataFrame(), 'FanGraphs returned no individual pitching data.'
        return df, None
    except ImportError:
        return pd.DataFrame(), _FG_IMPORT_ERR
    except Exception as exc:
        return pd.DataFrame(), f'FanGraphs individual pitching unavailable: {exc}'


def get_fg_batting_individual(season: int, qual: int = 0) -> tuple[pd.DataFrame, str | None]:
    """Individual batter stats from FanGraphs for WAR leaderboard."""
    try:
        from pybaseball import batting_stats
        df = batting_stats(season, qual=qual)
        if df is None or df.empty:
            return pd.DataFrame(), 'FanGraphs returned no individual batting data.'
        return df, None
    except ImportError:
        return pd.DataFrame(), _FG_IMPORT_ERR
    except Exception as exc:
        return pd.DataFrame(), f'FanGraphs individual batting unavailable: {exc}'


# ---------------------------------------------------------------------------
# Division standings
# ---------------------------------------------------------------------------

def get_division_standings(season: int) -> tuple[dict, str | None]:
    """Fetch division standings from the MLB Stats API.

    Returns the raw JSON response dict and an optional error string.
    """
    client = MLBClient()
    try:
        data = client.get_standings({
            'leagueId': '103,104',
            'standingsTypes': 'regularSeason',
            'season': season,
            'hydrate': 'team',
        })
        return data, None
    except Exception as exc:
        return {}, str(exc)