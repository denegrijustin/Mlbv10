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
# League-wide ranking helpers
# ---------------------------------------------------------------------------

def get_league_hitting_stats(season: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch MLB-wide team hitting stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'stats': 'season',
            'group': 'hitting',
            'season': str(season),
            'sportId': 1,
        })
        rows = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {}) or {}
                stat = split.get('stat', {}) or {}
                rows.append({
                    'team_id': coerce_int(team.get('id'), 0),
                    'team_name': clean_text(team.get('name')),
                    'gamesPlayed': coerce_int(stat.get('gamesPlayed'), 0),
                    'runs': coerce_int(stat.get('runs'), 0),
                    'hits': coerce_int(stat.get('hits'), 0),
                    'homeRuns': coerce_int(stat.get('homeRuns'), 0),
                    'stolenBases': coerce_int(stat.get('stolenBases'), 0),
                    'baseOnBalls': coerce_int(stat.get('baseOnBalls'), 0),
                    'strikeOuts': coerce_int(stat.get('strikeOuts'), 0),
                    'plateAppearances': coerce_int(stat.get('plateAppearances'), 0),
                    'atBats': coerce_int(stat.get('atBats'), 0),
                    'avg': coerce_float(stat.get('avg'), 0.0),
                    'obp': coerce_float(stat.get('obp'), 0.0),
                    'slg': coerce_float(stat.get('slg'), 0.0),
                    'ops': coerce_float(stat.get('ops'), 0.0),
                    'rbi': coerce_int(stat.get('rbi'), 0),
                    'doubles': coerce_int(stat.get('doubles'), 0),
                    'triples': coerce_int(stat.get('triples'), 0),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, f'No hitting stats returned for {season}.'
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_league_pitching_stats(season: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch MLB-wide team pitching stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'stats': 'season',
            'group': 'pitching',
            'season': str(season),
            'sportId': 1,
        })
        rows = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {}) or {}
                stat = split.get('stat', {}) or {}
                rows.append({
                    'team_id': coerce_int(team.get('id'), 0),
                    'team_name': clean_text(team.get('name')),
                    'gamesPlayed': coerce_int(stat.get('gamesPlayed'), 0),
                    'era': coerce_float(stat.get('era'), 0.0),
                    'whip': coerce_float(stat.get('whip'), 0.0),
                    'strikeOuts': coerce_int(stat.get('strikeOuts'), 0),
                    'baseOnBalls': coerce_int(stat.get('baseOnBalls'), 0),
                    'homeRunsPer9': coerce_float(stat.get('homeRunsPer9'), 0.0),
                    'qualityStarts': coerce_int(stat.get('qualityStarts'), 0),
                    'saves': coerce_int(stat.get('saves'), 0),
                    'holds': coerce_int(stat.get('holds'), 0),
                    'blownSaves': coerce_int(stat.get('blownSaves'), 0),
                    'inningsPitched': coerce_float(stat.get('inningsPitched'), 0.0),
                    'hits': coerce_int(stat.get('hits'), 0),
                    'earnedRuns': coerce_int(stat.get('earnedRuns'), 0),
                    'saveOpportunities': coerce_int(stat.get('saveOpportunities'), 0),
                    'battersFaced': coerce_int(stat.get('battersFaced'), 0),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, f'No pitching stats returned for {season}.'
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_league_fielding_stats(season: int) -> tuple[pd.DataFrame, str | None]:
    """Fetch MLB-wide team fielding stats from the MLB Stats API."""
    client = MLBClient()
    try:
        data = client._get('/teams/stats', {
            'stats': 'season',
            'group': 'fielding',
            'season': str(season),
            'sportId': 1,
        })
        rows = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                team = split.get('team', {}) or {}
                stat = split.get('stat', {}) or {}
                rows.append({
                    'team_id': coerce_int(team.get('id'), 0),
                    'team_name': clean_text(team.get('name')),
                    'gamesPlayed': coerce_int(stat.get('gamesPlayed'), 0),
                    'errors': coerce_int(stat.get('errors'), 0),
                    'fieldingPercentage': coerce_float(stat.get('fieldingPercentage'), 0.0),
                    'doublePlays': coerce_int(stat.get('doublePlays'), 0),
                    'assists': coerce_int(stat.get('assists'), 0),
                    'putOuts': coerce_int(stat.get('putOuts'), 0),
                    'chances': coerce_int(stat.get('chances'), 0),
                    'rangeFactorPerGame': coerce_float(stat.get('rangeFactorPerGame'), 0.0),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, f'No fielding stats returned for {season}.'
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_individual_pitcher_stats(season: int) -> tuple[pd.DataFrame, str | None]:
    """Get individual pitcher stats for SP/RP classification and team-level aggregation."""
    client = MLBClient()
    try:
        data = client._get('/stats', {
            'stats': 'season',
            'group': 'pitching',
            'season': str(season),
            'sportId': 1,
            'playerPool': 'all',
            'limit': 2000,
        })
        rows = []
        for stat_block in data.get('stats', []):
            for split in stat_block.get('splits', []):
                player = split.get('player', {}) or {}
                team = split.get('team', {}) or {}
                stat = split.get('stat', {}) or {}
                games_played = coerce_int(stat.get('gamesPlayed'), 0)
                games_started = coerce_int(stat.get('gamesStarted'), 0)
                if games_played == 0:
                    continue
                ip = coerce_float(stat.get('inningsPitched'), 0.0)
                bfp = coerce_int(stat.get('battersFaced'), 0)
                hr = coerce_int(stat.get('homeRuns'), 0)
                # Classify role: starter if ≥50% of appearances are starts
                role = 'SP' if (games_started / games_played) >= 0.5 else 'RP'
                rows.append({
                    'player_id': coerce_int(player.get('id'), 0),
                    'player_name': clean_text(player.get('fullName')),
                    'team_id': coerce_int(team.get('id'), 0),
                    'team_name': clean_text(team.get('name')),
                    'games_played': games_played,
                    'games_started': games_started,
                    'role': role,
                    'innings_pitched': ip,
                    'strikeouts': coerce_int(stat.get('strikeOuts'), 0),
                    'walks': coerce_int(stat.get('baseOnBalls'), 0),
                    'hits_allowed': coerce_int(stat.get('hits'), 0),
                    'earned_runs': coerce_int(stat.get('earnedRuns'), 0),
                    'home_runs': hr,
                    'quality_starts': coerce_int(stat.get('qualityStarts'), 0),
                    'saves': coerce_int(stat.get('saves'), 0),
                    'holds': coerce_int(stat.get('holds'), 0),
                    'blown_saves': coerce_int(stat.get('blownSaves'), 0),
                    'save_opportunities': coerce_int(stat.get('saveOpportunities'), 0),
                    'batters_faced': bfp,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, f'No individual pitcher data returned for {season}.'
        return df, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_war_leaderboard_data(season: int) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Get WAR leaderboard data from FanGraphs via pybaseball."""
    try:
        from pybaseball import batting_stats, pitching_stats  # type: ignore
        bat = batting_stats(season, qual=0)
        pit = pitching_stats(season, qual=0)
        bat_df = bat if (bat is not None and not bat.empty) else pd.DataFrame()
        pit_df = pit if (pit is not None and not pit.empty) else pd.DataFrame()
        return bat_df, pit_df, None
    except ImportError:
        return pd.DataFrame(), pd.DataFrame(), 'pybaseball not installed; WAR unavailable.'
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f'WAR data unavailable: {exc}'