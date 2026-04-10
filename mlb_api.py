from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any
from datetime import date

import pandas as pd
import requests

from formatting import clean_text, coerce_float, coerce_int

BASE_URL = 'https://statsapi.mlb.com/api/v1'
STATCAST_URL = 'https://baseballsavant.mlb.com/statcast_search/csv'
TIMEOUT = 30
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; MLBDashboard/1.0)',
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

    def get_team_stats(self, season: int, group: str) -> list[dict[str, Any]]:
        """Get season stats for all teams in a stat group (hitting/pitching/fielding)."""
        data = self._get('/teams/stats', {
            'stats': 'season',
            'group': group,
            'season': season,
            'sportId': 1,
            'gameType': 'R',
        })
        splits: list[dict[str, Any]] = []
        for stat_block in data.get('stats', []):
            splits.extend(stat_block.get('splits', []))
        return splits


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


def build_season_linescore_df(team_id: int, season: int, end_date: str) -> tuple[pd.DataFrame, str | None]:
    """Fetch full-season schedule with linescore hydration for inning-by-inning runs."""
    client = MLBClient()
    try:
        games = client.get_schedule({
            'sportId': 1,
            'teamId': team_id,
            'startDate': f'{season}-01-01',
            'endDate': end_date,
            'gameType': 'R',
            'hydrate': 'linescore',
        })
        rows: list[dict[str, Any]] = []
        for g in games:
            status_obj = g.get('status') or {}
            abstract_state = status_obj.get('abstractGameState', '')
            if abstract_state != 'Final':
                continue
            game_pk = coerce_int(g.get('gamePk'), 0)
            official_date = clean_text(g.get('officialDate'), '')
            teams = g.get('teams', {})
            home_id = coerce_int(((teams.get('home') or {}).get('team') or {}).get('id'), 0)
            is_home = (home_id == team_id)
            location = 'Home' if is_home else 'Away'
            linescore = g.get('linescore') or {}
            innings = linescore.get('innings') or []
            for inning_obj in innings:
                inn_num = coerce_int(inning_obj.get('num'), 0)
                home_runs = coerce_int((inning_obj.get('home') or {}).get('runs'), 0)
                away_runs = coerce_int((inning_obj.get('away') or {}).get('runs'), 0)
                team_runs = home_runs if is_home else away_runs
                opp_runs = away_runs if is_home else home_runs
                rows.append({
                    'gamePk': game_pk,
                    'date': official_date,
                    'inning': inn_num,
                    'team_runs': team_runs,
                    'opp_runs': opp_runs,
                    'location': location,
                })
        if not rows:
            return pd.DataFrame(columns=['gamePk', 'date', 'inning', 'team_runs', 'opp_runs', 'location']), None
        return pd.DataFrame(rows), None
    except Exception as exc:
        return pd.DataFrame(columns=['gamePk', 'date', 'inning', 'team_runs', 'opp_runs', 'location']), str(exc)


def get_division_standings(season: int) -> tuple[pd.DataFrame, str | None]:
    """Get current division standings for all divisions."""
    client = MLBClient()
    try:
        data = client.get_standings({
            'leagueId': '103,104',
            'standingsTypes': 'regularSeason',
            'season': season,
            'hydrate': 'team,division',
        })
        rows = []
        for div_record in data.get('records', []):
            div_info = div_record.get('division') or {}
            div_name = clean_text(div_info.get('name'), 'Unknown Division')
            for tr in div_record.get('teamRecords', []):
                team_info = tr.get('team') or {}
                rows.append({
                    'division': div_name,
                    'team_id': coerce_int(team_info.get('id'), 0),
                    'team_name': clean_text(team_info.get('name')),
                    'wins': coerce_int(tr.get('wins'), 0),
                    'losses': coerce_int(tr.get('losses'), 0),
                    'pct': clean_text(tr.get('winningPercentage'), '.000'),
                    'gb': clean_text(tr.get('gamesBack'), '-'),
                    'streak': clean_text((tr.get('streak') or {}).get('streakCode'), '-'),
                    'last10': clean_text((tr.get('records') or {}).get('splitRecords', [{}])[0].get('wins', ''), '') +
                              '-' + clean_text((tr.get('records') or {}).get('splitRecords', [{}])[0].get('losses', ''), ''),
                    'runs_scored': coerce_int((tr.get('runsScored')), 0),
                    'runs_allowed': coerce_int((tr.get('runsAllowed')), 0),
                    'div_rank': coerce_int(tr.get('divisionRank'), 0),
                    'wc_rank': clean_text(tr.get('wildCardRank'), '-'),
                    'elimination_number': clean_text(tr.get('eliminationNumber'), '-'),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df, None
        return df.sort_values(['division', 'div_rank']).reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


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
                    'pct': clean_text(team_record.get('winningPercentage'), '.000'),
                    'gb': clean_text(team_record.get('gamesBack'), '-'),
                    'wc_gb': clean_text(team_record.get('wildCardGamesBack'), '-'),
                })
        df = pd.DataFrame(rows)
        return df, None
    except Exception as exc:
        return pd.DataFrame(columns=['team_id', 'team_name', 'wildcard_rank', 'wins', 'losses']), str(exc)


def get_all_teams_stats(season: int) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Fetch hitting, pitching, and fielding stats for all teams in one call each."""
    client = MLBClient()
    results: dict[str, pd.DataFrame] = {}
    errors = []

    for group in ('hitting', 'pitching', 'fielding'):
        try:
            splits = client.get_team_stats(season, group)
            rows = []
            for split in splits:
                team_obj = split.get('team') or {}
                stat = split.get('stat') or {}
                row = {
                    'team_id': coerce_int(team_obj.get('id'), 0),
                    'team_name': clean_text(team_obj.get('name')),
                }
                row.update({k: v for k, v in stat.items()})
                rows.append(row)
            results[group] = pd.DataFrame(rows)
        except Exception as exc:
            results[group] = pd.DataFrame()
            errors.append(f'{group}: {exc}')

    err = '; '.join(errors) if errors else None
    return results, err


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
            # Extract current play info
            plays = live_data.get('plays', {})
            current_play = plays.get('currentPlay', {})
            matchup = current_play.get('matchup', {})
            batter = (matchup.get('batter') or {}).get('fullName', '-')
            pitcher = (matchup.get('pitcher') or {}).get('fullName', '-')
            # Recent plays for play-by-play
            all_plays = plays.get('allPlays', [])
            recent = []
            for p in reversed(all_plays[-10:]):
                result = p.get('result', {})
                about = p.get('about', {})
                recent.append({
                    'inning': about.get('inning', '-'),
                    'half': about.get('halfInning', '-'),
                    'description': result.get('description', '-'),
                    'event': result.get('event', '-'),
                })
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
                'batter': batter,
                'pitcher': pitcher,
                'recent_plays': recent,
            }, None
        except Exception as exc:
            if attempt == max_retries - 1:
                return {}, str(exc)
    return {}, 'Live feed unavailable'


def get_war_leaders(season: int, n: int = 50) -> tuple[pd.DataFrame, str | None]:
    """Get WAR leaders. Tries FanGraphs (pybaseball), falls back to Baseball-Reference."""
    # Method 1: pybaseball / FanGraphs
    try:
        from pybaseball import batting_stats, pitching_stats
        bat = batting_stats(season, qual=80)
        pit = pitching_stats(season, qual=20)
        bat_cols = {c for c in ['Name', 'Team', 'G', 'PA', 'HR', 'AVG', 'OBP', 'SLG', 'WAR'] if c in bat.columns}
        pit_cols = {c for c in ['Name', 'Team', 'G', 'IP', 'ERA', 'WHIP', 'SO9', 'WAR'] if c in pit.columns}
        bat_df = bat[list(bat_cols)].copy()
        bat_df['Type'] = 'Batting'
        pit_df = pit[list(pit_cols)].copy()
        pit_df['Type'] = 'Pitching'
        combined = pd.concat([bat_df, pit_df], ignore_index=True, sort=False)
        combined['WAR'] = pd.to_numeric(combined['WAR'], errors='coerce')
        combined = combined.dropna(subset=['WAR']).sort_values('WAR', ascending=False).head(n)
        combined['Source'] = 'FanGraphs'
        return combined.reset_index(drop=True), None
    except Exception as exc_fg:
        pass

    # Method 2: Baseball-Reference HTML (batting WAR)
    try:
        br_url = f'https://www.baseball-reference.com/leagues/majors/{season}-value-batting.shtml'
        tables = pd.read_html(br_url, flavor='lxml', header=0)
        for tbl in tables:
            if 'WAR' in tbl.columns and 'Name' in tbl.columns:
                df = tbl.copy()
                # Remove header rows repeated in the table
                df = df[df['Name'].astype(str).str.strip() != 'Name']
                df = df[df['Name'].astype(str).str.strip().ne('')]
                df = df.dropna(subset=['WAR'])
                df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce')
                df = df.dropna(subset=['WAR']).sort_values('WAR', ascending=False).head(n)
                keep = [c for c in ['Name', 'Tm', 'G', 'PA', 'HR', 'BA', 'OBP', 'SLG', 'WAR'] if c in df.columns]
                df = df[keep].rename(columns={'Name': 'Player', 'Tm': 'Team', 'BA': 'AVG'})
                df['Type'] = 'Batting'
                df['Source'] = 'Baseball-Reference'
                return df.reset_index(drop=True), 'FanGraphs unavailable. Showing Baseball-Reference batting WAR.'
    except Exception as exc_br:
        pass

    # Method 3: BR pitching WAR fallback
    try:
        br_pit_url = f'https://www.baseball-reference.com/leagues/majors/{season}-value-pitching.shtml'
        tables = pd.read_html(br_pit_url, flavor='lxml', header=0)
        for tbl in tables:
            if 'WAR' in tbl.columns and 'Name' in tbl.columns:
                df = tbl.copy()
                df = df[df['Name'].astype(str).str.strip() != 'Name']
                df = df.dropna(subset=['WAR'])
                df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce')
                df = df.dropna(subset=['WAR']).sort_values('WAR', ascending=False).head(n)
                keep = [c for c in ['Name', 'Tm', 'G', 'IP', 'ERA', 'WHIP', 'WAR'] if c in df.columns]
                df = df[keep].rename(columns={'Name': 'Player', 'Tm': 'Team'})
                df['Type'] = 'Pitching'
                df['Source'] = 'Baseball-Reference'
                return df.reset_index(drop=True), 'FanGraphs unavailable. Showing Baseball-Reference pitching WAR.'
    except Exception:
        pass

    return pd.DataFrame(columns=['Player', 'Team', 'WAR', 'Type', 'Source']), (
        f'WAR data temporarily unavailable. FanGraphs returned an error. '
        f'Baseball-Reference page could not be parsed. Try refreshing.'
    )


def get_statcast_team_df(
    team_abbr: str,
    start_date: str,
    end_date: str,
    player_type: str = 'batter',
) -> tuple[pd.DataFrame, str | None]:
    """Fetch Statcast pitch-level data for a team over a date range."""
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
        for col in [
            'launch_speed', 'launch_angle', 'hit_distance_sc',
            'release_speed', 'release_spin_rate',
            'estimated_woba_using_speedangle', 'woba_value',
            'hc_x', 'hc_y',
        ]:
            if col in out.columns:
                out[col] = out[col].apply(coerce_float)
        return out, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def get_statcast_season_df(
    team_abbr: str,
    season: int,
    player_type: str = 'batter',
) -> tuple[pd.DataFrame, str | None]:
    """Fetch full-season Statcast data for spray charts and HR analysis."""
    today = date.today().isoformat()
    start = f'{season}-03-01'
    return get_statcast_team_df(team_abbr, start, today, player_type)


def get_team_roster_names(team_id: int, season: int) -> tuple[list[str], str | None]:
    """Return sorted list of active batter names for the team."""
    client = MLBClient()
    try:
        data = client._get('/roster', {
            'teamId': team_id,
            'season': season,
            'rosterType': 'active',
        })
        names = []
        for entry in data.get('roster', []):
            person = entry.get('person') or {}
            position = entry.get('position') or {}
            pos_type = position.get('type', '')
            if pos_type != 'Pitcher':
                full_name = clean_text(person.get('fullName'), '')
                if full_name and full_name != '-':
                    names.append(full_name)
        return sorted(names), None
    except Exception as exc:
        return [], str(exc)
