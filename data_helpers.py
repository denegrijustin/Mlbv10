from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from formatting import clean_text, coerce_float, coerce_int, format_record, safe_pct, signed, stoplight

SWING_DESCRIPTIONS = {
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score', 'missed_bunt',
}
WHIFF_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'foul_tip'}
HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
OUT_EVENTS = {
    'field_out', 'force_out', 'double_play', 'fielders_choice_out', 'grounded_into_double_play',
    'sac_fly', 'sac_bunt', 'fielders_choice', 'triple_play', 'sac_fly_double_play',
}

# Statcast hc_x / hc_y → standard field coords (home plate centered, RF right)
_HC_X_HOME = 125.42
_HC_Y_HOME = 198.27


def safe_team_row(teams_df: pd.DataFrame, selected_team: str) -> dict[str, Any] | None:
    if teams_df.empty or not selected_team:
        return None
    match = teams_df.loc[teams_df['name'] == selected_team]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _team_games(season_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if season_df.empty:
        return season_df.copy()
    df = season_df[(season_df['away'] == team_name) | (season_df['home'] == team_name)].copy()
    if df.empty:
        return df
    df['location'] = df.apply(lambda r: 'Away' if r['away'] == team_name else 'Home', axis=1)
    df['opponent'] = df.apply(lambda r: r['home'] if r['away'] == team_name else r['away'], axis=1)
    df['team_runs'] = df.apply(
        lambda r: coerce_int(r['away_score'], 0) if r['away'] == team_name else coerce_int(r['home_score'], 0),
        axis=1,
    )
    df['opp_runs'] = df.apply(
        lambda r: coerce_int(r['home_score'], 0) if r['away'] == team_name else coerce_int(r['away_score'], 0),
        axis=1,
    )
    df['run_diff'] = df['team_runs'] - df['opp_runs']
    df['is_final'] = df['status'].astype(str).str.contains('Final', case=False, na=False)
    df['result'] = df.apply(
        lambda r: 'W' if r['team_runs'] > r['opp_runs'] else ('L' if r['team_runs'] < r['opp_runs'] else 'T'),
        axis=1,
    )
    return df.reset_index(drop=True)


def build_team_snapshot(
    team_row: dict[str, Any] | None,
    season_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> dict[str, Any]:
    team_name = (team_row or {}).get('name', '-')
    division = (team_row or {}).get('division', '-')
    team_id = coerce_int((team_row or {}).get('id'), 0)

    games = _team_games(season_df, team_name)
    finals = games[games['is_final']].copy() if not games.empty else games
    wins = int((finals['result'] == 'W').sum()) if not finals.empty else 0
    losses = int((finals['result'] == 'L').sum()) if not finals.empty else 0
    runs_for = int(finals['team_runs'].sum()) if not finals.empty else 0
    runs_against = int(finals['opp_runs'].sum()) if not finals.empty else 0
    games_played = len(finals)

    games_today = len(daily_df) if not daily_df.empty else 0
    today_status = (
        ', '.join([s for s in daily_df['status'].astype(str).tolist() if s])
        if not daily_df.empty
        else 'No game today'
    )

    return {
        'team_id': team_id,
        'team': team_name,
        'division': division,
        'games_played': games_played,
        'games_today': games_today,
        'today_status': today_status,
        'runs_for': runs_for,
        'runs_against': runs_against,
        'run_diff': runs_for - runs_against,
        'record': format_record(wins, losses),
        'win_pct': safe_pct(wins, games_played, 1) if games_played else 0.0,
        'avg_runs_for': round(runs_for / games_played, 2) if games_played else 0.0,
        'avg_runs_against': round(runs_against / games_played, 2) if games_played else 0.0,
    }


def build_summary_df(snapshot: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        'Team': snapshot.get('team', '-'),
        'Division': snapshot.get('division', '-'),
        'Record': snapshot.get('record', '0-0'),
        'Win %': snapshot.get('win_pct', 0.0),
        'Games Played': snapshot.get('games_played', 0),
        'Season Avg Runs For': snapshot.get('avg_runs_for', 0.0),
        'Season Avg Runs Against': snapshot.get('avg_runs_against', 0.0),
        'Season Run Differential': snapshot.get('run_diff', 0),
        'Games Today': snapshot.get('games_today', 0),
        'Today Status': snapshot.get('today_status', '-'),
    }])


def build_trend_df(season_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    games = _team_games(season_df, team_name)
    if games.empty:
        return pd.DataFrame(columns=['Metric', 'Value', 'Trend'])
    finals = games[games['is_final']].copy()
    if finals.empty:
        return pd.DataFrame(columns=['Metric', 'Value', 'Trend'])

    last_10 = finals.tail(10)
    last_5 = finals.tail(5)
    prev_5 = finals.iloc[-10:-5] if len(finals) >= 10 else finals.head(0)

    season_rf = finals['team_runs'].mean()
    season_ra = finals['opp_runs'].mean()
    last5_rf = last_5['team_runs'].mean() if not last_5.empty else 0.0
    last5_ra = last_5['opp_runs'].mean() if not last_5.empty else 0.0
    prev5_rf = prev_5['team_runs'].mean() if not prev_5.empty else last5_rf
    prev5_ra = prev_5['opp_runs'].mean() if not prev_5.empty else last5_ra

    last10_wins = int((last_10['result'] == 'W').sum())
    consistency = round(last_10['team_runs'].std(ddof=0), 2) if len(last_10) > 1 else 0.0
    home_split = finals[finals['location'] == 'Home']
    away_split = finals[finals['location'] == 'Away']

    rows = [
        {'Metric': 'Season Avg Runs For', 'Value': round(season_rf, 2), 'Trend': '🟡 Baseline'},
        {'Metric': 'Season Avg Runs Against', 'Value': round(season_ra, 2), 'Trend': '🟡 Baseline'},
        {'Metric': 'Last 5 Avg Runs For', 'Value': round(last5_rf, 2), 'Trend': stoplight(last5_rf - prev5_rf)},
        {'Metric': 'Last 5 Avg Runs Against', 'Value': round(last5_ra, 2), 'Trend': stoplight(prev5_ra - last5_ra)},
        {
            'Metric': 'Last 10 Record',
            'Value': format_record(last10_wins, len(last_10) - last10_wins),
            'Trend': stoplight((last10_wins / max(len(last_10), 1)) - 0.5),
        },
        {'Metric': 'Last 10 Run Diff / Game', 'Value': round(last_10['run_diff'].mean(), 2), 'Trend': stoplight(last_10['run_diff'].mean())},
        {'Metric': 'Scoring Consistency Std Dev', 'Value': consistency, 'Trend': stoplight(2.5 - consistency)},
        {
            'Metric': 'Home Avg Runs',
            'Value': round(home_split['team_runs'].mean(), 2) if not home_split.empty else 0.0,
            'Trend': '🟡 Split',
        },
        {
            'Metric': 'Away Avg Runs',
            'Value': round(away_split['team_runs'].mean(), 2) if not away_split.empty else 0.0,
            'Trend': '🟡 Split',
        },
    ]
    return pd.DataFrame(rows)


def build_recent_games_df(season_df: pd.DataFrame, team_name: str, count: int = 10) -> pd.DataFrame:
    games = _team_games(season_df, team_name)
    if games.empty:
        return pd.DataFrame(columns=['Date', 'Opponent', 'Location', 'Result', 'Team Runs', 'Opp Runs', 'Run Diff'])
    finals = games[games['is_final']].tail(count).copy()
    if finals.empty:
        return pd.DataFrame(columns=['Date', 'Opponent', 'Location', 'Result', 'Team Runs', 'Opp Runs', 'Run Diff'])
    finals['Date'] = finals['officialDate']
    finals['Opponent'] = finals['opponent']
    finals['Location'] = finals['location']
    finals['Result'] = finals['result']
    finals['Team Runs'] = finals['team_runs']
    finals['Opp Runs'] = finals['opp_runs']
    finals['Run Diff'] = finals['run_diff']
    return finals[['Date', 'Opponent', 'Location', 'Result', 'Team Runs', 'Opp Runs', 'Run Diff']].reset_index(drop=True)


def build_schedule_table(schedule_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=['Game Date', 'Matchup', 'Status', 'Score'])
    out = schedule_df.copy()
    out['Matchup'] = out['away'] + ' at ' + out['home']
    out['Game Date'] = out['gameDate'].astype(str).str[:19].str.replace('T', ' ', regex=False)
    out['Score'] = out['away_score'].astype(str) + '-' + out['home_score'].astype(str)
    out['Team Side'] = out.apply(lambda r: 'Away' if r['away'] == team_name else 'Home', axis=1)
    out['Status'] = out['status']
    return out[['Game Date', 'Matchup', 'Team Side', 'Status', 'Score']].reset_index(drop=True)


def build_live_box_df(live_summary: dict[str, Any]) -> pd.DataFrame:
    if not live_summary:
        return pd.DataFrame(columns=['Field', 'Value'])
    rows = [
        {'Field': 'Matchup', 'Value': f"{live_summary.get('away_team', '-')} at {live_summary.get('home_team', '-')}"},
        {'Field': 'Status', 'Value': live_summary.get('status', '-')},
        {'Field': 'Inning', 'Value': f"{live_summary.get('inning_state', '-')} {live_summary.get('inning', '-')}"},
        {'Field': 'Score', 'Value': f"{live_summary.get('away_runs', 0)}-{live_summary.get('home_runs', 0)}"},
        {'Field': 'Count', 'Value': f"{live_summary.get('balls', 0)} balls, {live_summary.get('strikes', 0)} strikes, {live_summary.get('outs', 0)} outs"},
        {'Field': 'Batter', 'Value': live_summary.get('batter', '-')},
        {'Field': 'Pitcher', 'Value': live_summary.get('pitcher', '-')},
    ]
    return pd.DataFrame(rows)


def build_kpi_cards(snapshot: dict[str, Any], trend_df: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = {row['Metric']: row for _, row in trend_df.iterrows()} if not trend_df.empty else {}
    return [
        {'label': 'Record', 'value': snapshot.get('record', '0-0'), 'delta': f"{snapshot.get('win_pct', 0.0):.1f}% win pct"},
        {'label': 'Season Avg RF', 'value': snapshot.get('avg_runs_for', 0.0), 'delta': lookup.get('Last 5 Avg Runs For', {}).get('Trend', '🟡 Even')},
        {'label': 'Season Avg RA', 'value': snapshot.get('avg_runs_against', 0.0), 'delta': lookup.get('Last 5 Avg Runs Against', {}).get('Trend', '🟡 Even')},
        {'label': 'Run Diff', 'value': snapshot.get('run_diff', 0), 'delta': signed(coerce_float(snapshot.get('avg_runs_for', 0.0)) - coerce_float(snapshot.get('avg_runs_against', 0.0)), 2)},
    ]


def build_team_rolling_df(recent_games_df: pd.DataFrame) -> pd.DataFrame:
    if recent_games_df.empty:
        return pd.DataFrame(columns=['Date', 'Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'])
    df = recent_games_df.copy()
    df['Runs 3'] = df['Team Runs'].rolling(3, min_periods=1).mean().round(2)
    df['Runs 5'] = df['Team Runs'].rolling(5, min_periods=1).mean().round(2)
    df['Diff 3'] = df['Run Diff'].rolling(3, min_periods=1).mean().round(2)
    df['Diff 5'] = df['Run Diff'].rolling(5, min_periods=1).mean().round(2)
    return df[['Date', 'Runs 3', 'Runs 5', 'Diff 3', 'Diff 5']]


# ─── Runs by inning ───────────────────────────────────────────────────────────

def build_runs_by_inning(
    linescore_df: pd.DataFrame,
    location_filter: str = 'All',
    last_n: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate runs scored / allowed per inning across the season.

    Parameters
    ----------
    linescore_df : DataFrame with columns [gamePk, date, inning, team_runs, opp_runs, location]
    location_filter : 'All', 'Home', or 'Away'
    last_n : if set, use only the last N games by date

    Returns
    -------
    DataFrame with columns [Inning, Runs Scored, Runs Allowed, Differential, Games]
    """
    if linescore_df.empty:
        return pd.DataFrame(columns=['Inning', 'Runs Scored', 'Runs Allowed', 'Differential', 'Games'])

    df = linescore_df.copy()

    # Location filter
    if location_filter in ('Home', 'Away'):
        df = df[df['location'] == location_filter]

    if df.empty:
        return pd.DataFrame(columns=['Inning', 'Runs Scored', 'Runs Allowed', 'Differential', 'Games'])

    # Last N games
    if last_n is not None:
        game_dates = df[['gamePk', 'date']].drop_duplicates().sort_values('date')
        recent_pks = game_dates.tail(last_n)['gamePk'].tolist()
        df = df[df['gamePk'].isin(recent_pks)]

    if df.empty:
        return pd.DataFrame(columns=['Inning', 'Runs Scored', 'Runs Allowed', 'Differential', 'Games'])

    agg = (
        df.groupby('inning', as_index=False)
        .agg(
            **{
                'Runs Scored': ('team_runs', 'sum'),
                'Runs Allowed': ('opp_runs', 'sum'),
                'Games': ('gamePk', 'nunique'),
            }
        )
        .rename(columns={'inning': 'Inning'})
    )
    agg['Differential'] = agg['Runs Scored'] - agg['Runs Allowed']
    agg = agg.sort_values('Inning').reset_index(drop=True)
    return agg


# ─── MLB-wide and team rankings ───────────────────────────────────────────────

def _rank_col(df: pd.DataFrame, stat_col: str, ascending: bool = False) -> pd.Series:
    """Return rank series for a stat column (lower ascending = better defense)."""
    if stat_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    return df[stat_col].apply(coerce_float).rank(method='min', ascending=ascending).astype('Int64')


def build_mlb_hitting_rankings(hitting_df: pd.DataFrame) -> pd.DataFrame:
    """Return hitting rankings table for all teams."""
    if hitting_df.empty:
        return pd.DataFrame()
    df = hitting_df.copy()
    stat_map = {
        'runs': ('Runs', False),
        'avg': ('AVG', False),
        'obp': ('OBP', False),
        'slg': ('SLG', False),
        'ops': ('OPS', False),
        'homeRuns': ('HR', False),
        'stolenBases': ('SB', False),
        'strikeOuts': ('K', True),
        'baseOnBalls': ('BB', False),
    }
    out_rows = []
    for _, row in df.iterrows():
        entry = {
            'Team': clean_text(row.get('team_name')),
            'team_id': coerce_int(row.get('team_id')),
        }
        for api_col, (label, _) in stat_map.items():
            entry[label] = coerce_float(row.get(api_col, 0))
        out_rows.append(entry)
    out = pd.DataFrame(out_rows)
    for api_col, (label, asc) in stat_map.items():
        if label in out.columns:
            out[f'{label} Rank'] = out[label].rank(method='min', ascending=asc).astype('Int64')
    return out


def build_mlb_pitching_rankings(pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Return pitching rankings table for all teams."""
    if pitching_df.empty:
        return pd.DataFrame()
    df = pitching_df.copy()
    stat_map = {
        'era': ('ERA', True),
        'whip': ('WHIP', True),
        'strikeOuts': ('K', False),
        'baseOnBalls': ('BB', True),
        'homeRuns': ('HR Allowed', True),
        'saves': ('Saves', False),
    }
    out_rows = []
    for _, row in df.iterrows():
        entry = {
            'Team': clean_text(row.get('team_name')),
            'team_id': coerce_int(row.get('team_id')),
        }
        for api_col, (label, _) in stat_map.items():
            entry[label] = coerce_float(row.get(api_col, 0))
        out_rows.append(entry)
    out = pd.DataFrame(out_rows)
    for api_col, (label, asc) in stat_map.items():
        if label in out.columns:
            out[f'{label} Rank'] = out[label].rank(method='min', ascending=asc).astype('Int64')
    return out


def build_mlb_fielding_rankings(fielding_df: pd.DataFrame) -> pd.DataFrame:
    """Return fielding rankings table for all teams."""
    if fielding_df.empty:
        return pd.DataFrame()
    df = fielding_df.copy()
    stat_map = {
        'errors': ('Errors', True),
        'fieldingPercentage': ('FLD%', False),
        'doublePlays': ('DP', False),
    }
    out_rows = []
    for _, row in df.iterrows():
        entry = {
            'Team': clean_text(row.get('team_name')),
            'team_id': coerce_int(row.get('team_id')),
        }
        for api_col, (label, _) in stat_map.items():
            entry[label] = coerce_float(row.get(api_col, 0))
        out_rows.append(entry)
    out = pd.DataFrame(out_rows)
    for api_col, (label, asc) in stat_map.items():
        if label in out.columns:
            out[f'{label} Rank'] = out[label].rank(method='min', ascending=asc).astype('Int64')
    return out


def build_team_ranking_summary(
    team_id: int,
    hitting_rank_df: pd.DataFrame,
    pitching_rank_df: pd.DataFrame,
    fielding_rank_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a selected-team rank summary: each stat, the team's value, and rank out of 30.
    """
    rows: list[dict[str, Any]] = []

    def _extract(df: pd.DataFrame, label_cols: list[str], category: str) -> None:
        if df.empty:
            return
        team_row = df[df['team_id'] == team_id]
        if team_row.empty:
            return
        r = team_row.iloc[0]
        for col in label_cols:
            val = r.get(col)
            rank_val = r.get(f'{col} Rank')
            if val is not None:
                rows.append({
                    'Category': category,
                    'Stat': col,
                    'Value': round(coerce_float(val), 3),
                    'Rank': int(rank_val) if rank_val is not None and not pd.isna(rank_val) else None,
                    'Out Of': 30,
                })

    hit_stats = ['Runs', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'SB', 'BB', 'K']
    pit_stats = ['ERA', 'WHIP', 'K', 'BB', 'HR Allowed', 'Saves']
    fld_stats = ['Errors', 'FLD%', 'DP']

    _extract(hitting_rank_df, hit_stats, 'Offense')
    _extract(pitching_rank_df, pit_stats, 'Pitching')
    _extract(fielding_rank_df, fld_stats, 'Defense')

    if not rows:
        return pd.DataFrame(columns=['Category', 'Stat', 'Value', 'Rank', 'Out Of'])

    df_out = pd.DataFrame(rows)
    # Stoplight: rank 1 = best for all stats (ranking already handles ascending/descending).
    # stoplight(15.5 - rank): rank 1 → +14.5 🟢, rank 16 ≈ -0.5 🟡, rank 30 → -14.5 🔴
    df_out['Signal'] = df_out['Rank'].apply(
        lambda r: stoplight(15.5 - coerce_float(r, 15.5))
    )
    df_out['Rank'] = df_out['Rank'].apply(lambda x: f'#{x}' if x is not None else 'N/A')
    return df_out


# ─── WAR ──────────────────────────────────────────────────────────────────────

def build_war_leaderboard_df(war_raw: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Normalize raw WAR data from any source into a clean leaderboard."""
    if war_raw.empty:
        return pd.DataFrame(columns=['Player', 'Team', 'WAR', 'Type', 'Source'])
    df = war_raw.copy()
    if 'WAR' not in df.columns:
        return pd.DataFrame(columns=['Player', 'Team', 'WAR', 'Type', 'Source'])
    df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce')
    df = df.dropna(subset=['WAR']).sort_values('WAR', ascending=False)
    # Ensure Player col
    for candidate in ('Player', 'Name', 'player_name'):
        if candidate in df.columns and 'Player' not in df.columns:
            df = df.rename(columns={candidate: 'Player'})
    if 'Player' not in df.columns:
        df['Player'] = '-'
    if 'Team' not in df.columns:
        df['Team'] = '-'
    if 'Type' not in df.columns:
        df['Type'] = 'Batting'
    if 'Source' not in df.columns:
        df['Source'] = 'Unknown'
    return df.head(top_n).reset_index(drop=True)


# ─── Spray chart ──────────────────────────────────────────────────────────────

def _transform_statcast_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Statcast hc_x/hc_y to standard field coordinates centered at home plate."""
    df = df.copy()
    df['spray_x'] = df['hc_x'] - _HC_X_HOME
    df['spray_y'] = _HC_Y_HOME - df['hc_y']
    return df


def build_spray_df(
    statcast_df: pd.DataFrame,
    player_name: str | None = None,
    event_filter: str = 'All',
) -> pd.DataFrame:
    """
    Build a spray chart dataset from full-team Statcast data.

    Parameters
    ----------
    statcast_df : Full team Statcast DataFrame
    player_name : If set, filter to this player (batter)
    event_filter : 'All', 'Hits', 'Extra-Base Hits', 'Home Runs'

    Returns
    -------
    DataFrame with spray_x, spray_y, result, hover fields
    """
    if statcast_df.empty:
        return pd.DataFrame()
    df = statcast_df.copy()
    # Need hit coordinates
    if 'hc_x' not in df.columns or 'hc_y' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['hc_x', 'hc_y'])
    if df.empty:
        return pd.DataFrame()

    # Player filter
    if player_name:
        name_col = next((c for c in ('player_name', 'batter_name') if c in df.columns), None)
        if name_col:
            df = df[df[name_col].astype(str).str.strip() == player_name.strip()]

    # Event filter
    if 'events' in df.columns:
        if event_filter == 'Hits':
            df = df[df['events'].isin(HIT_EVENTS)]
        elif event_filter == 'Extra-Base Hits':
            df = df[df['events'].isin({'double', 'triple', 'home_run'})]
        elif event_filter == 'Home Runs':
            df = df[df['events'] == 'home_run']

    if df.empty:
        return pd.DataFrame()

    df = _transform_statcast_coords(df)

    # Result label
    if 'events' in df.columns:
        df['result'] = df['events'].apply(
            lambda e: 'Hit' if str(e) in HIT_EVENTS else ('Out' if str(e) in OUT_EVENTS else str(e))
        )
    else:
        df['result'] = 'Unknown'

    # Build hover parts
    name_col = next((c for c in ('player_name', 'batter_name') if c in df.columns), None)
    df['_player'] = df[name_col] if name_col else '-'

    hover_parts = ['_player', 'result']
    if 'events' in df.columns:
        hover_parts.append('events')
    for col in ('game_date', 'home_team', 'away_team', 'launch_speed', 'launch_angle', 'hit_distance_sc'):
        if col in df.columns:
            hover_parts.append(col)

    keep = ['spray_x', 'spray_y', 'result'] + [c for c in hover_parts if c in df.columns]
    return df[list(dict.fromkeys(keep))].reset_index(drop=True)


def get_spray_player_list(statcast_df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique batter names in the Statcast dataset."""
    if statcast_df.empty:
        return []
    for col in ('player_name', 'batter_name'):
        if col in statcast_df.columns:
            names = statcast_df[col].dropna().astype(str).str.strip().unique().tolist()
            return sorted(n for n in names if n and n != 'nan')
    return []


# ─── Home run dataset ─────────────────────────────────────────────────────────

def _estimate_apex_ft(exit_velo_mph: float, launch_angle_deg: float) -> float | None:
    """
    Estimate apex height using simplified projectile physics.
    H = (v0 * sin(theta))^2 / (2 * g), with drag approximation factor.
    """
    try:
        v0_fps = exit_velo_mph * 1.467  # mph to ft/s
        theta_rad = math.radians(launch_angle_deg)
        g = 32.174  # ft/s²
        drag_factor = 0.55  # empirical reduction for air resistance
        apex_ft = ((v0_fps * math.sin(theta_rad)) ** 2) / (2 * g) * drag_factor
        return round(apex_ft, 1) if apex_ft > 0 else None
    except Exception:
        return None


def build_hr_df(
    statcast_df: pd.DataFrame,
    location_filter: str = 'All',
) -> pd.DataFrame:
    """
    Build a home run event dataset for the team.

    Parameters
    ----------
    statcast_df : Full team Statcast batter DataFrame
    location_filter : 'All', 'Home', or 'Away'

    Returns
    -------
    DataFrame with HR events, coordinates, distances, estimated apex
    """
    if statcast_df.empty:
        return pd.DataFrame()
    if 'events' not in statcast_df.columns:
        return pd.DataFrame()

    df = statcast_df[statcast_df['events'] == 'home_run'].copy()
    if df.empty:
        return pd.DataFrame()

    # Location filter
    if location_filter in ('Home', 'Away') and 'home_team' in df.columns:
        # We don't always know which team the batter's team is in Statcast;
        # use the 'inning_topbot' field if available: Top = away batting, Bot = home batting
        if 'inning_topbot' in df.columns:
            if location_filter == 'Home':
                df = df[df['inning_topbot'].str.lower() == 'bot']
            else:
                df = df[df['inning_topbot'].str.lower() == 'top']

    if df.empty:
        return pd.DataFrame()

    # Coordinates
    has_coords = 'hc_x' in df.columns and 'hc_y' in df.columns
    if has_coords:
        df = _transform_statcast_coords(df)

    # Estimated apex
    if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
        df['apex_ft'] = df.apply(
            lambda r: _estimate_apex_ft(
                coerce_float(r.get('launch_speed'), 0),
                coerce_float(r.get('launch_angle'), 0),
            ),
            axis=1,
        )
    else:
        df['apex_ft'] = None

    # Player name
    name_col = next((c for c in ('player_name', 'batter_name') if c in df.columns), None)
    df['player'] = df[name_col] if name_col else '-'

    # Distance
    df['distance'] = df['hit_distance_sc'].apply(coerce_float) if 'hit_distance_sc' in df.columns else None

    # Game date
    df['game_date'] = df['game_date'].astype(str) if 'game_date' in df.columns else '-'

    # Build output
    out_cols = ['player', 'game_date', 'distance', 'apex_ft']
    if has_coords:
        out_cols += ['spray_x', 'spray_y']
    for col in ('launch_speed', 'launch_angle', 'home_team', 'away_team', 'inning'):
        if col in df.columns:
            out_cols.append(col)

    out_cols = list(dict.fromkeys(out_cols))
    return df[[c for c in out_cols if c in df.columns]].reset_index(drop=True)


def build_hr_summary_cards(hr_df: pd.DataFrame, full_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build KPI summary cards for the HR tab."""
    total_hr = len(hr_df)
    avg_dist = round(hr_df['distance'].dropna().mean(), 1) if 'distance' in hr_df.columns and total_hr else None
    max_dist = round(hr_df['distance'].dropna().max(), 0) if 'distance' in hr_df.columns and total_hr else None
    avg_la = round(hr_df['launch_angle'].dropna().mean(), 1) if 'launch_angle' in hr_df.columns and total_hr else None
    avg_apex = round(hr_df['apex_ft'].dropna().mean(), 1) if 'apex_ft' in hr_df.columns and total_hr else None

    # Home/Away split from full dataset
    home_hr = away_hr = 0
    if not full_df.empty and 'inning_topbot' in full_df.columns:
        full_hr = full_df[full_df['events'] == 'home_run'] if 'events' in full_df.columns else pd.DataFrame()
        if not full_hr.empty:
            home_hr = int((full_hr['inning_topbot'].str.lower() == 'bot').sum())
            away_hr = int((full_hr['inning_topbot'].str.lower() == 'top').sum())

    cards = [
        {'label': 'Total HR', 'value': total_hr},
        {'label': 'Avg Distance', 'value': f'{avg_dist} ft' if avg_dist else 'N/A'},
        {'label': 'Max Distance', 'value': f'{int(max_dist)} ft' if max_dist else 'N/A'},
        {'label': 'Avg Launch Angle', 'value': f'{avg_la}°' if avg_la else 'N/A'},
        {'label': 'Estimated Avg Apex', 'value': f'{avg_apex} ft' if avg_apex else 'N/A'},
        {'label': 'Home HRs', 'value': home_hr},
        {'label': 'Away HRs', 'value': away_hr},
    ]
    return cards


# ─── Statcast batter / pitcher grades (kept from original) ───────────────────

def _player_name_series(df: pd.DataFrame) -> pd.Series:
    for col in ['player_name', 'batter_name', 'pitcher_name']:
        if col in df.columns:
            return df[col].astype(str).fillna('Unknown')
    return pd.Series(['Unknown'] * len(df), index=df.index)


def _grade_from_score(score: float) -> str:
    if score >= 85:
        return 'A'
    if score >= 75:
        return 'B'
    if score >= 65:
        return 'C'
    if score >= 55:
        return 'D'
    return 'F'


def _statcast_batter_score(avg_ev: float, hard_hit: float, xwoba: float, whiff_pct: float, contact_quality: float) -> float:
    ev_component = min(max((avg_ev - 85) / 10, 0), 1) * 25
    hh_component = min(max(hard_hit / 50, 0), 1) * 25
    whiff_component = max(0, min((100 - whiff_pct) / 100 * 40, 1)) * 15
    xwoba_component = min(max((xwoba - 0.28) / 0.12, 0), 1) * 20
    contact_component = min(max(contact_quality / 100, 0), 1) * 15
    return round(ev_component + hh_component + whiff_component + xwoba_component + contact_component, 1)


def build_batter_grades_df(statcast_batter_df: pd.DataFrame) -> pd.DataFrame:
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Whiff %', 'xwOBA', 'Grade', 'Trend'])
    df = statcast_batter_df.copy()
    df['Batter'] = _player_name_series(df)
    if 'launch_speed' not in df.columns:
        df['launch_speed'] = 0.0
    if 'estimated_woba_using_speedangle' not in df.columns:
        df['estimated_woba_using_speedangle'] = 0.0
    if 'description' not in df.columns:
        df['description'] = ''

    contact = df[df['launch_speed'] > 0].copy()
    if contact.empty:
        return pd.DataFrame(columns=['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Whiff %', 'xwOBA', 'Grade', 'Trend'])

    rows = []
    for batter, grp in contact.groupby('Batter'):
        bip = len(grp)
        avg_ev = grp['launch_speed'].mean()
        hard_hit = (grp['launch_speed'] >= 95).mean() * 100 if bip else 0.0
        all_pa = df[df['Batter'] == batter]
        swings = all_pa['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = all_pa['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        xwoba = grp['estimated_woba_using_speedangle'].replace(0, pd.NA).dropna().mean()
        xwoba = 0.0 if pd.isna(xwoba) else float(xwoba)
        contact_quality = (grp['launch_speed'] >= 95).mean() * 100
        score = _statcast_batter_score(avg_ev, hard_hit, xwoba, whiff_pct, contact_quality)
        rows.append({
            'Batter': batter,
            'PA': len(all_pa),
            'Avg EV': round(avg_ev, 1),
            'Hard Hit %': round(hard_hit, 1),
            'Whiff %': round(whiff_pct, 1),
            'xwOBA': round(xwoba, 3),
            'Grade': _grade_from_score(score),
            'Trend': stoplight(score - 70, neutral_band=5),
            'Score': score,
        })
    out = pd.DataFrame(rows).sort_values(['Score', 'PA'], ascending=[False, False]).reset_index(drop=True)
    return out[['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Whiff %', 'xwOBA', 'Grade', 'Trend']]


def _statcast_pitcher_score(avg_spin: float, whiff_pct: float, avg_woba: float, strike_pct: float) -> float:
    spin_component = min(max((avg_spin - 2100) / 500, 0), 1) * 30
    whiff_component = min(max(whiff_pct / 40, 0), 1) * 30
    woba_component = min(max((0.36 - avg_woba) / 0.12, 0), 1) * 25
    efficiency_component = min(max(strike_pct / 70, 0), 1) * 15
    return round(spin_component + whiff_component + woba_component + efficiency_component, 1)


def build_pitcher_grades_df(statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    if statcast_pitcher_df.empty:
        return pd.DataFrame(columns=['Pitcher', 'Pitches', 'Avg Velo', 'Avg Spin', 'Whiff %', 'K%', 'wOBA Allowed', 'Grade', 'Trend'])
    df = statcast_pitcher_df.copy()
    df['Pitcher'] = _player_name_series(df)
    for col in ['release_speed', 'release_spin_rate', 'woba_value']:
        if col not in df.columns:
            df[col] = 0.0
    if 'description' not in df.columns:
        df['description'] = ''
    if 'events' not in df.columns:
        df['events'] = ''

    rows = []
    for pitcher, grp in df.groupby('Pitcher'):
        pitches = len(grp)
        swings = grp['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = grp['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        strikes = grp['description'].isin({'swinging_strike', 'swinging_strike_blocked', 'called_strike'}).sum()
        strike_pct = (strikes / pitches * 100) if pitches else 0.0
        avg_woba = grp['woba_value'].replace(0, pd.NA).dropna().mean()
        avg_woba = 0.0 if pd.isna(avg_woba) else float(avg_woba)
        k_events = grp['events'].isin({'strikeout', 'strikeout_double_play'}).sum()
        k_pct = (k_events / pitches * 100) if pitches else 0.0
        score = _statcast_pitcher_score(grp['release_spin_rate'].mean(), whiff_pct, avg_woba if avg_woba > 0 else 0.3, strike_pct)
        rows.append({
            'Pitcher': pitcher,
            'Pitches': pitches,
            'Avg Velo': round(grp['release_speed'].mean(), 1),
            'Avg Spin': round(grp['release_spin_rate'].mean(), 0),
            'Whiff %': round(whiff_pct, 1),
            'K%': round(k_pct, 1),
            'wOBA Allowed': round(avg_woba, 3),
            'Grade': _grade_from_score(score),
            'Trend': stoplight(score - 70, neutral_band=5),
            'Score': score,
        })
    out = pd.DataFrame(rows).sort_values(['Score', 'Pitches'], ascending=[False, False]).reset_index(drop=True)
    return out[['Pitcher', 'Pitches', 'Avg Velo', 'Avg Spin', 'Whiff %', 'K%', 'wOBA Allowed', 'Grade', 'Trend']]


def build_pitch_mix_df(statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    if statcast_pitcher_df.empty or 'pitch_type' not in statcast_pitcher_df.columns:
        return pd.DataFrame(columns=['Pitch Type', 'Usage %', 'Avg Velo', 'Avg Spin', 'Whiff %', 'Hit %', 'Success'])
    df = statcast_pitcher_df.copy()
    if 'description' not in df.columns:
        df['description'] = ''
    if 'events' not in df.columns:
        df['events'] = ''
    total = len(df)
    rows = []
    for pitch_type, grp in df.groupby('pitch_type'):
        swings = grp['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = grp['description'].isin(WHIFF_DESCRIPTIONS).sum()
        balls_in_play = grp['events'].isin(HIT_EVENTS | OUT_EVENTS).sum()
        hits = grp['events'].isin(HIT_EVENTS).sum()
        usage = len(grp) / total * 100 if total else 0.0
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        hit_pct = (hits / balls_in_play * 100) if balls_in_play else 0.0
        success_score = (usage * 0.1) + (whiff_pct * 1.2) + max(0.0, 30 - hit_pct)
        rows.append({
            'Pitch Type': pitch_type,
            'Usage %': round(usage, 1),
            'Avg Velo': round(grp['release_speed'].mean(), 1) if 'release_speed' in grp else 0.0,
            'Avg Spin': round(grp['release_spin_rate'].mean(), 0) if 'release_spin_rate' in grp else 0.0,
            'Whiff %': round(whiff_pct, 1),
            'Hit %': round(hit_pct, 1),
            'Success': stoplight(success_score - 35, neutral_band=5),
            'Score': round(success_score, 1),
        })
    out = pd.DataFrame(rows).sort_values(['Usage %', 'Score'], ascending=[False, False]).reset_index(drop=True)
    return out[['Pitch Type', 'Usage %', 'Avg Velo', 'Avg Spin', 'Whiff %', 'Hit %', 'Success']]


def build_statcast_summary_df(statcast_batter_df: pd.DataFrame, statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    batter_contact = (
        statcast_batter_df[statcast_batter_df['launch_speed'] > 0].copy()
        if not statcast_batter_df.empty and 'launch_speed' in statcast_batter_df.columns
        else pd.DataFrame()
    )
    total_pitcher = len(statcast_pitcher_df)
    swings = (
        statcast_pitcher_df['description'].isin(SWING_DESCRIPTIONS).sum()
        if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns
        else 0
    )
    whiffs = (
        statcast_pitcher_df['description'].isin(WHIFF_DESCRIPTIONS).sum()
        if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns
        else 0
    )
    pitch_whiff = (whiffs / swings * 100) if swings else 0.0
    avg_spin = (
        statcast_pitcher_df['release_spin_rate'].mean()
        if not statcast_pitcher_df.empty and 'release_spin_rate' in statcast_pitcher_df.columns
        else 0.0
    )
    avg_ev = batter_contact['launch_speed'].mean() if not batter_contact.empty else 0.0
    hard_hit = (batter_contact['launch_speed'] >= 95).mean() * 100 if not batter_contact.empty else 0.0
    return pd.DataFrame([
        {'Metric': 'Team Avg Exit Velocity', 'Value': round(avg_ev, 1), 'Trend': stoplight(avg_ev - 89, neutral_band=1)},
        {'Metric': 'Team Hard Hit %', 'Value': round(hard_hit, 1), 'Trend': stoplight(hard_hit - 40, neutral_band=3)},
        {'Metric': 'Staff Avg Spin Rate', 'Value': round(avg_spin, 0), 'Trend': stoplight(avg_spin - 2250, neutral_band=50)},
        {'Metric': 'Staff Whiff %', 'Value': round(pitch_whiff, 1), 'Trend': stoplight(pitch_whiff - 28, neutral_band=2)},
        {'Metric': 'Pitch Sample Size', 'Value': total_pitcher, 'Trend': '🟡 Context'},
    ])

