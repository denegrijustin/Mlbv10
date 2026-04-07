from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from formatting import coerce_float, coerce_int, format_record, safe_pct, signed, stoplight

# ---------------------------------------------------------------------------
# Pitch description / event classification sets
# ---------------------------------------------------------------------------

SWING_DESCRIPTIONS = {
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score', 'missed_bunt',
}
WHIFF_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'foul_tip'}
HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
XBH_EVENTS = {'double', 'triple', 'home_run'}
OUT_EVENTS = {
    'field_out', 'force_out', 'double_play', 'fielders_choice_out', 'grounded_into_double_play',
    'sac_fly', 'sac_bunt', 'fielders_choice', 'triple_play', 'sac_fly_double_play',
}
STRIKE_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'called_strike', 'foul_tip'}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _player_name_series(df: pd.DataFrame) -> pd.Series:
    for col in ('player_name', 'batter_name', 'pitcher_name'):
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


def _nan_mean(series: pd.Series) -> float | None:
    """Return float mean or None if series has no valid values."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.mean()) if len(s) else None


def _weighted_score(components: list[tuple[float | None, float]]) -> float:
    """
    Compute a weighted sum, redistributing weight from missing components.
    Each element is (value_0_to_1_or_None, weight).
    Returns a 0-100 score.
    """
    available = [(v, w) for v, w in components if v is not None]
    if not available:
        return 0.0
    total_w = sum(w for _, w in available)
    score = sum(v * (w / total_w) for v, w in available) * 100
    return float(min(max(round(score, 2), 0.0), 100.0))


# ---------------------------------------------------------------------------
# Validation & display helpers (new)
# ---------------------------------------------------------------------------

def validate_required_columns(df: pd.DataFrame, required_cols: list[str], section_name: str) -> tuple[bool, str]:
    """Return (True, '') if all cols present and df not empty, else (False, message)."""
    if df.empty:
        return False, f'{section_name}: DataFrame is empty.'
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f'{section_name}: missing columns: {missing}'
    return True, ''


def safe_display_value(value: Any, fmt: str | None = None) -> str:
    """Convert value to a safe display string. Returns 'N/A' for missing data."""
    if value is None:
        return 'N/A'
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 'N/A'
    except TypeError:
        pass
    try:
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            return 'N/A'
        if fmt is not None:
            return format(fv, fmt)
        return str(value)
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s if s not in ('nan', 'None', 'NaN', 'inf', '-inf', '') else 'N/A'


# ---------------------------------------------------------------------------
# Existing public helpers (kept / refactored internals)
# ---------------------------------------------------------------------------

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
        lambda r: coerce_int(r['away_score'], 0) if r['away'] == team_name else coerce_int(r['home_score'], 0), axis=1
    )
    df['opp_runs'] = df.apply(
        lambda r: coerce_int(r['home_score'], 0) if r['away'] == team_name else coerce_int(r['away_score'], 0), axis=1
    )
    df['run_diff'] = df['team_runs'] - df['opp_runs']
    df['is_final'] = df['status'].astype(str).str.contains('Final', case=False, na=False)
    df['result'] = df.apply(
        lambda r: 'W' if r['team_runs'] > r['opp_runs'] else ('L' if r['team_runs'] < r['opp_runs'] else 'T'), axis=1
    )
    return df.reset_index(drop=True)


def build_team_snapshot(team_row: dict[str, Any] | None, season_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict[str, Any]:
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
        if not daily_df.empty else 'No game today'
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
        {'Metric': 'Season Avg Runs For', 'Value': str(round(season_rf, 2)), 'Trend': '🟡 Baseline'},
        {'Metric': 'Season Avg Runs Against', 'Value': str(round(season_ra, 2)), 'Trend': '🟡 Baseline'},
        {'Metric': 'Last 5 Avg Runs For', 'Value': str(round(last5_rf, 2)), 'Trend': stoplight(last5_rf - prev5_rf)},
        {'Metric': 'Last 5 Avg Runs Against', 'Value': str(round(last5_ra, 2)), 'Trend': stoplight(prev5_ra - last5_ra)},
        {
            'Metric': 'Last 10 Record',
            'Value': format_record(last10_wins, len(last_10) - last10_wins),
            'Trend': stoplight((last10_wins / max(len(last_10), 1)) - 0.5),
        },
        {
            'Metric': 'Last 10 Run Differential / Game',
            'Value': str(round(last_10['run_diff'].mean(), 2)),
            'Trend': stoplight(last_10['run_diff'].mean()),
        },
        {'Metric': 'Scoring Consistency Std Dev', 'Value': str(consistency), 'Trend': stoplight(2.5 - consistency)},
        {
            'Metric': 'Home Avg Runs',
            'Value': str(round(home_split['team_runs'].mean(), 2)) if not home_split.empty else '0.0',
            'Trend': '🟡 Split',
        },
        {
            'Metric': 'Away Avg Runs',
            'Value': str(round(away_split['team_runs'].mean(), 2)) if not away_split.empty else '0.0',
            'Trend': '🟡 Split',
        },
    ]
    return pd.DataFrame(rows)


def build_recent_games_df(season_df: pd.DataFrame, team_name: str, count: int = 10) -> pd.DataFrame:
    games = _team_games(season_df, team_name)
    cols = ['Date', 'Opponent', 'Location', 'Result', 'Team Runs', 'Opp Runs', 'Run Diff']
    if games.empty:
        return pd.DataFrame(columns=cols)
    finals = games[games['is_final']].tail(count).copy()
    if finals.empty:
        return pd.DataFrame(columns=cols)
    finals['Date'] = finals['officialDate']
    finals['Opponent'] = finals['opponent']
    finals['Location'] = finals['location']
    finals['Result'] = finals['result']
    finals['Team Runs'] = finals['team_runs']
    finals['Opp Runs'] = finals['opp_runs']
    finals['Run Diff'] = finals['run_diff']
    return finals[cols].reset_index(drop=True)


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
    return pd.DataFrame([
        {'Field': 'Matchup', 'Value': f"{live_summary.get('away_team', '-')} at {live_summary.get('home_team', '-')}"},
        {'Field': 'Status', 'Value': live_summary.get('status', '-')},
        {'Field': 'Inning', 'Value': f"{live_summary.get('inning_state', '-')} {live_summary.get('inning', '-')}"},
        {'Field': 'Score', 'Value': f"{live_summary.get('away_runs', 0)}-{live_summary.get('home_runs', 0)}"},
        {'Field': 'Count', 'Value': f"{live_summary.get('balls', 0)} balls, {live_summary.get('strikes', 0)} strikes, {live_summary.get('outs', 0)} outs"},
    ])


def build_kpi_cards(snapshot: dict[str, Any], trend_df: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = {row['Metric']: row for _, row in trend_df.iterrows()} if not trend_df.empty else {}
    return [
        {'label': 'Record', 'value': snapshot.get('record', '0-0'), 'delta': f"{snapshot.get('win_pct', 0.0)}% win pct"},
        {'label': 'Season Avg RF', 'value': snapshot.get('avg_runs_for', 0.0), 'delta': lookup.get('Last 5 Avg Runs For', {}).get('Trend', '🟡 Even')},
        {'label': 'Season Avg RA', 'value': snapshot.get('avg_runs_against', 0.0), 'delta': lookup.get('Last 5 Avg Runs Against', {}).get('Trend', '🟡 Even')},
        {
            'label': 'Run Diff',
            'value': snapshot.get('run_diff', 0),
            'delta': signed(coerce_float(snapshot.get('avg_runs_for', 0.0)) - coerce_float(snapshot.get('avg_runs_against', 0.0)), 2),
        },
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


# ---------------------------------------------------------------------------
# Statcast grade helpers (existing, refactored internals)
# ---------------------------------------------------------------------------

def _statcast_batter_score(avg_ev: float, hard_hit: float, xwoba: float, whiff_pct: float, contact_quality: float) -> float:
    ev_component = min(max((avg_ev - 85) / 10, 0), 1) * 25
    hh_component = min(max(hard_hit / 50, 0), 1) * 25
    whiff_component = max(0, min((100 - whiff_pct) / 100 * 40, 1)) * 15
    xwoba_component = min(max((xwoba - 0.28) / 0.12, 0), 1) * 20
    contact_component = min(max(contact_quality / 100, 0), 1) * 15
    return round(ev_component + hh_component + whiff_component + xwoba_component + contact_component, 1)


def build_batter_grades_df(statcast_batter_df: pd.DataFrame) -> pd.DataFrame:
    cols_out = ['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Whiff %', 'xwOBA', 'Grade', 'Trend']
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=cols_out)
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
        return pd.DataFrame(columns=cols_out)

    rows: list[dict[str, Any]] = []
    for batter, grp in contact.groupby('Batter'):
        bip = len(grp)
        avg_ev = grp['launch_speed'].mean()
        hard_hit = (grp['launch_speed'] >= 95).mean() * 100 if bip else 0.0
        all_pa = df[df['Batter'] == batter]
        swings = all_pa['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = all_pa['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        xwoba_raw = grp['estimated_woba_using_speedangle'].replace(0, pd.NA).dropna().mean()
        xwoba = 0.0 if pd.isna(xwoba_raw) else float(xwoba_raw)
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
    return out[cols_out]


def _statcast_pitcher_score(avg_spin: float, whiff_pct: float, avg_woba: float, strike_pct: float) -> float:
    spin_component = min(max((avg_spin - 2100) / 500, 0), 1) * 30
    whiff_component = min(max(whiff_pct / 40, 0), 1) * 30
    woba_component = min(max((0.36 - avg_woba) / 0.12, 0), 1) * 25
    efficiency_component = min(max(strike_pct / 70, 0), 1) * 15
    return round(spin_component + whiff_component + woba_component + efficiency_component, 1)


def build_pitcher_grades_df(statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    cols_out = ['Pitcher', 'Pitches', 'Avg Velo', 'Avg Spin', 'Whiff %', 'K%', 'wOBA Allowed', 'Grade', 'Trend']
    if statcast_pitcher_df.empty:
        return pd.DataFrame(columns=cols_out)
    df = statcast_pitcher_df.copy()
    df['Pitcher'] = _player_name_series(df)
    for col in ('release_speed', 'release_spin_rate', 'woba_value'):
        if col not in df.columns:
            df[col] = 0.0
    if 'description' not in df.columns:
        df['description'] = ''
    if 'events' not in df.columns:
        df['events'] = ''

    rows: list[dict[str, Any]] = []
    for pitcher, grp in df.groupby('Pitcher'):
        pitches = len(grp)
        swings = grp['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = grp['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        strikes = grp['description'].isin(STRIKE_DESCRIPTIONS).sum()
        strike_pct = (strikes / pitches * 100) if pitches else 0.0
        avg_woba_raw = grp['woba_value'].replace(0, pd.NA).dropna().mean()
        avg_woba = 0.0 if pd.isna(avg_woba_raw) else float(avg_woba_raw)
        k_events = grp['events'].isin({'strikeout', 'strikeout_double_play'}).sum()
        k_pct = (k_events / pitches * 100) if pitches else 0.0
        score = _statcast_pitcher_score(
            grp['release_spin_rate'].mean(), whiff_pct, avg_woba if avg_woba > 0 else 0.3, strike_pct
        )
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
    return out[cols_out]


def build_pitch_mix_df(statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    cols_out = ['Pitch Type', 'Usage %', 'Avg Velo', 'Avg Spin', 'Whiff %', 'Hit %', 'Success']
    if statcast_pitcher_df.empty or 'pitch_type' not in statcast_pitcher_df.columns:
        return pd.DataFrame(columns=cols_out)
    df = statcast_pitcher_df.copy()
    if 'description' not in df.columns:
        df['description'] = ''
    if 'events' not in df.columns:
        df['events'] = ''
    total = len(df)
    rows: list[dict[str, Any]] = []
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
            'Avg Velo': round(grp['release_speed'].mean(), 1) if 'release_speed' in grp.columns else 0.0,
            'Avg Spin': round(grp['release_spin_rate'].mean(), 0) if 'release_spin_rate' in grp.columns else 0.0,
            'Whiff %': round(whiff_pct, 1),
            'Hit %': round(hit_pct, 1),
            'Success': stoplight(success_score - 35, neutral_band=5),
            'Score': round(success_score, 1),
        })
    out = pd.DataFrame(rows).sort_values(['Usage %', 'Score'], ascending=[False, False]).reset_index(drop=True)
    return out[cols_out]


def build_statcast_summary_df(statcast_batter_df: pd.DataFrame, statcast_pitcher_df: pd.DataFrame) -> pd.DataFrame:
    if not statcast_batter_df.empty and 'launch_speed' in statcast_batter_df.columns:
        batter_contact = statcast_batter_df[statcast_batter_df['launch_speed'] > 0].copy()
    else:
        batter_contact = pd.DataFrame()
    total_pitcher = len(statcast_pitcher_df)
    swings = (
        statcast_pitcher_df['description'].isin(SWING_DESCRIPTIONS).sum()
        if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns else 0
    )
    whiffs = (
        statcast_pitcher_df['description'].isin(WHIFF_DESCRIPTIONS).sum()
        if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns else 0
    )
    pitch_whiff = (whiffs / swings * 100) if swings else 0.0
    avg_spin = (
        statcast_pitcher_df['release_spin_rate'].mean()
        if not statcast_pitcher_df.empty and 'release_spin_rate' in statcast_pitcher_df.columns else 0.0
    )
    avg_ev = batter_contact['launch_speed'].mean() if not batter_contact.empty else 0.0
    hard_hit = (batter_contact['launch_speed'] >= 95).mean() * 100 if not batter_contact.empty else 0.0
    return pd.DataFrame([
        {'Metric': 'Team Avg Exit Velocity', 'Value': str(round(avg_ev, 1)), 'Trend': stoplight(avg_ev - 89, neutral_band=1)},
        {'Metric': 'Team Hard Hit %', 'Value': str(round(hard_hit, 1)), 'Trend': stoplight(hard_hit - 40, neutral_band=3)},
        {'Metric': 'Staff Avg Spin Rate', 'Value': str(round(avg_spin, 0)), 'Trend': stoplight(avg_spin - 2250, neutral_band=50)},
        {'Metric': 'Staff Whiff %', 'Value': str(round(pitch_whiff, 1)), 'Trend': stoplight(pitch_whiff - 28, neutral_band=2)},
        {'Metric': 'Pitch Sample Size', 'Value': str(total_pitcher), 'Trend': '🟡 Context'},
    ])


# ---------------------------------------------------------------------------
# New functions: hitter / pitcher impact scores
# ---------------------------------------------------------------------------

def compute_hitter_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom 0-100 hitter impact score from statcast batter data.
    Uses missing-field reweighting; never returns zeros for completely missing data.
    """
    cols_out = ['player_name', 'impact_score', 'PA', 'avg_ev', 'hard_hit_pct', 'xwoba', 'bb_rate', 'k_rate', 'hr_count', 'xbh_count']
    if df.empty:
        return pd.DataFrame(columns=cols_out)

    work = df.copy()
    work['_name'] = _player_name_series(work)
    if 'description' not in work.columns:
        work['description'] = ''
    if 'events' not in work.columns:
        work['events'] = ''

    rows: list[dict[str, Any]] = []
    for name, grp in work.groupby('_name'):
        pa = len(grp)

        # Exit velocity / hard hit
        if 'launch_speed' in grp.columns:
            ev_vals = pd.to_numeric(grp['launch_speed'], errors='coerce').dropna()
            avg_ev = float(ev_vals.mean()) if len(ev_vals) else None
            hard_hit_pct = float((ev_vals >= 95).mean() * 100) if len(ev_vals) else None
        else:
            avg_ev = None
            hard_hit_pct = None

        # xwOBA
        if 'estimated_woba_using_speedangle' in grp.columns:
            xw = pd.to_numeric(grp['estimated_woba_using_speedangle'], errors='coerce').replace(0.0, pd.NA).dropna()
            xwoba = float(xw.mean()) if len(xw) else None
        elif 'woba_value' in grp.columns:
            xw = pd.to_numeric(grp['woba_value'], errors='coerce').replace(0.0, pd.NA).dropna()
            xwoba = float(xw.mean()) if len(xw) else None
        else:
            xwoba = None

        # Barrel %
        if 'barrel' in grp.columns:
            b_vals = pd.to_numeric(grp['barrel'], errors='coerce').fillna(0)
            if 'launch_speed' in grp.columns:
                contact_n = int(pd.to_numeric(grp['launch_speed'], errors='coerce').dropna().size)
            else:
                contact_n = len(grp)
            barrel_pct = float(b_vals.sum() / contact_n * 100) if contact_n else None
        else:
            barrel_pct = None

        # Discipline: BB rate, K rate
        swings = grp['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = grp['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_rate = float(whiffs / swings) if swings else None

        bb_events = grp['events'].isin({'walk', 'intent_walk'}).sum()
        k_events = grp['events'].isin({'strikeout', 'strikeout_double_play'}).sum()
        plate_events = grp['events'].notna() & (grp['events'] != '')
        plate_n = plate_events.sum()
        bb_rate = float(bb_events / plate_n) if plate_n else None
        k_rate = float(k_events / plate_n) if plate_n else None

        # Power
        hr_count = int(grp['events'].isin({'home_run'}).sum())
        xbh_count = int(grp['events'].isin(XBH_EVENTS).sum())

        # --- Weighted scoring (0-1 per component, weights sum to 1.0) ---
        # 35% production
        xwoba_norm = min(max((xwoba - 0.250) / 0.200, 0), 1) if xwoba is not None else None
        # 25% quality of contact
        ev_norm = min(max((avg_ev - 80) / 20, 0), 1) if avg_ev is not None else None
        hh_norm = min(max(hard_hit_pct / 60, 0), 1) if hard_hit_pct is not None else None
        barrel_norm = min(max(barrel_pct / 20, 0), 1) if barrel_pct is not None else None
        # 20% discipline
        bb_norm = min(max(bb_rate / 0.15, 0), 1) if bb_rate is not None else None
        # 10% power
        hr_norm = min(max(hr_count / 30, 0), 1) if hr_count is not None else None
        xbh_norm = min(max(xbh_count / 60, 0), 1) if xbh_count is not None else None
        # 10% penalty (negative: high K/whiff is bad)
        k_penalty = min(max(1 - (k_rate / 0.40), 0), 1) if k_rate is not None else None
        whiff_penalty = min(max(1 - (whiff_rate / 0.50), 0), 1) if whiff_rate is not None else None

        components: list[tuple[float | None, float]] = [
            (xwoba_norm, 0.35),
            (ev_norm, 0.10),
            (hh_norm, 0.10),
            (barrel_norm, 0.05),
            (bb_norm, 0.20),
            (hr_norm, 0.05),
            (xbh_norm, 0.05),
            (k_penalty, 0.05),
            (whiff_penalty, 0.05),
        ]
        impact = _weighted_score(components)

        rows.append({
            'player_name': str(name),
            'impact_score': round(impact, 2),
            'PA': pa,
            'avg_ev': safe_display_value(avg_ev, '.1f'),
            'hard_hit_pct': safe_display_value(hard_hit_pct, '.1f'),
            'xwoba': safe_display_value(xwoba, '.3f'),
            'bb_rate': safe_display_value(bb_rate, '.3f'),
            'k_rate': safe_display_value(k_rate, '.3f'),
            'hr_count': hr_count,
            'xbh_count': xbh_count,
        })

    if not rows:
        return pd.DataFrame(columns=cols_out)
    return pd.DataFrame(rows).sort_values('impact_score', ascending=False).reset_index(drop=True)[cols_out]


def classify_pitcher_role(pitcher_name: str, df: pd.DataFrame) -> str:
    """Classify a pitcher as 'Starter' or 'Reliever' using pitch-count heuristic."""
    if df.empty:
        return 'Reliever'
    name_col = None
    for c in ('player_name', 'pitcher_name'):
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return 'Reliever'
    grp = df[df[name_col].astype(str) == str(pitcher_name)]
    if grp.empty:
        return 'Reliever'
    # Group by game_pk or game_date to count pitches per appearance
    game_col = 'game_pk' if 'game_pk' in grp.columns else ('game_date' if 'game_date' in grp.columns else None)
    if game_col:
        per_game = grp.groupby(game_col).size()
        if (per_game >= 50).any():
            return 'Starter'
    else:
        if len(grp) >= 50:
            return 'Starter'
    return 'Reliever'


def compute_pitcher_impact(df: pd.DataFrame, role: str = 'Starter') -> pd.DataFrame:
    """
    Custom 0-100 pitcher impact score from statcast pitcher data.
    role: 'Starter' or 'Reliever'. Uses missing-field reweighting.
    """
    cols_out = ['player_name', 'role', 'impact_score', 'pitches', 'avg_velo', 'whiff_pct', 'k_rate', 'bb_rate', 'woba_allowed', 'avg_ev_allowed']
    if df.empty:
        return pd.DataFrame(columns=cols_out)

    work = df.copy()
    work['_name'] = _player_name_series(work)
    if 'description' not in work.columns:
        work['description'] = ''
    if 'events' not in work.columns:
        work['events'] = ''

    rows: list[dict[str, Any]] = []
    for name, grp in work.groupby('_name'):
        pitches = len(grp)

        # Velocity
        avg_velo = _nan_mean(grp.get('release_speed', pd.Series(dtype=float)))

        # Whiff / K
        swings = grp['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = grp['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_rate = float(whiffs / swings) if swings else None

        plate_events = grp['events'].notna() & (grp['events'] != '')
        plate_n = plate_events.sum()
        k_events = grp['events'].isin({'strikeout', 'strikeout_double_play'}).sum()
        bb_events = grp['events'].isin({'walk', 'intent_walk'}).sum()
        k_rate = float(k_events / plate_n) if plate_n else None
        bb_rate = float(bb_events / plate_n) if plate_n else None

        # Command: strike %
        strikes = grp['description'].isin(STRIKE_DESCRIPTIONS).sum()
        strike_pct = float(strikes / pitches) if pitches else None

        # wOBA allowed
        if 'woba_value' in grp.columns:
            wv = pd.to_numeric(grp['woba_value'], errors='coerce').replace(0.0, pd.NA).dropna()
            woba_allowed = float(wv.mean()) if len(wv) else None
        else:
            woba_allowed = None

        # Contact suppression: hard hit % allowed, barrel % allowed
        if 'launch_speed' in grp.columns:
            ev_vals = pd.to_numeric(grp['launch_speed'], errors='coerce').dropna()
            avg_ev_allowed = float(ev_vals.mean()) if len(ev_vals) else None
            hh_allowed_pct = float((ev_vals >= 95).mean() * 100) if len(ev_vals) else None
        else:
            avg_ev_allowed = None
            hh_allowed_pct = None

        if 'barrel' in grp.columns:
            b_vals = pd.to_numeric(grp['barrel'], errors='coerce').fillna(0)
            if 'launch_speed' in grp.columns:
                bip = int(pd.to_numeric(grp['launch_speed'], errors='coerce').dropna().size)
            else:
                bip = len(grp)
            barrel_allowed_pct = float(b_vals.sum() / bip * 100) if bip else None
        else:
            barrel_allowed_pct = None

        # IP proxy
        ip_proxy = pitches / 15 if pitches else None

        # --- Build component list by role ---
        # All components: (value_0_to_1_or_None, weight)
        # Run prevention (higher wOBA allowed = worse)
        woba_norm = min(max((0.400 - woba_allowed) / 0.200, 0), 1) if woba_allowed is not None else None
        # Bat-missing
        whiff_norm = min(max(whiff_rate / 0.40, 0), 1) if whiff_rate is not None else None
        k_norm = min(max((k_rate or 0) / 0.35, 0), 1) if k_rate is not None else None
        # Command
        bb_norm = min(max(1 - (bb_rate or 0) / 0.15, 0), 1) if bb_rate is not None else None
        strike_norm = min(max((strike_pct or 0) / 0.70, 0), 1) if strike_pct is not None else None
        # Contact suppression
        hh_norm = min(max(1 - (hh_allowed_pct or 0) / 60, 0), 1) if hh_allowed_pct is not None else None
        barrel_norm = min(max(1 - (barrel_allowed_pct or 0) / 20, 0), 1) if barrel_allowed_pct is not None else None
        # Innings (starter only)
        ip_norm = min(max((ip_proxy or 0) / 7, 0), 1) if ip_proxy is not None else None

        if role == 'Starter':
            components: list[tuple[float | None, float]] = [
                (woba_norm, 0.30),
                (whiff_norm, 0.125),
                (k_norm, 0.125),
                (bb_norm, 0.10),
                (strike_norm, 0.10),
                (hh_norm, 0.075),
                (barrel_norm, 0.075),
                (ip_norm, 0.10),
            ]
        else:  # Reliever
            components = [
                (woba_norm, 0.35),
                (whiff_norm, 0.15),
                (k_norm, 0.15),
                (bb_norm, 0.10),
                (strike_norm, 0.10),
                (hh_norm, 0.075),
                (barrel_norm, 0.075),
            ]

        impact = _weighted_score(components)
        rows.append({
            'player_name': str(name),
            'role': role,
            'impact_score': round(impact, 2),
            'pitches': pitches,
            'avg_velo': safe_display_value(avg_velo, '.1f'),
            'whiff_pct': safe_display_value((whiff_rate * 100) if whiff_rate is not None else None, '.1f'),
            'k_rate': safe_display_value(k_rate, '.3f'),
            'bb_rate': safe_display_value(bb_rate, '.3f'),
            'woba_allowed': safe_display_value(woba_allowed, '.3f'),
            'avg_ev_allowed': safe_display_value(avg_ev_allowed, '.1f'),
        })

    if not rows:
        return pd.DataFrame(columns=cols_out)
    return pd.DataFrame(rows).sort_values('impact_score', ascending=False).reset_index(drop=True)[cols_out]


# ---------------------------------------------------------------------------
# New functions: team record / schedule analysis
# ---------------------------------------------------------------------------

def get_team_last_n_record(season_df: pd.DataFrame, team_name: str, n_games: int = 10) -> dict[str, Any]:
    """Return win/loss record for last n_games completed games."""
    _empty = {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'sample_size': 0, 'games_played': 0}
    if season_df.empty or not team_name:
        return _empty
    games = _team_games(season_df, team_name)
    finals = games[games['is_final']] if not games.empty else games
    if finals.empty:
        return _empty
    last_n = finals.tail(n_games)
    wins = int((last_n['result'] == 'W').sum())
    losses = int((last_n['result'] == 'L').sum())
    played = len(last_n)
    return {
        'wins': wins,
        'losses': losses,
        'win_pct': round(wins / played, 3) if played else 0.0,
        'sample_size': n_games,
        'games_played': played,
    }


def get_next_three_opponents(season_df: pd.DataFrame, team_name: str, from_date_str: str) -> list[dict[str, Any]]:
    """Return next 3 scheduled (not Final) games for team_name after from_date_str."""
    if season_df.empty or not team_name:
        return []
    df = season_df[(season_df['away'] == team_name) | (season_df['home'] == team_name)].copy()
    if df.empty:
        return []
    # Filter out Final games
    df = df[~df['status'].astype(str).str.contains('Final', case=False, na=False)]
    # Filter to games after from_date_str
    date_col = 'officialDate' if 'officialDate' in df.columns else ('gameDate' if 'gameDate' in df.columns else None)
    if date_col:
        df[date_col] = df[date_col].astype(str)
        df = df[df[date_col] > str(from_date_str)]
        df = df.sort_values(date_col)
    result = []
    for _, row in df.head(3).iterrows():
        is_home = row.get('home') == team_name
        opponent = row.get('away') if is_home else row.get('home')
        result.append({
            'team_name': str(opponent),
            'date': str(row.get(date_col, '')) if date_col else '',
            'is_home': bool(is_home),
            'game_pk': row.get('game_pk', row.get('gamePk', None)),
        })
    return result


def get_team_trend_snapshot(season_df: pd.DataFrame, team_name: str, games_window: int = 30) -> dict[str, Any]:
    """Rolling stats for team over last games_window completed games."""
    _empty = {
        'team': team_name,
        'win_pct': 0.0, 'avg_runs_for': 0.0, 'avg_runs_against': 0.0,
        'run_diff_per_game': 0.0, 'record': '0-0', 'sample_size': 0,
    }
    if season_df.empty or not team_name:
        return _empty
    games = _team_games(season_df, team_name)
    finals = games[games['is_final']] if not games.empty else games
    if finals.empty:
        return _empty
    window = finals.tail(games_window)
    wins = int((window['result'] == 'W').sum())
    losses = int((window['result'] == 'L').sum())
    played = len(window)
    avg_rf = float(window['team_runs'].mean()) if played else 0.0
    avg_ra = float(window['opp_runs'].mean()) if played else 0.0
    return {
        'team': team_name,
        'win_pct': round(wins / played, 3) if played else 0.0,
        'avg_runs_for': round(avg_rf, 2),
        'avg_runs_against': round(avg_ra, 2),
        'run_diff_per_game': round(avg_rf - avg_ra, 2),
        'record': format_record(wins, losses),
        'sample_size': played,
    }


def compare_teams(season_df: pd.DataFrame, team_a_name: str, team_b_name: str, games_window: int = 30) -> pd.DataFrame:
    """
    Compare two teams over last games_window completed games.
    Returns Arrow-safe DataFrame with cols: Metric, Team_A, Team_B, Edge.
    """
    def _team_stats(team_name: str) -> dict[str, Any]:
        games = _team_games(season_df, team_name)
        finals = games[games['is_final']] if not games.empty else games
        window = finals.tail(games_window) if not finals.empty else finals
        if window.empty:
            return {'win_pct': 0.0, 'avg_rf': 0.0, 'avg_ra': 0.0, 'run_diff': 0.0,
                    'home_wpct': 0.0, 'away_wpct': 0.0}
        played = len(window)
        wins = (window['result'] == 'W').sum()
        home = window[window['location'] == 'Home']
        away = window[window['location'] == 'Away']
        hw = (home['result'] == 'W').sum()
        aw = (away['result'] == 'W').sum()
        return {
            'win_pct': round(wins / played, 3) if played else 0.0,
            'avg_rf': round(float(window['team_runs'].mean()), 2) if played else 0.0,
            'avg_ra': round(float(window['opp_runs'].mean()), 2) if played else 0.0,
            'run_diff': round(float(window['run_diff'].mean()), 2) if played else 0.0,
            'home_wpct': round(hw / len(home), 3) if len(home) else 0.0,
            'away_wpct': round(aw / len(away), 3) if len(away) else 0.0,
        }

    a = _team_stats(team_a_name)
    b = _team_stats(team_b_name)

    def _edge(a_val: float, b_val: float, higher_is_better: bool = True) -> str:
        if a_val == b_val:
            return '—'
        if higher_is_better:
            return 'A' if a_val > b_val else 'B'
        return 'A' if a_val < b_val else 'B'

    metrics = [
        ('Win %', 'win_pct', True),
        ('Runs/G', 'avg_rf', True),
        ('Runs Allowed/G', 'avg_ra', False),
        ('Run Diff/G', 'run_diff', True),
        ('Home W%', 'home_wpct', True),
        ('Away W%', 'away_wpct', True),
    ]
    rows = []
    for label, key, higher_better in metrics:
        av = a[key]
        bv = b[key]
        rows.append({
            'Metric': str(label),
            'Team_A': str(av),
            'Team_B': str(bv),
            'Edge': _edge(float(av), float(bv), higher_better),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# New functions: Statcast analytics
# ---------------------------------------------------------------------------

def get_home_run_distance_summary(statcast_batter_df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract HR events and return home/away split distance/EV summary.
    Uses 'inning_topbot' if present ('Top'=away batter, 'Bot'=home batter).
    """
    _zero: dict[str, Any] = {
        'home_hr_count': 0, 'home_avg_distance': 0.0, 'home_avg_ev': 0.0,
        'away_hr_count': 0, 'away_avg_distance': 0.0, 'away_avg_ev': 0.0,
        'total_hr_count': 0, 'total_avg_distance': 0.0,
    }
    if statcast_batter_df.empty or 'events' not in statcast_batter_df.columns:
        return _zero

    hrs = statcast_batter_df[statcast_batter_df['events'].astype(str) == 'home_run'].copy()
    if hrs.empty:
        return _zero

    total_count = len(hrs)
    dist_col = 'hit_distance_sc' if 'hit_distance_sc' in hrs.columns else None
    ev_col = 'launch_speed' if 'launch_speed' in hrs.columns else None

    def _safe_mean(series: pd.Series | None) -> float:
        if series is None or series.empty:
            return 0.0
        v = pd.to_numeric(series, errors='coerce').dropna()
        return round(float(v.mean()), 1) if len(v) else 0.0

    total_avg_dist = _safe_mean(hrs[dist_col] if dist_col else None)

    # Split by inning_topbot: 'Top' = away batter batting, 'Bot' = home batter batting
    if 'inning_topbot' in hrs.columns:
        home_hrs = hrs[hrs['inning_topbot'].astype(str).str.lower() == 'bot']
        away_hrs = hrs[hrs['inning_topbot'].astype(str).str.lower() == 'top']
        return {
            'home_hr_count': len(home_hrs),
            'home_avg_distance': _safe_mean(home_hrs[dist_col] if dist_col else None),
            'home_avg_ev': _safe_mean(home_hrs[ev_col] if ev_col else None),
            'away_hr_count': len(away_hrs),
            'away_avg_distance': _safe_mean(away_hrs[dist_col] if dist_col else None),
            'away_avg_ev': _safe_mean(away_hrs[ev_col] if ev_col else None),
            'total_hr_count': total_count,
            'total_avg_distance': total_avg_dist,
        }

    # Fallback: no split available
    return {
        'home_hr_count': 0,
        'home_avg_distance': 0.0,
        'home_avg_ev': 0.0,
        'away_hr_count': 0,
        'away_avg_distance': 0.0,
        'away_avg_ev': 0.0,
        'total_hr_count': total_count,
        'total_avg_distance': total_avg_dist,
    }


def build_spray_chart_last_30(statcast_batter_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Filter statcast data to last `days` days and return spray chart coordinates.
    Converts hc_x/hc_y to centered baseball field coordinates.
    Uses real hit_distance_sc when available; estimates distance from coordinates otherwise.
    """
    cols_out = ['field_x', 'field_y', 'events', 'player_name', 'launch_speed', 'hit_distance_sc',
                'game_date', 'distance_source', 'inning', 'pitcher_name', 'opponent']
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=cols_out)

    df = statcast_batter_df.copy()

    # Ensure player_name column exists
    df['player_name'] = _player_name_series(df)

    # Filter by date
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=days)
        df = df[df['game_date'] >= cutoff]

    # Require hc_x and hc_y
    if 'hc_x' not in df.columns or 'hc_y' not in df.columns:
        return pd.DataFrame(columns=cols_out)

    df = df.dropna(subset=['hc_x', 'hc_y'])
    if df.empty:
        return pd.DataFrame(columns=cols_out)

    df['hc_x'] = pd.to_numeric(df['hc_x'], errors='coerce')
    df['hc_y'] = pd.to_numeric(df['hc_y'], errors='coerce')
    df = df.dropna(subset=['hc_x', 'hc_y'])

    # Convert to centered baseball field coordinates
    df['field_x'] = df['hc_x'] - 125.42
    df['field_y'] = 198.27 - df['hc_y']

    # Distance: prefer Statcast hit_distance_sc, fallback to estimated from coordinates
    if 'hit_distance_sc' in df.columns:
        df['hit_distance_sc'] = pd.to_numeric(df['hit_distance_sc'], errors='coerce')
        has_real = df['hit_distance_sc'].notna() & (df['hit_distance_sc'] > 0)
        # Estimate from field coords for rows missing real distance
        estimated = np.sqrt(df['field_x'] ** 2 + df['field_y'] ** 2) * 2.5
        df.loc[~has_real, 'hit_distance_sc'] = estimated[~has_real].round(0)
        df['distance_source'] = 'Statcast'
        df.loc[~has_real, 'distance_source'] = 'Estimated'
    else:
        df['hit_distance_sc'] = (np.sqrt(df['field_x'] ** 2 + df['field_y'] ** 2) * 2.5).round(0)
        df['distance_source'] = 'Estimated'

    # Fill optional columns
    for col in ('events', 'launch_speed'):
        if col not in df.columns:
            df[col] = None

    # Add inning and pitcher info for hover
    df['inning'] = pd.to_numeric(df.get('inning', pd.Series(dtype=float)), errors='coerce') if 'inning' in df.columns else np.nan
    if 'pitcher' in df.columns:
        df['pitcher_name'] = df['pitcher'].astype(str)
    elif 'pitcher_name' not in df.columns:
        df['pitcher_name'] = ''

    # Opponent
    if 'opponent' in df.columns:
        pass
    elif 'home_team' in df.columns and 'away_team' in df.columns and 'inning_topbot' in df.columns:
        df['opponent'] = df.apply(
            lambda r: r.get('away_team', '') if str(r.get('inning_topbot', '')).lower() == 'bot'
            else r.get('home_team', ''), axis=1
        )
    else:
        df['opponent'] = ''

    df['game_date'] = df['game_date'].astype(str) if 'game_date' in df.columns else ''

    return df[cols_out].reset_index(drop=True)


# ---------------------------------------------------------------------------
# New function: Monte Carlo matchup simulation
# ---------------------------------------------------------------------------

def run_monte_carlo_matchup(
    team_a_snapshot: dict[str, Any],
    team_b_snapshot: dict[str, Any],
    n_sims: int = 10_000,
    home_team: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Poisson-based Monte Carlo simulation for a two-team matchup.
    Both snapshots should have: avg_runs_for, avg_runs_against, win_pct, sample_size.
    """
    _minimal_err: dict[str, Any] = {
        'team_a_win_pct': 0.5,
        'team_b_win_pct': 0.5,
        'avg_score_a': 0.0,
        'avg_score_b': 0.0,
        'median_score_a': 0.0,
        'median_score_b': 0.0,
        'expected_run_diff': 0.0,
        'p10_a': 0.0, 'p90_a': 0.0,
        'p10_b': 0.0, 'p90_b': 0.0,
        'n_sims': 0,
        'lambda_a': 0.0,
        'lambda_b': 0.0,
        'model_inputs_used': [],
        'note': 'Insufficient sample size.',
    }

    sa_n = coerce_int(team_a_snapshot.get('sample_size', 0), 0)
    sb_n = coerce_int(team_b_snapshot.get('sample_size', 0), 0)
    if sa_n < 5 or sb_n < 5:
        return _minimal_err

    rf_a = coerce_float(team_a_snapshot.get('avg_runs_for', 0.0), 0.0)
    ra_a = coerce_float(team_a_snapshot.get('avg_runs_against', 0.0), 0.0)
    rf_b = coerce_float(team_b_snapshot.get('avg_runs_for', 0.0), 0.0)
    ra_b = coerce_float(team_b_snapshot.get('avg_runs_against', 0.0), 0.0)

    inputs_used: list[str] = []
    if rf_a > 0:
        inputs_used.append('avg_runs_for_a')
    if ra_a > 0:
        inputs_used.append('avg_runs_against_a')
    if rf_b > 0:
        inputs_used.append('avg_runs_for_b')
    if ra_b > 0:
        inputs_used.append('avg_runs_against_b')

    # Blend offense vs opponent defense; fallback to league average if zero.
    # 4.5 R/G is the approximate MLB-wide average over the 2020s era.
    _league_avg = 4.5
    safe_rf_a = rf_a if rf_a > 0 else _league_avg
    safe_ra_b = ra_b if ra_b > 0 else _league_avg
    safe_rf_b = rf_b if rf_b > 0 else _league_avg
    safe_ra_a = ra_a if ra_a > 0 else _league_avg

    lambda_a = (safe_rf_a + safe_ra_b) / 2
    lambda_b = (safe_rf_b + safe_ra_a) / 2

    # Home field advantage
    if home_team is not None:
        a_name = str(team_a_snapshot.get('team', team_a_snapshot.get('team_name', 'A')))
        if str(home_team) == a_name:
            lambda_a *= 1.04
            inputs_used.append('home_field_a')
        else:
            lambda_b *= 1.04
            inputs_used.append('home_field_b')

    rng = np.random.default_rng(seed=seed)
    scores_a = rng.poisson(lambda_a, size=n_sims).astype(float)
    scores_b = rng.poisson(lambda_b, size=n_sims).astype(float)

    # Handle ties: one extra-innings re-roll
    ties = scores_a == scores_b
    if ties.any():
        xi_a = rng.poisson(lambda_a, size=int(ties.sum())).astype(float)
        xi_b = rng.poisson(lambda_b, size=int(ties.sum())).astype(float)
        scores_a[ties] += xi_a
        scores_b[ties] += xi_b

    wins_a = int((scores_a > scores_b).sum())
    wins_b = int((scores_b > scores_a).sum())
    remaining = n_sims - wins_a - wins_b
    # Distribute remaining ties as 50/50
    wins_a += remaining // 2
    wins_b += remaining - remaining // 2

    return {
        'team_a_win_pct': round(wins_a / n_sims, 4),
        'team_b_win_pct': round(wins_b / n_sims, 4),
        'avg_score_a': round(float(scores_a.mean()), 2),
        'avg_score_b': round(float(scores_b.mean()), 2),
        'median_score_a': round(float(np.median(scores_a)), 2),
        'median_score_b': round(float(np.median(scores_b)), 2),
        'expected_run_diff': round(float(scores_a.mean() - scores_b.mean()), 2),
        'p10_a': round(float(np.percentile(scores_a, 10)), 2),
        'p90_a': round(float(np.percentile(scores_a, 90)), 2),
        'p10_b': round(float(np.percentile(scores_b, 10)), 2),
        'p90_b': round(float(np.percentile(scores_b, 90)), 2),
        'n_sims': n_sims,
        'lambda_a': round(lambda_a, 4),
        'lambda_b': round(lambda_b, 4),
        'model_inputs_used': inputs_used,
    }


# ---------------------------------------------------------------------------
# Player of the Game  –  WAR-style Game Impact Score (GIS) model
# ---------------------------------------------------------------------------

# Standard RE24 run expectancy matrix: {(runners_code, outs): expected_runs}
# runners_code: '---'=empty, '1--'=1st, '-2-'=2nd, '--3'=3rd, '12-'=1st&2nd, etc.
# Based on 2010-2015 MLB averages.
_RE24_MATRIX = {
    ('---', 0): 0.481, ('---', 1): 0.254, ('---', 2): 0.098,
    ('1--', 0): 0.859, ('1--', 1): 0.509, ('1--', 2): 0.224,
    ('-2-', 0): 1.100, ('-2-', 1): 0.664, ('-2-', 2): 0.319,
    ('--3', 0): 1.357, ('--3', 1): 0.950, ('--3', 2): 0.353,
    ('12-', 0): 1.437, ('12-', 1): 0.884, ('12-', 2): 0.429,
    ('1-3', 0): 1.798, ('1-3', 1): 1.140, ('1-3', 2): 0.494,
    ('-23', 0): 1.920, ('-23', 1): 1.352, ('-23', 2): 0.570,
    ('123', 0): 2.282, ('123', 1): 1.520, ('123', 2): 0.736,
}

# Positional defense premiums (abbreviation -> normalized 0-1 bonus)
_DEFENSE_PREMIUM: dict[str, float] = {
    'C': 0.8, 'SS': 0.7, 'CF': 0.6, '2B': 0.4, '3B': 0.3,
    'RF': 0.2, 'LF': 0.1, '1B': 0.05, 'DH': 0.0,
}


def _ordinal(n: int) -> str:
    """Return an integer with its ordinal suffix, e.g. 1 -> '1st'."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{('th','st','nd','rd')[min(n % 10, 4)] if n % 10 < 4 else 'th'}"


def _encode_runners(play: dict) -> str:
    """Convert play's runner / matchup data to a runners code like '1-3' or '---'."""
    matchup = play.get('matchup', {})
    on_first = matchup.get('postOnFirst') is not None
    on_second = matchup.get('postOnSecond') is not None
    on_third = matchup.get('postOnThird') is not None

    # Fallback: infer from runners list when matchup fields are absent
    if not (on_first or on_second or on_third):
        for r in play.get('runners', []):
            start = (r.get('movement') or {}).get('start', '')
            if start == '1B':
                on_first = True
            elif start == '2B':
                on_second = True
            elif start == '3B':
                on_third = True

    code = ('1' if on_first else '-') + ('2' if on_second else '-') + ('3' if on_third else '-')
    return code


def _calc_re24_swing(play: dict) -> float:
    """Compute the RE24 run-expectancy swing for a single play.

    RE24 swing = (runs_scored_on_play + RE_after) - RE_before
    """
    about = play.get('about', {})
    outs_before = coerce_int(about.get('outs'), 0)

    # Runs scored on this play
    runners = play.get('runners', [])
    runs_scored = sum(
        1 for r in runners
        if (r.get('movement') or {}).get('end', '') == 'score'
    )

    # State before the play
    runners_before = _encode_runners(play)
    re_before = _RE24_MATRIX.get((runners_before, min(outs_before, 2)), 0.0)

    # State after – derive from runner movements
    on1 = on2 = on3 = False
    outs_added = 0
    for r in runners:
        mov = r.get('movement') or {}
        end = mov.get('end', '')
        is_out = mov.get('isOut', False)
        if is_out:
            outs_added += 1
        elif end == '1B':
            on1 = True
        elif end == '2B':
            on2 = True
        elif end == '3B':
            on3 = True

    outs_after = min(outs_before + outs_added, 3)
    if outs_after >= 3:
        re_after = 0.0
    else:
        code_after = ('1' if on1 else '-') + ('2' if on2 else '-') + ('3' if on3 else '-')
        re_after = _RE24_MATRIX.get((code_after, outs_after), 0.0)

    return round(runs_scored + re_after - re_before, 4)


def _accumulate_player_impact(all_plays: list, winning_side: str) -> dict:
    """Accumulate WPA and RE24 for each player across all plays.

    *winning_side* is ``'home'`` or ``'away'``.

    Returns dict keyed by player_id (int) with::

        wpa:            float  – total WPA from the perspective of that player's team winning
        re24:           float  – total RE24 impact
        key_event:      str    – description of their highest-|WPA| play
        key_wpa:        float  – WPA of that single play
        negative_plays: int    – count of plays with negative WPA
        late_positive:  bool   – had a positive-WPA play in inning >= 7
        team_side:      str    – 'home' or 'away'
    """

    accum: dict[int, dict] = {}

    def _ensure(pid: int, side: str) -> dict:
        if pid not in accum:
            accum[pid] = {
                'wpa': 0.0, 're24': 0.0,
                'key_event': '', 'key_wpa': 0.0,
                'negative_plays': 0, 'late_positive': False,
                'team_side': side,
            }
        return accum[pid]

    for play in all_plays:
        result = play.get('result', {})
        event = result.get('event', '')
        if not event:
            continue

        about = play.get('about', {})
        matchup = play.get('matchup', {})
        half = about.get('halfInning', 'top')
        inning = coerce_int(about.get('inning'), 1)
        batter_side = 'away' if half == 'top' else 'home'
        pitcher_side = 'home' if half == 'top' else 'away'

        batter_id = coerce_int(matchup.get('batter', {}).get('id'), 0)
        pitcher_id = coerce_int(matchup.get('pitcher', {}).get('id'), 0)
        if batter_id == 0 and pitcher_id == 0:
            continue

        # WPA from the home team's perspective
        raw_wpa: float | None = None
        hw = about.get('homeWinProbabilityAdded')
        if hw is not None:
            try:
                raw_wpa = float(hw)
            except (TypeError, ValueError):
                raw_wpa = None

        # If unavailable, approximate from fallback leverage
        if raw_wpa is None:
            fb = _calc_fallback_leverage(play) / 100.0
            raw_wpa = fb if batter_side == 'home' else -fb

        re24 = _calc_re24_swing(play)

        # Translate WPA to winning-side perspective
        if winning_side == 'home':
            home_wpa = raw_wpa
        else:
            home_wpa = -raw_wpa

        batter_wpa = home_wpa if batter_side == winning_side else -home_wpa
        pitcher_wpa = -batter_wpa  # zero-sum between batter and pitcher

        desc = result.get('description', event.replace('_', ' '))

        # Credit batter
        if batter_id:
            b = _ensure(batter_id, batter_side)
            b['wpa'] += batter_wpa
            b['re24'] += re24
            if batter_wpa < 0:
                b['negative_plays'] += 1
            if batter_wpa > 0 and inning >= 7:
                b['late_positive'] = True
            if abs(batter_wpa) > abs(b['key_wpa']):
                b['key_wpa'] = batter_wpa
                b['key_event'] = desc

        # Credit pitcher (RE24 inverted: runs prevented = positive)
        if pitcher_id:
            p = _ensure(pitcher_id, pitcher_side)
            p['wpa'] += pitcher_wpa
            p['re24'] += -re24
            if pitcher_wpa < 0:
                p['negative_plays'] += 1
            if pitcher_wpa > 0 and inning >= 7:
                p['late_positive'] = True
            if abs(pitcher_wpa) > abs(p['key_wpa']):
                p['key_wpa'] = pitcher_wpa
                p['key_event'] = desc

    return accum


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _parse_ip(ip_str: str) -> float:
    """Parse an innings-pitched string like '6.2' into a float (6.667)."""
    try:
        parts = str(ip_str).split('.')
        whole = float(parts[0])
        frac = float(parts[1]) / 3.0 if len(parts) > 1 else 0.0
        return whole + frac
    except (ValueError, IndexError):
        return 0.0


# ---- GIS: batters ----------------------------------------------------------

def _compute_batter_gis(player_data: dict, player_id: int,
                         impact_accum: dict, boxscore_stats: dict) -> dict:
    """Compute Game Impact Score for a hitter."""
    acc = impact_accum.get(player_id, {})
    wpa = acc.get('wpa', 0.0)
    re24 = acc.get('re24', 0.0)

    stats = boxscore_stats

    hits = coerce_int(stats.get('hits'), 0)
    hr = coerce_int(stats.get('homeRuns'), 0)
    triples = coerce_int(stats.get('triples'), 0)
    doubles = coerce_int(stats.get('doubles'), 0)
    singles = max(0, hits - hr - triples - doubles)
    ab = coerce_int(stats.get('atBats'), 0)
    bb = coerce_int(stats.get('baseOnBalls'), 0)
    hbp = coerce_int(stats.get('hitByPitch'), 0)
    sb = coerce_int(stats.get('stolenBases'), 0)
    cs = coerce_int(stats.get('caughtStealing'), 0)

    # Offensive run value (linear weights)
    orv = (singles * 0.46 + doubles * 0.76 + triples * 1.06
           + hr * 1.40 + (bb + hbp) * 0.33
           - max(0, ab - hits) * 0.25)

    # Baserunning run value
    brv = sb * 0.20 - cs * 0.41

    # Defensive positional premium
    pos = player_data.get('position', {}).get('abbreviation', '')
    defense = _DEFENSE_PREMIUM.get(pos, 0.0)

    # Normalize each component
    norm_wpa = _clamp(wpa / 0.3)
    norm_re24 = _clamp(re24 / 3.0)
    norm_off = _clamp(orv / 4.0)
    norm_br = _clamp(brv / 0.5)
    norm_def = _clamp(defense, 0.0, 1.0)

    gis = (0.55 * norm_wpa + 0.25 * norm_re24 + 0.10 * norm_off
           + 0.05 * norm_br + 0.05 * norm_def)

    return {
        'gis': round(gis, 4),
        'wpa': round(wpa, 4),
        're24': round(re24, 4),
        'offensive_rv': round(orv, 4),
        'baserunning_rv': round(brv, 4),
        'defense_bonus': round(defense, 2),
        'key_event': acc.get('key_event', ''),
        'negative_plays': acc.get('negative_plays', 0),
        'late_positive': acc.get('late_positive', False),
    }


# ---- GIS: pitchers ---------------------------------------------------------

def _compute_pitcher_gis(player_data: dict, player_id: int,
                          impact_accum: dict, boxscore_stats: dict,
                          is_starter: bool) -> dict:
    """Compute Game Impact Score for a pitcher."""
    acc = impact_accum.get(player_id, {})
    wpa = acc.get('wpa', 0.0)
    re24 = acc.get('re24', 0.0)

    stats = boxscore_stats
    ip = _parse_ip(str(stats.get('inningsPitched', '0')))
    k = coerce_int(stats.get('strikeOuts'), 0)
    er = coerce_int(stats.get('earnedRuns'), 0)
    h = coerce_int(stats.get('hits'), 0)
    bb = coerce_int(stats.get('baseOnBalls'), 0)
    hr_allowed = coerce_int(stats.get('homeRuns'), 0)

    # Pitching run value
    prv = ip * 0.7 + k * 0.15 - er * 1.0 - (h + bb) * 0.15
    prv_max = 6.0 if is_starter else 2.0

    # Leverage / workload bonus (0-1 scale)
    lev = 0.0
    if is_starter:
        if ip >= 6.0 and er <= 3:
            lev += 0.5  # quality start
        if ip >= 7.0:
            lev += 0.3  # deep outing
    else:
        saves = coerce_int(stats.get('saves'), 0)
        holds = coerce_int(stats.get('holds'), 0)
        if saves > 0:
            lev += 0.5
        if holds > 0:
            lev += 0.3
        if ip >= 1.0 and er == 0:
            lev += 0.2  # clean outing

    # Run prevention context (0-1 scale)
    rpc = 0.0
    if is_starter and er == 0 and h <= 3:
        rpc += 0.5
    if bb > 0 and k >= 3 * bb:
        rpc += 0.3
    elif bb == 0 and k > 0:
        rpc += 0.4

    # Normalize
    norm_wpa = _clamp(wpa / 0.3)
    norm_re24 = _clamp(re24 / 3.0)
    norm_prv = _clamp(prv / prv_max) if prv_max else 0.0
    norm_lev = _clamp(lev, 0.0, 1.0)
    norm_rpc = _clamp(rpc, 0.0, 1.0)

    gis = (0.55 * norm_wpa + 0.25 * norm_re24 + 0.10 * norm_prv
           + 0.05 * norm_lev + 0.05 * norm_rpc)

    return {
        'gis': round(gis, 4),
        'wpa': round(wpa, 4),
        're24': round(re24, 4),
        'pitching_rv': round(prv, 4),
        'leverage_bonus': round(lev, 2),
        'run_prevention': round(rpc, 2),
        'key_event': acc.get('key_event', ''),
        'negative_plays': acc.get('negative_plays', 0),
        'late_positive': acc.get('late_positive', False),
    }


# ---- Summary formatters ----------------------------------------------------

def _format_batter_summary(player: dict[str, Any]) -> str:
    """Full batter stat line: H-AB, R, HR, RBI, BB, K, SB, TB."""
    stats = player.get('stats', {}).get('batting', {})
    ab = coerce_int(stats.get('atBats'), 0)
    h = coerce_int(stats.get('hits'), 0)
    r = coerce_int(stats.get('runs'), 0)
    hr = coerce_int(stats.get('homeRuns'), 0)
    rbi = coerce_int(stats.get('rbi'), 0)
    bb = coerce_int(stats.get('baseOnBalls'), 0)
    k = coerce_int(stats.get('strikeOuts'), 0)
    sb = coerce_int(stats.get('stolenBases'), 0)
    doubles = coerce_int(stats.get('doubles'), 0)
    triples = coerce_int(stats.get('triples'), 0)
    singles = max(0, h - hr - triples - doubles)
    tb = singles + doubles * 2 + triples * 3 + hr * 4
    return (f"{h}-{ab}, {r} R, {hr} HR, {rbi} RBI, "
            f"{bb} BB, {k} K, {sb} SB, {tb} TB")


def _format_pitcher_summary(player: dict[str, Any]) -> str:
    """Full pitcher stat line: IP, H, R, ER, BB, K, HR, P-S."""
    stats = player.get('stats', {}).get('pitching', {})
    ip = stats.get('inningsPitched', '0')
    h = coerce_int(stats.get('hits'), 0)
    r = coerce_int(stats.get('runs'), 0)
    er = coerce_int(stats.get('earnedRuns'), 0)
    bb = coerce_int(stats.get('baseOnBalls'), 0)
    k = coerce_int(stats.get('strikeOuts'), 0)
    hr = coerce_int(stats.get('homeRuns'), 0)
    raw_pitches = stats.get('pitchesThrown')
    if raw_pitches is None:
        raw_pitches = stats.get('numberOfPitches')
    pitches = coerce_int(raw_pitches, 0)
    strikes = coerce_int(stats.get('strikes'), 0)
    saves = coerce_int(stats.get('saves'), 0)
    holds = coerce_int(stats.get('holds'), 0)
    line = (f"{ip} IP, {h} H, {r} R, {er} ER, {bb} BB, "
            f"{k} K, {hr} HR, {pitches}-{strikes} P-S")
    if saves > 0:
        line += f", {saves} SV"
    if holds > 0:
        line += f", {holds} HLD"
    return line


# ---- Player of the Game (main entry point) ---------------------------------

def build_player_of_game(live_feed_data: dict[str, Any]) -> dict[str, Any] | None:
    """Identify the Player of the Game using a WAR-style Game Impact Score.

    Returns dict with backward-compatible keys (``score``, ``summary``, …) plus
    new GIS-related fields, or ``None`` if data is insufficient.
    """
    if not live_feed_data:
        return None
    live_data = live_feed_data.get('liveData', {})
    boxscore = live_data.get('boxscore', {})
    if not boxscore:
        return None

    game_data = live_feed_data.get('gameData', {})
    teams_info = game_data.get('teams', {})
    probable = game_data.get('probablePitchers', {})
    away_starter_id = coerce_int((probable.get('away') or {}).get('id'), 0)
    home_starter_id = coerce_int((probable.get('home') or {}).get('id'), 0)

    # Determine winning side from linescore / final score
    linescore = live_data.get('linescore', {})
    home_runs_total = coerce_int(
        linescore.get('teams', {}).get('home', {}).get('runs'), 0)
    away_runs_total = coerce_int(
        linescore.get('teams', {}).get('away', {}).get('runs'), 0)
    winning_side = 'home' if home_runs_total >= away_runs_total else 'away'

    # Accumulate per-play WPA / RE24 across all plays
    all_plays = live_data.get('plays', {}).get('allPlays', [])
    impact = _accumulate_player_impact(all_plays, winning_side)

    candidates: list[dict[str, Any]] = []

    for side in ('away', 'home'):
        team_box = boxscore.get('teams', {}).get(side, {})
        team_name = teams_info.get(side, {}).get('name', side.title())
        players = team_box.get('players', {})
        pitcher_order = team_box.get('pitchers', [])
        first_pitcher_id = pitcher_order[0] if pitcher_order else None

        for pid_key, pdata in players.items():
            pid = coerce_int(pdata.get('person', {}).get('id'), 0)
            name = pdata.get('person', {}).get('fullName', 'Unknown')

            # Batting GIS
            bat_stats = pdata.get('stats', {}).get('batting', {})
            if bat_stats and coerce_int(bat_stats.get('atBats'), 0) > 0:
                gis_info = _compute_batter_gis(pdata, pid, impact, bat_stats)
                candidates.append({
                    'player_name': name,
                    'player_id': pid,
                    'team': team_name,
                    'role_type': 'Hitter',
                    'summary': _format_batter_summary(pdata),
                    '_gis': gis_info,
                })

            # Pitching GIS
            pit_stats = pdata.get('stats', {}).get('pitching', {})
            if pit_stats and pit_stats.get('inningsPitched') is not None:
                is_starter = (pid == away_starter_id or pid == home_starter_id
                              or (first_pitcher_id and pid == first_pitcher_id))
                role = 'Starter' if is_starter else 'Reliever'
                gis_info = _compute_pitcher_gis(
                    pdata, pid, impact, pit_stats, bool(is_starter))
                candidates.append({
                    'player_name': name,
                    'player_id': pid,
                    'team': team_name,
                    'role_type': role,
                    'summary': _format_pitcher_summary(pdata),
                    '_gis': gis_info,
                })

    if not candidates:
        return None

    # Rank: GIS (desc) → WPA → RE24 → late_positive → fewer negative plays
    candidates.sort(key=lambda c: (
        c['_gis']['gis'],
        c['_gis']['wpa'],
        c['_gis']['re24'],
        1 if c['_gis']['late_positive'] else 0,
        -c['_gis']['negative_plays'],
    ), reverse=True)

    best = candidates[0]
    gi = best['_gis']
    pid = best['player_id']
    headshot_url = (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:67:current.png/"
        f"w_213,q_auto:best/v1/people/{pid}/headshot/67/current"
    )

    return {
        'player_name': best['player_name'],
        'player_id': pid,
        'team': best['team'],
        'role_type': best['role_type'],
        'headshot_url': headshot_url,
        'summary': best['summary'],
        'game_impact_score': gi['gis'],
        'wpa': gi['wpa'],
        're24': gi['re24'],
        'key_event': gi['key_event'],
        'explanation': (
            "Won Player of the Game because he produced the highest "
            "Game Impact Score in the game."
        ),
        # Backward-compatible key
        'score': gi['gis'],
    }


# ---------------------------------------------------------------------------
# Play of the Game  –  WPA / RE24 based ranking
# ---------------------------------------------------------------------------

def _calc_fallback_leverage(play: dict[str, Any], total_innings: int = 9) -> float:
    """Calculate a fallback leverage + RE24-style score when WPA is unavailable."""
    about = play.get('about', {})
    inning = coerce_int(about.get('inning'), 1)
    outs = coerce_int(about.get('outs'), 0)
    result = play.get('result', {})
    event = result.get('event', '')

    away_score = coerce_int(result.get('awayScore'), 0)
    home_score = coerce_int(result.get('homeScore'), 0)
    diff = abs(away_score - home_score)

    runners = play.get('runners', [])
    runners_on = len([r for r in runners if (r.get('movement') or {}).get('start')])
    rbi = coerce_int(result.get('rbi'), 0)

    inning_factor = min(inning / total_innings, 1.5)

    if diff <= 1:
        close_factor = 2.0
    elif diff <= 3:
        close_factor = 1.2
    else:
        close_factor = 0.5

    out_factor = 1.0 + (outs * 0.3)
    scoring_factor = rbi * 1.5 + (1.0 if event in ('home_run', 'triple', 'double') else 0.5)
    runner_factor = 1.0 + (runners_on * 0.3)

    # Incorporate RE24 swing as an additive boost
    re24 = _calc_re24_swing(play)
    re24_boost = max(0.0, re24) * 0.5

    leverage = inning_factor * close_factor * out_factor * scoring_factor * runner_factor + re24_boost
    return round(leverage, 3)


def _play_sort_key(entry: dict) -> tuple:
    """Multi-criteria sort key for play ranking (all descending)."""
    return (
        entry['wpa_swing'],
        entry['re24_swing'],
        entry['inning'],
        entry['leverage'],
        entry['score_delta'],
    )


def build_play_of_game(live_feed_data: dict[str, Any]) -> dict[str, Any] | None:
    """Find the highest-impact play from a game's live feed data.

    Ranks plays by WPA swing → RE24 swing → later inning → leverage →
    scoreboard swing.  Returns a dict with backward-compatible keys plus new
    WPA/RE24 fields and runner-up plays, or ``None`` if data is insufficient.
    """
    if not live_feed_data:
        return None
    live_data = live_feed_data.get('liveData', {})
    all_plays = live_data.get('plays', {}).get('allPlays', [])
    if not all_plays:
        return None

    # Determine winning side
    linescore = live_data.get('linescore', {})
    home_runs_total = coerce_int(
        linescore.get('teams', {}).get('home', {}).get('runs'), 0)
    away_runs_total = coerce_int(
        linescore.get('teams', {}).get('away', {}).get('runs'), 0)
    winning_side = 'home' if home_runs_total >= away_runs_total else 'away'

    scored_plays: list[dict] = []

    for play in all_plays:
        result = play.get('result', {})
        event = result.get('event', '')
        if not event:
            continue

        about = play.get('about', {})
        matchup = play.get('matchup', {})

        # WPA swing (winning-side perspective)
        raw_wpa: float | None = None
        hw = about.get('homeWinProbabilityAdded')
        if hw is not None:
            try:
                raw_wpa = float(hw)
            except (TypeError, ValueError):
                raw_wpa = None

        half = about.get('halfInning', 'top')
        if raw_wpa is not None:
            wpa_swing = raw_wpa if winning_side == 'home' else -raw_wpa
        else:
            fb = _calc_fallback_leverage(play) / 100.0
            batter_side = 'away' if half == 'top' else 'home'
            wpa_swing = fb if batter_side == winning_side else -fb

        re24_swing = _calc_re24_swing(play)
        inning = coerce_int(about.get('inning'), 0)
        leverage = _calc_fallback_leverage(play)

        away_score = coerce_int(result.get('awayScore'), 0)
        home_score = coerce_int(result.get('homeScore'), 0)
        rbi = coerce_int(result.get('rbi'), 0)

        scored_plays.append({
            'play': play,
            'wpa_swing': wpa_swing,
            're24_swing': re24_swing,
            'inning': inning,
            'leverage': leverage,
            'score_delta': rbi,
        })

    if not scored_plays:
        return None

    scored_plays.sort(key=_play_sort_key, reverse=True)

    def _build_play_dict(entry: dict, include_full: bool = False) -> dict:
        play = entry['play']
        about = play.get('about', {})
        result = play.get('result', {})
        matchup = play.get('matchup', {})

        inning = coerce_int(about.get('inning'), 0)
        half = about.get('halfInning', '')
        inning_label = f"{'Top' if half == 'top' else 'Bot'} {inning}"
        batter = matchup.get('batter', {}).get('fullName', 'Unknown')
        pitcher = matchup.get('pitcher', {}).get('fullName', 'Unknown')
        event = result.get('event', '')
        away_score = coerce_int(result.get('awayScore'), 0)
        home_score = coerce_int(result.get('homeScore'), 0)
        rbi = coerce_int(result.get('rbi'), 0)
        description = result.get('description', '')

        if half == 'top':
            score_before = f"{away_score - rbi}-{home_score}"
        else:
            score_before = f"{away_score}-{home_score - rbi}"
        score_after = f"{away_score}-{home_score}"

        outs = coerce_int(about.get('outs'), 0)
        runners = play.get('runners', [])
        runner_starts = [
            (r.get('movement') or {}).get('start', '')
            for r in runners if (r.get('movement') or {}).get('start')
        ]
        bases = [s for s in runner_starts if s and s != 'null']
        runners_before = ', '.join(bases) if bases else 'bases empty'

        # Human-readable short description (backward compat)
        parts = [inning_label]
        if outs > 0:
            parts.append(f"{outs} out{'s' if outs > 1 else ''}")
        if bases:
            parts.append(f"runner(s) on {', '.join(bases)}")
        if away_score == home_score:
            parts.append("tie game")
        parts.append(event.replace('_', ' '))
        readable = ', '.join(parts)

        d: dict[str, Any] = {
            'description': readable,
            'wpa_swing': round(entry['wpa_swing'], 4),
            're24_swing': round(entry['re24_swing'], 4),
        }

        if include_full:
            # Narrative
            half_word = 'Top' if half == 'top' else 'Bottom'
            outs_word = {0: 'no outs', 1: 'one out', 2: 'two outs'}.get(outs, f'{outs} outs')
            wp_before = about.get('homeWinProbability')
            wp_after_raw = None
            if wp_before is not None and about.get('homeWinProbabilityAdded') is not None:
                try:
                    wp_after_raw = float(wp_before) + float(about['homeWinProbabilityAdded'])
                except (TypeError, ValueError):
                    pass
            wp_part = ''
            if wp_before is not None and wp_after_raw is not None:
                wp_part = (f", flipping win expectancy from "
                           f"{coerce_float(wp_before, 0):.0f}% to {wp_after_raw:.0f}%")

            narrative = (
                f"{half_word} of the {_ordinal(inning)}, "
                f"score {score_before} with {outs_word}"
                f"{' and ' + runners_before if bases else ''}, "
                f"{batter} {event.replace('_', ' ')}"
                f"{wp_part}."
            )

            d.update({
                'inning': inning,
                'inning_label': inning_label,
                'batter': batter,
                'pitcher': pitcher,
                'event': event,
                'score_before': score_before,
                'score_after': score_after,
                'outs_before': outs,
                'runners_before': runners_before,
                'leverage_score': round(entry['leverage'], 2),
                'win_prob_delta': about.get('homeWinProbabilityAdded'),
                'narrative': narrative,
                'full_description': description,
            })
        return d

    top = _build_play_dict(scored_plays[0], include_full=True)

    # Runner-ups (2nd and 3rd best)
    runner_ups = [
        _build_play_dict(scored_plays[i])
        for i in range(1, min(3, len(scored_plays)))
    ]
    top['runner_ups'] = runner_ups

    return top


# ---------------------------------------------------------------------------
# Frozen game state persistence (JSON-backed)
# ---------------------------------------------------------------------------

import json
import os
from datetime import datetime, timezone

_FROZEN_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.frozen_states')


def _ensure_frozen_dir() -> None:
    os.makedirs(_FROZEN_STATE_DIR, exist_ok=True)


def _frozen_path(team_id: int) -> str:
    return os.path.join(_FROZEN_STATE_DIR, f'team_{team_id}.json')


def load_frozen_game_state(team_id: int) -> dict[str, Any] | None:
    """Load the most recent frozen game state for a team from disk."""
    path = _frozen_path(team_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('Failed to load frozen state for team %s: %s', team_id, exc)
        return None


def save_frozen_game_state(team_id: int, snapshot: dict[str, Any]) -> None:
    """Save a frozen game state snapshot for a team to disk."""
    _ensure_frozen_dir()
    path = _frozen_path(team_id)
    try:
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning('Failed to save frozen state for team %s: %s', team_id, exc)


def should_replace_frozen_state(old_snapshot: dict[str, Any] | None,
                                current_game: dict[str, Any]) -> bool:
    """Determine whether the frozen state should be replaced.

    Replace only when:
    - There is no old snapshot, OR
    - The current game is a different gamePk AND it has actually started
      (status is Live or Final, not Pre-Game/Scheduled).
    """
    if old_snapshot is None:
        return True
    old_pk = old_snapshot.get('gamePk')
    new_pk = current_game.get('gamePk')
    if old_pk == new_pk:
        # Same game — replace if it's now Final and wasn't before
        return (current_game.get('status') == 'Final'
                and old_snapshot.get('status') != 'Final')
    # Different game — replace only if the new game has started
    new_status = current_game.get('status', '')
    return new_status in ('Live', 'Final')


def build_frozen_snapshot(team_id: int, game_pk: int, game_date: str,
                          status: str, scorecard: dict[str, Any],
                          player_of_game: dict[str, Any] | None,
                          play_of_game: dict[str, Any] | None) -> dict[str, Any]:
    """Build a frozen snapshot dict."""
    return {
        'team_id': team_id,
        'gamePk': game_pk,
        'game_date': game_date,
        'status': status,
        'scorecard': scorecard,
        'player_of_game': player_of_game,
        'play_of_game': play_of_game,
        'frozen_at': datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Live scorecard HTML rendering helpers
# ---------------------------------------------------------------------------

def render_scorecard_html(scorecard: dict[str, Any], is_live: bool = False) -> str:
    """Build HTML/CSS for a custom scorecard matching the Caught Looking style.

    Returns a complete HTML string ready for st.markdown(unsafe_allow_html=True).
    """
    if not scorecard:
        return '<p style="color:#888;">Scorecard data unavailable.</p>'

    innings = scorecard.get('innings', [])
    away_abbr = scorecard.get('away_abbr', 'AWAY')
    home_abbr = scorecard.get('home_abbr', 'HOME')
    away_runs = scorecard.get('away_runs', 0)
    home_runs = scorecard.get('home_runs', 0)
    away_hits = scorecard.get('away_hits', 0)
    home_hits = scorecard.get('home_hits', 0)
    away_errors = scorecard.get('away_errors', 0)
    home_errors = scorecard.get('home_errors', 0)
    status = scorecard.get('status', '')
    detailed_status = scorecard.get('detailed_status', '')
    current_inning = scorecard.get('current_inning', 0)
    inning_state = scorecard.get('inning_state', '')
    inning_ordinal = scorecard.get('inning_ordinal', '')
    venue = scorecard.get('venue', '')
    game_date = scorecard.get('game_date', '')

    # Determine how many inning columns to show (min 9)
    num_innings = max(9, len(innings))

    # Status label
    if status == 'Final':
        status_label = '<span style="color:#e74c3c;font-weight:700;">Final</span>'
        if len(innings) > 9:
            status_label = f'<span style="color:#e74c3c;font-weight:700;">Final/{len(innings)}</span>'
    elif status == 'Live':
        arrow = '▲' if inning_state == 'Top' else '▼' if inning_state == 'Bottom' else ''
        status_label = f'<span style="color:#27ae60;font-weight:700;">🔴 {arrow} {inning_ordinal}</span>'
    else:
        status_label = f'<span style="color:#888;">{detailed_status or status}</span>'

    # Build inning headers
    inning_headers = ''
    for i in range(1, num_innings + 1):
        highlight = ''
        if is_live and i == current_inning:
            highlight = 'background:#2a3a4a;'
        inning_headers += f'<th style="min-width:32px;text-align:center;padding:4px 6px;font-weight:600;{highlight}">{i}</th>'

    # Build away row
    away_cells = ''
    for i in range(num_innings):
        val = ''
        highlight = ''
        if i < len(innings):
            r = innings[i].get('away_runs')
            val = str(r) if r is not None else ''
            if is_live and (i + 1) == current_inning:
                highlight = 'background:#2a3a4a;'
        away_cells += f'<td style="text-align:center;padding:4px 6px;{highlight}">{val}</td>'

    # Build home row
    home_cells = ''
    for i in range(num_innings):
        val = ''
        highlight = ''
        if i < len(innings):
            r = innings[i].get('home_runs')
            val = str(r) if r is not None else ''
            if is_live and (i + 1) == current_inning:
                highlight = 'background:#2a3a4a;'
        home_cells += f'<td style="text-align:center;padding:4px 6px;{highlight}">{val}</td>'

    # Bold the winner's total
    away_total_style = 'font-weight:700;' if away_runs > home_runs else ''
    home_total_style = 'font-weight:700;' if home_runs > away_runs else ''

    away_logo = scorecard.get('away_logo', '')
    home_logo = scorecard.get('home_logo', '')
    away_record = scorecard.get('away_record', '')
    home_record = scorecard.get('home_record', '')
    away_team = scorecard.get('away_team', '')
    home_team = scorecard.get('home_team', '')

    html = f"""
<div style="background:#0e1117;border:1px solid #333;border-radius:10px;padding:16px;margin-bottom:12px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <!-- Game Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="text-align:center;">
        <img src="{away_logo}" style="width:40px;height:40px;" alt="{away_abbr}">
        <div style="font-size:0.7rem;color:#888;">{away_record}</div>
      </div>
      <div style="font-size:0.85rem;color:#ccc;">
        <strong>{away_team}</strong>
      </div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#888;">
        {game_date} · {venue}
      </div>
      <div style="font-size:0.9rem;margin-top:2px;">
        {status_label}
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="font-size:0.85rem;color:#ccc;text-align:right;">
        <strong>{home_team}</strong>
      </div>
      <div style="text-align:center;">
        <img src="{home_logo}" style="width:40px;height:40px;" alt="{home_abbr}">
        <div style="font-size:0.7rem;color:#888;">{home_record}</div>
      </div>
    </div>
  </div>

  <!-- Scorecard Table -->
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#ddd;">
      <thead>
        <tr style="border-bottom:2px solid #444;">
          <th style="text-align:left;padding:4px 8px;min-width:60px;font-weight:700;">Team</th>
          {inning_headers}
          <th style="min-width:36px;text-align:center;padding:4px 6px;font-weight:700;background:#1a2a3a;">R</th>
          <th style="min-width:36px;text-align:center;padding:4px 6px;font-weight:700;background:#1a2a3a;">H</th>
          <th style="min-width:36px;text-align:center;padding:4px 6px;font-weight:700;background:#1a2a3a;">E</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #333;">
          <td style="padding:6px 8px;font-weight:600;">{away_abbr}</td>
          {away_cells}
          <td style="text-align:center;padding:6px;background:#1a2a3a;{away_total_style}">{away_runs}</td>
          <td style="text-align:center;padding:6px;background:#1a2a3a;">{away_hits}</td>
          <td style="text-align:center;padding:6px;background:#1a2a3a;">{away_errors}</td>
        </tr>
        <tr>
          <td style="padding:6px 8px;font-weight:600;">{home_abbr}</td>
          {home_cells}
          <td style="text-align:center;padding:6px;background:#1a2a3a;{home_total_style}">{home_runs}</td>
          <td style="text-align:center;padding:6px;background:#1a2a3a;">{home_hits}</td>
          <td style="text-align:center;padding:6px;background:#1a2a3a;">{home_errors}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>"""
    return html


def render_game_situation_html(scorecard: dict[str, Any]) -> str:
    """Render live game situation: count, outs, baserunners."""
    if not scorecard or scorecard.get('status') != 'Live':
        return ''

    balls = scorecard.get('balls', 0)
    strikes = scorecard.get('strikes', 0)
    outs = scorecard.get('outs', 0)
    runners = scorecard.get('runners', {})
    inning_state = scorecard.get('inning_state', '')
    inning_ordinal = scorecard.get('inning_ordinal', '')

    # Base indicators
    first = '🔶' if runners.get('first') else '⬜'
    second = '🔶' if runners.get('second') else '⬜'
    third = '🔶' if runners.get('third') else '⬜'

    # Outs indicators
    out_dots = '●' * outs + '○' * (3 - outs)

    html = f"""
<div style="background:#0e1117;border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:8px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="display:flex;justify-content:space-around;align-items:center;color:#ddd;font-size:0.85rem;">
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#888;margin-bottom:2px;">Count</div>
      <div><strong>{balls}-{strikes}</strong></div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#888;margin-bottom:2px;">Outs</div>
      <div style="letter-spacing:4px;color:#f39c12;">{out_dots}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#888;margin-bottom:2px;">Bases</div>
      <div style="line-height:1.4;">
        <div style="text-align:center;">{second}</div>
        <div>{third} {first}</div>
      </div>
    </div>
  </div>
</div>"""
    return html


def render_scoring_plays_html(scorecard: dict[str, Any]) -> str:
    """Render scoring plays list."""
    plays = scorecard.get('scoring_plays', [])
    if not plays:
        return ''
    rows = ''
    for sp in plays:
        inn = sp.get('inning', 0)
        half = sp.get('halfInning', '')
        half_label = '▲' if half == 'top' else '▼'
        desc = sp.get('description', '')
        score = f"{sp.get('awayScore', 0)}-{sp.get('homeScore', 0)}"
        rows += f"""
        <tr style="border-bottom:1px solid #222;">
          <td style="padding:4px 8px;color:#888;white-space:nowrap;vertical-align:top;">{half_label}{inn}</td>
          <td style="padding:4px 8px;font-size:0.8rem;">{desc}</td>
          <td style="padding:4px 8px;text-align:center;white-space:nowrap;color:#f39c12;vertical-align:top;">{score}</td>
        </tr>"""
    html = f"""
<div style="background:#0e1117;border:1px solid #333;border-radius:8px;padding:12px;margin-top:4px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="font-size:0.85rem;font-weight:700;color:#ddd;margin-bottom:8px;">Scoring Plays</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.8rem;color:#ccc;">
    {rows}
  </table>
</div>"""
    return html


# ---------------------------------------------------------------------------
# Playoff simulation helpers
# ---------------------------------------------------------------------------

_PLAYOFF_WEIGHTS = {
    'win_pct': 0.40,
    'run_diff': 0.25,
    'recent_form': 0.20,
    'home_field': 0.10,
    'head_to_head': 0.05,
}


def _matchup_win_prob(team_a: dict[str, Any], team_b: dict[str, Any],
                      home_team_name: str | None = None) -> float:
    """Estimate single-game win probability for team_a vs team_b.

    Uses weighted blend of season win_pct, run_diff, recent form, and home field.
    Returns probability that team_a wins (0.0-1.0).
    """
    wp_a = coerce_float(team_a.get('win_pct', 0.5), 0.5)
    wp_b = coerce_float(team_b.get('win_pct', 0.5), 0.5)
    rd_a = coerce_float(team_a.get('run_diff_per_game', 0.0), 0.0)
    rd_b = coerce_float(team_b.get('run_diff_per_game', 0.0), 0.0)

    # Win pct component (log5 method)
    if wp_a + wp_b > 0:
        denom = wp_a * (1 - wp_b) + wp_b * (1 - wp_a)
        log5 = (wp_a * (1 - wp_b)) / denom if denom > 0 else 0.5
    else:
        log5 = 0.5

    # Run differential component
    rd_diff = rd_a - rd_b
    rd_prob = 1 / (1 + 10 ** (-rd_diff / 3.0))

    # Recent form (win pct used as proxy)
    recent_prob = log5

    # Home field
    hfa = 0.0
    if home_team_name:
        if home_team_name == team_a.get('team_name', ''):
            hfa = 0.04
        elif home_team_name == team_b.get('team_name', ''):
            hfa = -0.04

    w = _PLAYOFF_WEIGHTS
    base_prob = (
        w['win_pct'] * log5
        + w['run_diff'] * rd_prob
        + w['recent_form'] * recent_prob
        + w['head_to_head'] * 0.5  # neutral since h2h not readily available
    )
    prob = base_prob + w['home_field'] * (0.5 + hfa)
    return float(min(max(prob, 0.05), 0.95))


def simulate_playoff_matchup(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
    n_sims: int = 1000,
    series_length: int = 7,
    home_team_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate a playoff series between two teams.

    team_a/team_b: dicts with win_pct, run_diff_per_game, team_name
    series_length: 3, 5, or 7
    Returns: team_a_wins_pct, team_b_wins_pct, avg_series_length, game_distribution
    """
    wins_needed = (series_length // 2) + 1
    prob_a = _matchup_win_prob(team_a, team_b, home_team_name)

    rng = np.random.default_rng(seed=seed)
    a_series_wins = 0
    length_counts: dict[int, int] = {i: 0 for i in range(wins_needed, series_length + 1)}
    total_games = 0

    for _ in range(n_sims):
        a_wins = 0
        b_wins = 0
        games = 0
        while a_wins < wins_needed and b_wins < wins_needed:
            games += 1
            if rng.random() < prob_a:
                a_wins += 1
            else:
                b_wins += 1
        if a_wins >= wins_needed:
            a_series_wins += 1
        total_games += games
        if games in length_counts:
            length_counts[games] += 1

    a_pct = round(a_series_wins / n_sims * 100, 1)
    b_pct = round((n_sims - a_series_wins) / n_sims * 100, 1)
    avg_len = round(total_games / n_sims, 1)

    game_dist = {str(k): round(v / n_sims * 100, 1) for k, v in sorted(length_counts.items())}

    return {
        'team_a_name': team_a.get('team_name', 'Team A'),
        'team_b_name': team_b.get('team_name', 'Team B'),
        'team_a_win_pct': a_pct,
        'team_b_win_pct': b_pct,
        'avg_series_length': avg_len,
        'game_distribution': game_dist,
        'n_sims': n_sims,
        'series_length': series_length,
        'model_win_prob': round(prob_a, 4),
    }


def simulate_full_playoff_bracket(
    seed_df: pd.DataFrame,
    n_sims: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the full MLB playoff bracket N times.

    seed_df: from build_playoff_seeds() with columns league, seed, team_name, win_pct, etc.
    Returns DataFrame with columns: team_name, league, seed, wc_advance_pct,
        ds_advance_pct, pennant_pct, ws_pct
    """
    _cols = ['team_name', 'league', 'seed', 'wc_advance_pct', 'ds_advance_pct',
             'pennant_pct', 'ws_pct']
    if seed_df.empty:
        return pd.DataFrame(columns=_cols)

    rng = np.random.default_rng(seed=seed)

    # Build team lookup
    teams: dict[str, dict[str, Any]] = {}
    for _, row in seed_df.iterrows():
        teams[f"{row['league']}_{row['seed']}"] = row.to_dict()

    # Counters
    counters: dict[str, dict[str, int]] = {}
    for key in teams:
        counters[key] = {'wc_advance': 0, 'ds_advance': 0, 'pennant': 0, 'ws': 0}

    def _sim_series(t_a: dict, t_b: dict, wins_needed: int) -> dict:
        p = _matchup_win_prob(t_a, t_b, t_a.get('team_name'))
        a_w, b_w = 0, 0
        while a_w < wins_needed and b_w < wins_needed:
            if rng.random() < p:
                a_w += 1
            else:
                b_w += 1
        return t_a if a_w >= wins_needed else t_b

    for _ in range(n_sims):
        league_champs = {}
        for league in ('AL', 'NL'):
            s1 = teams.get(f'{league}_1')
            s2 = teams.get(f'{league}_2')
            s3 = teams.get(f'{league}_3')
            s4 = teams.get(f'{league}_4')
            s5 = teams.get(f'{league}_5')
            s6 = teams.get(f'{league}_6')

            if not all([s1, s2, s3, s4, s5, s6]):
                continue

            # Wild Card Round (best of 3): 4v5, 3v6
            wc_winner_45 = _sim_series(s4, s5, 2)
            wc_winner_36 = _sim_series(s3, s6, 2)

            # Track WC advancement
            for s in [s4, s5]:
                k = f"{league}_{s['seed']}"
                if s == wc_winner_45:
                    counters[k]['wc_advance'] += 1
            for s in [s3, s6]:
                k = f"{league}_{s['seed']}"
                if s == wc_winner_36:
                    counters[k]['wc_advance'] += 1

            # Seeds 1 and 2 get byes (auto-advance through WC)
            counters[f'{league}_1']['wc_advance'] += 1
            counters[f'{league}_2']['wc_advance'] += 1

            # Division Series (best of 5): 1 vs wc_winner_45, 2 vs wc_winner_36
            ds_winner_1 = _sim_series(s1, wc_winner_45, 3)
            ds_winner_2 = _sim_series(s2, wc_winner_36, 3)

            for t in [s1, wc_winner_45]:
                k = f"{league}_{t['seed']}"
                if t == ds_winner_1:
                    counters[k]['ds_advance'] += 1
            for t in [s2, wc_winner_36]:
                k = f"{league}_{t['seed']}"
                if t == ds_winner_2:
                    counters[k]['ds_advance'] += 1

            # League Championship Series (best of 7)
            lcs_winner = _sim_series(ds_winner_1, ds_winner_2, 4)
            k_lcs = f"{league}_{lcs_winner['seed']}"
            counters[k_lcs]['pennant'] += 1
            league_champs[league] = lcs_winner

        # World Series (best of 7)
        al_champ = league_champs.get('AL')
        nl_champ = league_champs.get('NL')
        if al_champ and nl_champ:
            ws_winner = _sim_series(al_champ, nl_champ, 4)
            k_ws = f"{ws_winner['league']}_{ws_winner['seed']}"
            counters[k_ws]['ws'] += 1

    rows = []
    for key, team in teams.items():
        c = counters[key]
        rows.append({
            'team_name': team['team_name'],
            'league': team['league'],
            'seed': team['seed'],
            'wc_advance_pct': round(c['wc_advance'] / n_sims * 100, 1),
            'ds_advance_pct': round(c['ds_advance'] / n_sims * 100, 1),
            'pennant_pct': round(c['pennant'] / n_sims * 100, 1),
            'ws_pct': round(c['ws'] / n_sims * 100, 1),
        })

    result = pd.DataFrame(rows)[_cols].sort_values('ws_pct', ascending=False).reset_index(drop=True)
    return result