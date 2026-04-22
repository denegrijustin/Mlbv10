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
        return pd.DataFrame(columns=['Metric', 'Value', 'Signal', 'Compared To'])

    finals = games[games['is_final']].copy()
    if finals.empty:
        return pd.DataFrame(columns=['Metric', 'Value', 'Signal', 'Compared To'])

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
        {'Metric': 'Season Avg Runs For', 'Value': str(round(season_rf, 2)), 'Signal': '🟡 Reference', 'Compared To': 'Season baseline'},
        {'Metric': 'Season Avg Runs Against', 'Value': str(round(season_ra, 2)), 'Signal': '🟡 Reference', 'Compared To': 'Season baseline'},
        {'Metric': 'Last 5 Avg Runs For', 'Value': str(round(last5_rf, 2)), 'Signal': stoplight(last5_rf - prev5_rf), 'Compared To': f'Prev 5 avg: {round(prev5_rf, 2)}'},
        {'Metric': 'Last 5 Avg Runs Against', 'Value': str(round(last5_ra, 2)), 'Signal': stoplight(prev5_ra - last5_ra), 'Compared To': f'Prev 5 avg: {round(prev5_ra, 2)}'},
        {
            'Metric': 'Last 10 Record',
            'Value': format_record(last10_wins, len(last_10) - last10_wins),
            'Signal': stoplight((last10_wins / max(len(last_10), 1)) - 0.5),
            'Compared To': '.500 (break-even)',
        },
        {
            'Metric': 'Last 10 Run Differential / Game',
            'Value': str(round(last_10['run_diff'].mean(), 2)),
            'Signal': stoplight(last_10['run_diff'].mean()),
            'Compared To': '0.0 (break-even)',
        },
        {'Metric': 'Scoring Consistency Std Dev', 'Value': str(consistency), 'Signal': stoplight(2.5 - consistency), 'Compared To': '< 2.5 is consistent'},
        {
            'Metric': 'Home Avg Runs',
            'Value': str(round(home_split['team_runs'].mean(), 2)) if not home_split.empty else '0.0',
            'Signal': '🏠 Split',
            'Compared To': 'Home games only',
        },
        {
            'Metric': 'Away Avg Runs',
            'Value': str(round(away_split['team_runs'].mean(), 2)) if not away_split.empty else '0.0',
            'Signal': '✈️ Split',
            'Compared To': 'Away games only',
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
    """Return next 3 upcoming SERIES for team_name after from_date_str.

    Groups consecutive games against the same opponent into a series.
    Each entry contains: team_name, series_start, series_end, num_games, is_home, game_dates.
    """
    if season_df.empty or not team_name:
        return []
    df = season_df[(season_df['away'] == team_name) | (season_df['home'] == team_name)].copy()
    if df.empty:
        return []
    # Filter out Final games
    df = df[~df['status'].astype(str).str.contains('Final', case=False, na=False)]
    # Filter to games after from_date_str and sort
    date_col = 'officialDate' if 'officialDate' in df.columns else ('gameDate' if 'gameDate' in df.columns else None)
    if date_col:
        df[date_col] = df[date_col].astype(str).str[:10]
        df = df[df[date_col] > str(from_date_str)]
        df = df.sort_values(date_col)

    # Group consecutive games against the same opponent into series
    series_list: list[dict[str, Any]] = []
    current_opponent: str | None = None
    current_series: dict[str, Any] = {}
    for _, row in df.iterrows():
        is_home = row.get('home') == team_name
        opponent = str(row.get('away') if is_home else row.get('home'))
        game_date = str(row.get(date_col, '')) if date_col else ''

        if opponent != current_opponent:
            if current_series:
                series_list.append(current_series)
            if len(series_list) >= 3:
                break
            current_opponent = opponent
            current_series = {
                'team_name': opponent,
                'series_start': game_date,
                'series_end': game_date,
                'num_games': 1,
                'is_home': bool(is_home),
                'game_dates': [game_date],
            }
        else:
            current_series['series_end'] = game_date
            current_series['num_games'] += 1
            current_series['game_dates'].append(game_date)

    if current_series and len(series_list) < 3:
        series_list.append(current_series)

    return series_list[:3]


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
    """
    cols_out = ['field_x', 'field_y', 'events', 'player_name', 'launch_speed', 'hit_distance_sc', 'game_date']
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

    # Fill optional columns
    for col in ('events', 'launch_speed', 'hit_distance_sc'):
        if col not in df.columns:
            df[col] = None

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