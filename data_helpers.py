from __future__ import annotations

from typing import Any

import pandas as pd

from formatting import coerce_float, coerce_int, format_record, safe_pct, signed, stoplight

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
    df['team_runs'] = df.apply(lambda r: coerce_int(r['away_score'], 0) if r['away'] == team_name else coerce_int(r['home_score'], 0), axis=1)
    df['opp_runs'] = df.apply(lambda r: coerce_int(r['home_score'], 0) if r['away'] == team_name else coerce_int(r['away_score'], 0), axis=1)
    df['run_diff'] = df['team_runs'] - df['opp_runs']
    df['is_final'] = df['status'].astype(str).str.contains('Final', case=False, na=False)
    df['result'] = df.apply(lambda r: 'W' if r['team_runs'] > r['opp_runs'] else ('L' if r['team_runs'] < r['opp_runs'] else 'T'), axis=1)
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
    today_status = ', '.join([s for s in daily_df['status'].astype(str).tolist() if s]) if not daily_df.empty else 'No game today'

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
        {'Metric': 'Last 10 Record', 'Value': format_record(last10_wins, len(last_10) - last10_wins), 'Trend': stoplight((last10_wins / max(len(last_10), 1)) - 0.5)},
        {'Metric': 'Last 10 Run Differential / Game', 'Value': round(last_10['run_diff'].mean(), 2), 'Trend': stoplight(last_10['run_diff'].mean())},
        {'Metric': 'Scoring Consistency Std Dev', 'Value': consistency, 'Trend': stoplight(2.5 - consistency)},
        {'Metric': 'Home Avg Runs', 'Value': round(home_split['team_runs'].mean(), 2) if not home_split.empty else 0.0, 'Trend': '🟡 Split'},
        {'Metric': 'Away Avg Runs', 'Value': round(away_split['team_runs'].mean(), 2) if not away_split.empty else 0.0, 'Trend': '🟡 Split'},
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
    """Calculate batter score from multiple metrics."""
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
        
        # Calculate whiff rate from swings
        all_pa = df[df['Batter'] == batter]
        swings = all_pa['description'].isin(SWING_DESCRIPTIONS).sum()
        whiffs = all_pa['description'].isin(WHIFF_DESCRIPTIONS).sum()
        whiff_pct = (whiffs / swings * 100) if swings else 0.0
        
        xwoba = grp['estimated_woba_using_speedangle'].replace(0, pd.NA).dropna().mean()
        xwoba = 0.0 if pd.isna(xwoba) else float(xwoba)
        
        # Contact quality: percent of balls hit hard (95+ exit velo)
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
    """Calculate pitcher score from multiple metrics."""
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
        
        # Strike percentage (estimation from description)
        strikes = grp['description'].isin({'swinging_strike', 'swinging_strike_blocked', 'called_strike'}).sum()
        strike_pct = (strikes / pitches * 100) if pitches else 0.0
        
        avg_woba = grp['woba_value'].replace(0, pd.NA).dropna().mean()
        avg_woba = 0.0 if pd.isna(avg_woba) else float(avg_woba)
        
        # K% estimation (simplified from pitch data)
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
    batter_contact = statcast_batter_df[statcast_batter_df.get('launch_speed', pd.Series(dtype=float)) > 0].copy() if not statcast_batter_df.empty else pd.DataFrame()
    total_pitcher = len(statcast_pitcher_df)
    swings = statcast_pitcher_df['description'].isin(SWING_DESCRIPTIONS).sum() if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns else 0
    whiffs = statcast_pitcher_df['description'].isin(WHIFF_DESCRIPTIONS).sum() if not statcast_pitcher_df.empty and 'description' in statcast_pitcher_df.columns else 0
    pitch_whiff = (whiffs / swings * 100) if swings else 0.0
    avg_spin = statcast_pitcher_df['release_spin_rate'].mean() if not statcast_pitcher_df.empty and 'release_spin_rate' in statcast_pitcher_df.columns else 0.0
    avg_ev = batter_contact['launch_speed'].mean() if not batter_contact.empty else 0.0
    hard_hit = (batter_contact['launch_speed'] >= 95).mean() * 100 if not batter_contact.empty else 0.0
    return pd.DataFrame([
        {'Metric': 'Team Avg Exit Velocity', 'Value': round(avg_ev, 1), 'Trend': stoplight(avg_ev - 89, neutral_band=1)},
        {'Metric': 'Team Hard Hit %', 'Value': round(hard_hit, 1), 'Trend': stoplight(hard_hit - 40, neutral_band=3)},
        {'Metric': 'Staff Avg Spin Rate', 'Value': round(avg_spin, 0), 'Trend': stoplight(avg_spin - 2250, neutral_band=50)},
        {'Metric': 'Staff Whiff %', 'Value': round(pitch_whiff, 1), 'Trend': stoplight(pitch_whiff - 28, neutral_band=2)},
        {'Metric': 'Pitch Sample Size', 'Value': total_pitcher, 'Trend': '🟡 Context'},
    ])


# ---------------------------------------------------------------------------
# League-wide ranking builders
# ---------------------------------------------------------------------------


def _fg_col(df: pd.DataFrame, name: str, default: Any = 0) -> pd.Series:
    """Safely access a DataFrame column, returning *default* if missing."""
    if name in df.columns:
        return df[name].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def _fg_pct_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Parse a FanGraphs percentage column to a float Series."""
    if name not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return df[name].apply(_parse_fg_pct_val)


def _parse_fg_pct_val(val: Any) -> float:
    if isinstance(val, (int, float)):
        return 0.0 if pd.isna(val) else round(float(val), 1)
    s = str(val).strip().rstrip('%').strip()
    try:
        return round(float(s), 1)
    except (ValueError, TypeError):
        return 0.0


def _safe_sum(grp: pd.DataFrame, col: str) -> float:
    """Sum a column in a group, returning 0.0 if the column is missing."""
    if col in grp.columns:
        return float(grp[col].fillna(0).sum())
    return 0.0


# ---- Offense -----------------------------------------------------------------

def build_offensive_rankings(
    fg_batting: pd.DataFrame,
    mlb_hitting: pd.DataFrame,
) -> pd.DataFrame:
    """League-wide team offensive rankings.

    Sources
    -------
    Primary : FanGraphs team batting (wOBA, wRC+, BB%, K%, etc.)
    Fallback: MLB Stats API team hitting (standard metrics only).
    """
    if not fg_batting.empty and 'Team' in fg_batting.columns:
        df = fg_batting.copy()
        out = pd.DataFrame()
        out['Team'] = _fg_col(df, 'Team', '-').astype(str)
        g = _fg_col(df, 'G', 1).replace(0, 1)
        r = _fg_col(df, 'R', 0)
        out['R'] = r.astype(int)
        out['R/G'] = (r / g).round(2)
        out['AVG'] = _fg_col(df, 'AVG', 0.0).round(3)
        out['OBP'] = _fg_col(df, 'OBP', 0.0).round(3)
        out['SLG'] = _fg_col(df, 'SLG', 0.0).round(3)
        out['OPS'] = _fg_col(df, 'OPS', 0.0).round(3)
        out['HR'] = _fg_col(df, 'HR', 0).astype(int)
        out['SB'] = _fg_col(df, 'SB', 0).astype(int)
        out['BB%'] = _fg_pct_series(df, 'BB%')
        out['K%'] = _fg_pct_series(df, 'K%')
        for adv in ['wOBA', 'wRC+']:
            if adv in df.columns:
                out[adv] = df[adv].fillna(0).round(3 if adv == 'wOBA' else 0)
        out['Source'] = 'FanGraphs'
    elif not mlb_hitting.empty and 'Team' in mlb_hitting.columns:
        keep = [c for c in ['Team', 'R', 'R/G', 'HR', 'SB', 'AVG', 'OBP',
                            'SLG', 'OPS', 'BB%', 'K%'] if c in mlb_hitting.columns]
        out = mlb_hitting[keep].copy()
        out['Source'] = 'MLB Stats API'
    else:
        return pd.DataFrame()

    out = out.drop_duplicates(subset='Team', keep='first')
    out = out.sort_values('OPS', ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


# ---- Defense -----------------------------------------------------------------

def build_defensive_rankings(
    mlb_fielding: pd.DataFrame,
    fg_fielding: pd.DataFrame,
) -> pd.DataFrame:
    """League-wide team defensive rankings.

    Sources
    -------
    Standard : MLB Stats API (E, Fielding %, DP).
    Advanced : FanGraphs team fielding (DRS, UZR, Def).
    """
    if not mlb_fielding.empty and 'Team' in mlb_fielding.columns:
        keep = [c for c in ['Team', 'E', 'Fielding %', 'DP', 'A', 'PO']
                if c in mlb_fielding.columns]
        out = mlb_fielding[keep].copy()

        if not fg_fielding.empty and 'Team' in fg_fielding.columns:
            fg = fg_fielding.copy()
            adv = pd.DataFrame({'Team': _fg_col(fg, 'Team', '-').astype(str)})
            has_adv = False
            for col in ['DRS', 'UZR', 'Def']:
                if col in fg.columns:
                    adv[col] = fg[col].fillna(0).round(1 if col != 'DRS' else 0)
                    has_adv = True
            if has_adv:
                out = out.merge(adv, on='Team', how='left')
                for c in adv.columns:
                    if c != 'Team' and c in out.columns:
                        out[c] = out[c].fillna(0)
                out['Source'] = 'MLB Stats API + FanGraphs'
            else:
                out['Source'] = 'MLB Stats API'
        else:
            out['Source'] = 'MLB Stats API'

    elif not fg_fielding.empty and 'Team' in fg_fielding.columns:
        df = fg_fielding.copy()
        out = pd.DataFrame()
        out['Team'] = _fg_col(df, 'Team', '-').astype(str)
        out['E'] = _fg_col(df, 'E', 0).astype(int)
        out['Fielding %'] = _fg_col(df, 'FP', 0.0).round(3)
        out['DP'] = _fg_col(df, 'DP', 0).astype(int)
        for adv in ['DRS', 'UZR', 'Def']:
            if adv in df.columns:
                out[adv] = df[adv].fillna(0).round(1 if adv != 'DRS' else 0)
        out['Source'] = 'FanGraphs'
    else:
        return pd.DataFrame()

    out = out.drop_duplicates(subset='Team', keep='first')
    sort_col = 'Def' if 'Def' in out.columns else 'Fielding %'
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


# ---- Pitching helpers --------------------------------------------------------

def _classify_pitcher_role(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each pitcher row as ``'SP'`` or ``'RP'`` based on GS/G ratio."""
    out = df.copy()
    g = _fg_col(out, 'G', 0).astype(float).replace(0, 1)
    gs = _fg_col(out, 'GS', 0).astype(float)
    out['_role'] = 'RP'
    out.loc[(gs >= 3) & (gs / g >= 0.5), '_role'] = 'SP'
    return out


_FIP_CONSTANT = 3.10  # league-average FIP constant (approx)


def _agg_pitching_team(grp: pd.DataFrame, is_starter: bool) -> dict[str, Any]:
    """Aggregate individual pitcher stats for one team."""
    ip = _safe_sum(grp, 'IP')
    er = _safe_sum(grp, 'ER')
    h = _safe_sum(grp, 'H')
    bb = _safe_sum(grp, 'BB')
    so = _safe_sum(grp, 'SO')
    hr = _safe_sum(grp, 'HR')
    tbf = _safe_sum(grp, 'TBF')
    hbp = _safe_sum(grp, 'HBP')
    sv = _safe_sum(grp, 'SV')
    hld = _safe_sum(grp, 'HLD')
    bs = _safe_sum(grp, 'BS')
    gs = _safe_sum(grp, 'GS')

    ip_s = max(ip, 0.1)
    tbf_s = max(tbf, 1)
    ab_approx = max(tbf_s - bb - hbp, 1)

    row: dict[str, Any] = {
        'ERA': round(er / ip_s * 9, 2),
        'WHIP': round((bb + h) / ip_s, 2),
        'IP': round(ip, 1),
        'K%': round(so / tbf_s * 100, 1),
        'BB%': round(bb / tbf_s * 100, 1),
        'K-BB%': round((so - bb) / tbf_s * 100, 1),
        'HR/9': round(hr / ip_s * 9, 2),
        'FIP': round(((13 * hr + 3 * (bb + hbp) - 2 * so) / ip_s) + _FIP_CONSTANT, 2),
        'Opp AVG': round(h / ab_approx, 3),
    }

    if is_starter:
        row['IP/GS'] = round(ip / max(gs, 1), 1)
        if 'QS' in grp.columns:
            row['QS'] = int(grp['QS'].fillna(0).sum())
    else:
        row['SV'] = int(sv)
        row['HLD'] = int(hld)
        row['BS'] = int(bs)
        row['Save%'] = round(sv / max(sv + bs, 1) * 100, 1)

    return row


# ---- Starters ----------------------------------------------------------------

def build_starter_rankings(
    fg_pitch_ind: pd.DataFrame,
    mlb_pitching: pd.DataFrame,
) -> pd.DataFrame:
    """Team-level starting-pitching rankings.

    Sources
    -------
    Primary : FanGraphs individual pitching, classified by GS/G ratio.
    Fallback: MLB Stats API team pitching (combined, not split by role).
    """
    if not fg_pitch_ind.empty and 'Team' in fg_pitch_ind.columns:
        classified = _classify_pitcher_role(fg_pitch_ind)
        starters = classified[classified['_role'] == 'SP'].copy()
        if starters.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for team, grp in starters.groupby('Team'):
            row = _agg_pitching_team(grp, is_starter=True)
            row['Team'] = team
            rows.append(row)

        out = pd.DataFrame(rows)
        out['Source'] = 'FanGraphs (individual, SP classified)'
        out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
        out.insert(0, 'Rank', range(1, len(out) + 1))
        # Reorder so Team is right after Rank
        cols = ['Rank', 'Team'] + [c for c in out.columns if c not in ('Rank', 'Team')]
        return out[cols]

    # Fallback: MLB Stats API (combined pitching, no role split)
    if not mlb_pitching.empty and 'Team' in mlb_pitching.columns:
        df = mlb_pitching.copy()
        ip = _fg_col(df, 'IP', 0.1).replace(0, 0.1)
        tbf = _fg_col(df, 'TBF', 1).replace(0, 1)
        out = pd.DataFrame()
        out['Team'] = _fg_col(df, 'Team', '-').astype(str)
        out['ERA'] = _fg_col(df, 'ERA', 0.0)
        out['WHIP'] = _fg_col(df, 'WHIP', 0.0)
        out['IP'] = _fg_col(df, 'IP', 0.0)
        out['K%'] = (_fg_col(df, 'SO', 0) / tbf * 100).round(1)
        out['BB%'] = (_fg_col(df, 'BB', 0) / tbf * 100).round(1)
        out['K-BB%'] = (out['K%'] - out['BB%']).round(1)
        out['HR/9'] = (_fg_col(df, 'HR', 0) / ip * 9).round(2)
        out['Opp AVG'] = _fg_col(df, 'Opp AVG', 0.0)
        out['Source'] = 'MLB Stats API (combined, not split by role)'
        out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
        out.insert(0, 'Rank', range(1, len(out) + 1))
        return out

    return pd.DataFrame()


# ---- Relievers ---------------------------------------------------------------

def build_reliever_rankings(
    fg_pitch_ind: pd.DataFrame,
    mlb_pitching: pd.DataFrame,
) -> pd.DataFrame:
    """Team-level relief-pitching rankings.

    Sources
    -------
    Primary : FanGraphs individual pitching, classified by GS/G ratio.
    Fallback: MLB Stats API team pitching (combined, not split by role).
    """
    if not fg_pitch_ind.empty and 'Team' in fg_pitch_ind.columns:
        classified = _classify_pitcher_role(fg_pitch_ind)
        relievers = classified[classified['_role'] == 'RP'].copy()
        if relievers.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for team, grp in relievers.groupby('Team'):
            row = _agg_pitching_team(grp, is_starter=False)
            row['Team'] = team
            rows.append(row)

        out = pd.DataFrame(rows)
        out['Source'] = 'FanGraphs (individual, RP classified)'
        out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
        out.insert(0, 'Rank', range(1, len(out) + 1))
        cols = ['Rank', 'Team'] + [c for c in out.columns if c not in ('Rank', 'Team')]
        return out[cols]

    # Fallback: MLB Stats API combined
    if not mlb_pitching.empty and 'Team' in mlb_pitching.columns:
        df = mlb_pitching.copy()
        out = pd.DataFrame()
        out['Team'] = _fg_col(df, 'Team', '-').astype(str)
        out['ERA'] = _fg_col(df, 'ERA', 0.0)
        out['WHIP'] = _fg_col(df, 'WHIP', 0.0)
        out['SV'] = _fg_col(df, 'SV', 0).astype(int)
        out['HLD'] = _fg_col(df, 'HLD', 0).astype(int)
        out['BS'] = _fg_col(df, 'BS', 0).astype(int)
        sv = out['SV'].astype(float)
        bs = out['BS'].astype(float)
        out['Save%'] = (sv / (sv + bs).replace(0, 1) * 100).round(1)
        out['Source'] = 'MLB Stats API (combined, not split by role)'
        out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
        out.insert(0, 'Rank', range(1, len(out) + 1))
        return out

    return pd.DataFrame()


# ---- WAR Leaderboard ---------------------------------------------------------

def build_war_leaderboard(
    fg_bat_ind: pd.DataFrame,
    fg_pitch_ind: pd.DataFrame,
    filter_type: str = 'all',
) -> pd.DataFrame:
    """MLB-wide WAR leaderboard.

    Parameters
    ----------
    filter_type : ``'all'``, ``'hitters'``, or ``'pitchers'``.

    Sources
    -------
    FanGraphs individual batting + pitching stats.
    """
    parts: list[pd.DataFrame] = []

    if not fg_bat_ind.empty and 'WAR' in fg_bat_ind.columns and filter_type in ('all', 'hitters'):
        bat = fg_bat_ind.copy()
        bdf = pd.DataFrame()
        bdf['Player'] = _fg_col(bat, 'Name', '-').astype(str)
        bdf['Team'] = _fg_col(bat, 'Team', '-').astype(str)
        bdf['WAR'] = _fg_col(bat, 'WAR', 0.0).round(1)
        bdf['Type'] = 'Hitter'
        if 'Pos' in bat.columns:
            bdf['Pos'] = bat['Pos'].fillna('-').astype(str)
        elif 'POS' in bat.columns:
            bdf['Pos'] = bat['POS'].fillna('-').astype(str)
        else:
            bdf['Pos'] = '-'
        bdf['bWAR'] = bdf['WAR']
        bdf['pWAR'] = 0.0
        parts.append(bdf)

    if not fg_pitch_ind.empty and 'WAR' in fg_pitch_ind.columns and filter_type in ('all', 'pitchers'):
        pit = fg_pitch_ind.copy()
        pdf = pd.DataFrame()
        pdf['Player'] = _fg_col(pit, 'Name', '-').astype(str)
        pdf['Team'] = _fg_col(pit, 'Team', '-').astype(str)
        pdf['WAR'] = _fg_col(pit, 'WAR', 0.0).round(1)
        pdf['Type'] = 'Pitcher'
        pdf['Pos'] = 'P'
        pdf['bWAR'] = 0.0
        pdf['pWAR'] = pdf['WAR']
        parts.append(pdf)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    if filter_type == 'all':
        # For two-way players, aggregate WAR by name+team
        agg = combined.groupby(['Player', 'Team'], as_index=False).agg({
            'WAR': 'sum',
            'bWAR': 'sum',
            'pWAR': 'sum',
            'Type': lambda x: '/'.join(sorted(set(x))),
            'Pos': 'first',
        })
        combined = agg

    combined = combined.sort_values('WAR', ascending=False).reset_index(drop=True)
    combined.insert(0, 'Rank', range(1, len(combined) + 1))
    cols = ['Rank', 'Player', 'Team', 'Pos', 'Type', 'WAR', 'bWAR', 'pWAR']
    cols = [c for c in cols if c in combined.columns]
    combined['Source'] = 'FanGraphs'
    return combined[cols + ['Source']]


# ---- Summary helpers for leader cards ----------------------------------------

def top_n_summary(
    df: pd.DataFrame,
    sort_col: str,
    n: int = 5,
    ascending: bool = True,
    label_col: str = 'Team',
) -> pd.DataFrame:
    """Return the top *n* rows from *df* sorted by *sort_col*.

    Useful for quick "Top 5" / "Bottom 5" summary cards.
    """
    if df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    keep = [c for c in [label_col, sort_col] if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values(sort_col, ascending=ascending).head(n).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out