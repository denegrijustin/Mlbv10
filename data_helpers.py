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

def sort_rankings(df: pd.DataFrame, sort_col: str, ascending: bool) -> pd.DataFrame:
    """Sort a rankings DataFrame and re-assign rank column."""
    if df.empty or sort_col not in df.columns:
        return df
    out = df.copy()
    out = out.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    out['Rank'] = range(1, len(out) + 1)
    return out


def build_offensive_rankings(hitting_df: pd.DataFrame) -> pd.DataFrame:
    """Build MLB-wide team offensive rankings table from MLB Stats API hitting data."""
    if hitting_df.empty:
        return pd.DataFrame(columns=['Rank', 'Team', 'GP', 'R', 'R/G', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'SB', 'BB%', 'K%'])

    df = hitting_df.copy()
    gp = df['gamesPlayed'].replace(0, 1)
    pa = df['plateAppearances'].replace(0, 1)

    out = pd.DataFrame({
        'team_id': df['team_id'],
        'Team': df['team_name'],
        'GP': df['gamesPlayed'],
        'R': df['runs'],
        'R/G': (df['runs'] / gp).round(2),
        'AVG': df['avg'].round(3),
        'OBP': df['obp'].round(3),
        'SLG': df['slg'].round(3),
        'OPS': df['ops'].round(3),
        'HR': df['homeRuns'],
        'SB': df['stolenBases'],
        'BB%': (df['baseOnBalls'] / pa * 100).round(1),
        'K%': (df['strikeOuts'] / pa * 100).round(1),
    })

    # Default: sort by OPS descending (rank 1 = best offense)
    out = out.sort_values('OPS', ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


def build_defensive_rankings(fielding_df: pd.DataFrame) -> pd.DataFrame:
    """Build MLB-wide team defensive rankings table from MLB Stats API fielding data."""
    if fielding_df.empty:
        return pd.DataFrame(columns=['Rank', 'Team', 'GP', 'E', 'FLD%', 'DP', 'RF/G'])

    df = fielding_df.copy()

    out = pd.DataFrame({
        'team_id': df['team_id'],
        'Team': df['team_name'],
        'GP': df['gamesPlayed'],
        'E': df['errors'],
        'FLD%': df['fieldingPercentage'].round(4),
        'DP': df['doublePlays'],
        'RF/G': df['rangeFactorPerGame'].round(2),
    })

    # Default: sort by FLD% descending (rank 1 = best fielding %)
    out = out.sort_values('FLD%', ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


def _aggregate_pitchers_by_team(pitcher_df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Aggregate individual pitcher stats by team for a given role (SP or RP)."""
    df = pitcher_df[pitcher_df['role'] == role].copy()
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(['team_id', 'team_name'], as_index=False).agg(
        games_played=('games_played', 'sum'),
        games_started=('games_started', 'sum'),
        innings_pitched=('innings_pitched', 'sum'),
        strikeouts=('strikeouts', 'sum'),
        walks=('walks', 'sum'),
        hits_allowed=('hits_allowed', 'sum'),
        earned_runs=('earned_runs', 'sum'),
        home_runs=('home_runs', 'sum'),
        quality_starts=('quality_starts', 'sum'),
        saves=('saves', 'sum'),
        holds=('holds', 'sum'),
        blown_saves=('blown_saves', 'sum'),
        save_opportunities=('save_opportunities', 'sum'),
        batters_faced=('batters_faced', 'sum'),
    )

    ip = agg['innings_pitched'].replace(0, 0.001)
    bf = agg['batters_faced'].replace(0, 1)

    agg['ERA'] = (agg['earned_runs'] * 9 / ip).round(2)
    agg['WHIP'] = ((agg['walks'] + agg['hits_allowed']) / ip).round(3)
    agg['K%'] = (agg['strikeouts'] / bf * 100).round(1)
    agg['BB%'] = (agg['walks'] / bf * 100).round(1)
    agg['K-BB%'] = (agg['K%'] - agg['BB%']).round(1)
    agg['HR/9'] = (agg['home_runs'] * 9 / ip).round(2)

    if role == 'SP':
        gs = agg['games_started'].replace(0, 1)
        agg['IP/GS'] = (agg['innings_pitched'] / gs).round(1)

    sv_opp = agg['save_opportunities'].replace(0, 1)
    agg['SV%'] = (agg['saves'] / sv_opp * 100).round(1)

    return agg


def build_starter_rankings(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """Build MLB-wide team starting pitcher rankings from individual stats."""
    empty_cols = ['Rank', 'Team', 'ERA', 'WHIP', 'IP', 'IP/GS', 'K%', 'BB%', 'K-BB%', 'HR/9', 'QS']
    if pitcher_df.empty:
        return pd.DataFrame(columns=empty_cols)

    agg = _aggregate_pitchers_by_team(pitcher_df, 'SP')
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame({
        'team_id': agg['team_id'],
        'Team': agg['team_name'],
        'ERA': agg['ERA'],
        'WHIP': agg['WHIP'],
        'IP': agg['innings_pitched'].round(1),
        'IP/GS': agg.get('IP/GS', pd.Series(0.0, index=agg.index)),
        'K%': agg['K%'],
        'BB%': agg['BB%'],
        'K-BB%': agg['K-BB%'],
        'HR/9': agg['HR/9'],
        'QS': agg['quality_starts'],
    })

    # Default: sort by ERA ascending (rank 1 = lowest ERA)
    out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


def build_reliever_rankings(pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """Build MLB-wide team bullpen/reliever rankings from individual stats."""
    empty_cols = ['Rank', 'Team', 'ERA', 'WHIP', 'K%', 'BB%', 'K-BB%', 'SV', 'HLD', 'SV%', 'BS']
    if pitcher_df.empty:
        return pd.DataFrame(columns=empty_cols)

    agg = _aggregate_pitchers_by_team(pitcher_df, 'RP')
    if agg.empty:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame({
        'team_id': agg['team_id'],
        'Team': agg['team_name'],
        'ERA': agg['ERA'],
        'WHIP': agg['WHIP'],
        'K%': agg['K%'],
        'BB%': agg['BB%'],
        'K-BB%': agg['K-BB%'],
        'SV': agg['saves'],
        'HLD': agg['holds'],
        'SV%': agg['SV%'],
        'BS': agg['blown_saves'],
    })

    # Default: sort by ERA ascending (rank 1 = lowest ERA)
    out = out.sort_values('ERA', ascending=True).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


def build_war_leaderboard(
    bat_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    min_pa: int = 10,
    min_ip: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build WAR leaderboards (overall, batters, pitchers) from FanGraphs data."""
    batter_rows: list[dict] = []
    pitcher_rows: list[dict] = []

    if not bat_df.empty and 'WAR' in bat_df.columns:
        name_col = next((c for c in ['Name', 'name', 'playerName'] if c in bat_df.columns), None)
        team_col = next((c for c in ['Team', 'team', 'Team Name'] if c in bat_df.columns), None)
        pos_col = next((c for c in ['Pos', 'pos', 'Position', 'position'] if c in bat_df.columns), None)
        pa_col = next((c for c in ['PA', 'pa', 'plateAppearances'] if c in bat_df.columns), None)

        for _, row in bat_df.iterrows():
            war_val = coerce_float(row.get('WAR'), 0.0)
            pa_val = coerce_int(row.get(pa_col, 0) if pa_col else 0, 0)
            if pa_col and pa_val < min_pa:
                continue
            batter_rows.append({
                'Name': coerce_text(row, name_col),
                'Team': coerce_text(row, team_col),
                'Pos': coerce_text(row, pos_col) if pos_col else 'OF/IF',
                'WAR': war_val,
                'HR': coerce_int(row.get('HR'), 0),
                'AVG': coerce_float(row.get('AVG'), 0.0),
                'OPS': coerce_float(row.get('OPS'), 0.0),
                'wRC+': coerce_int(row.get('wRC+'), 0) if 'wRC+' in bat_df.columns else 0,
            })

    if not pit_df.empty and 'WAR' in pit_df.columns:
        name_col = next((c for c in ['Name', 'name', 'playerName'] if c in pit_df.columns), None)
        team_col = next((c for c in ['Team', 'team', 'Team Name'] if c in pit_df.columns), None)
        ip_col = next((c for c in ['IP', 'ip', 'inningsPitched'] if c in pit_df.columns), None)
        gs_col = next((c for c in ['GS', 'gs', 'gamesStarted'] if c in pit_df.columns), None)

        for _, row in pit_df.iterrows():
            war_val = coerce_float(row.get('WAR'), 0.0)
            ip_val = coerce_float(row.get(ip_col, 0.0) if ip_col else 0.0, 0.0)
            if ip_col and ip_val < min_ip:
                continue
            gs_val = coerce_int(row.get(gs_col, 0) if gs_col else 0, 0)
            role = 'SP' if gs_val >= 3 else 'RP'
            pitcher_rows.append({
                'Name': coerce_text(row, name_col),
                'Team': coerce_text(row, team_col),
                'Pos': role,
                'WAR': war_val,
                'ERA': coerce_float(row.get('ERA'), 0.0),
                'IP': ip_val,
                'WHIP': coerce_float(row.get('WHIP'), 0.0),
                'FIP': coerce_float(row.get('FIP'), 0.0) if 'FIP' in pit_df.columns else 0.0,
            })

    def _make_df(rows: list[dict], extra_cols: list[str], base_cols: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=['Rank'] + base_cols + extra_cols)
        df = pd.DataFrame(rows).sort_values('WAR', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        return df

    bat_war_df = _make_df(batter_rows, ['HR', 'AVG', 'OPS', 'wRC+'], ['Name', 'Team', 'Pos', 'WAR'])
    pit_war_df = _make_df(pitcher_rows, ['ERA', 'IP', 'WHIP', 'FIP'], ['Name', 'Team', 'Pos', 'WAR'])

    # Combined overall leaderboard
    combined = []
    for r in batter_rows:
        combined.append({'Name': r['Name'], 'Team': r['Team'], 'Pos': r['Pos'], 'WAR': r['WAR']})
    for r in pitcher_rows:
        combined.append({'Name': r['Name'], 'Team': r['Team'], 'Pos': r['Pos'], 'WAR': r['WAR']})

    if combined:
        overall_df = pd.DataFrame(combined).sort_values('WAR', ascending=False).reset_index(drop=True)
        overall_df.insert(0, 'Rank', range(1, len(overall_df) + 1))
    else:
        overall_df = pd.DataFrame(columns=['Rank', 'Name', 'Team', 'Pos', 'WAR'])

    return overall_df, bat_war_df, pit_war_df


def coerce_text(row: Any, col: str | None, default: str = '-') -> str:
    """Safely extract a text value from a Series row."""
    if col is None:
        return default
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip() or default