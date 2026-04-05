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

# ── Hitter grade calibration constants ─────────────────────────────────────────
# These values were tuned so that league-average Statcast metrics map to the D
# grade range (70-74) and elite performers reach the A range (90+).
GRADE_FLOOR_OFFSET: float = 47.0       # score when raw sub-components are all 0
GRADE_FLOOR_SCALE: float = 0.55        # scale factor applied to raw sub-component sum
HARD_HIT_CEILING: float = 55.0        # hard-hit% that saturates the component (elite ~57%)
BARREL_PCT_CEILING: float = 25.0      # barrel% that saturates the component (elite ~22%)
BARREL_MIN_EV: float = 98.0           # minimum exit velocity for a barrel
BARREL_BASE_MIN_ANGLE: float = 26.0   # minimum launch angle window at BARREL_MIN_EV
BARREL_BASE_MAX_ANGLE: float = 30.0   # maximum launch angle window at BARREL_MIN_EV
# Blend weights (must sum to 1.0)
WEIGHT_OFFENSE: float = 0.45
WEIGHT_BASERUNNING: float = 0.15
WEIGHT_DEFENSE: float = 0.15
WEIGHT_SEASON_BASELINE: float = 0.20
WEIGHT_WAR: float = 0.05
# WAR normalisation: a typical MLB starter (~2 WAR) maps to a neutral 50 on the grade scale.
WAR_AVERAGE_STARTER: float = 2.0
WAR_SCALE_FACTOR: float = 10.0
WAR_BASELINE_SCORE: float = 50.0
# Heat-check thresholds (opponent recent-form stoplight on the Schedule tab).
HEAT_HOT_THRESHOLD: float = 80.0   # 80 %+ wins L10 → 🟢 Hot
HEAT_WARM_THRESHOLD: float = 50.0  # 50-79 % → 🟡 Warm; below → 🔴 Cold


def _diff_emoji(diff: float) -> str:
    """Return 🟢/🔴/🟡 emoji for a positive/negative/zero differential."""
    if diff > 0:
        return '🟢'
    if diff < 0:
        return '🔴'
    return '🟡'


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
    """Convert numeric score (0-100) to letter grade."""
    if score >= 97:
        return 'A+'
    if score >= 93:
        return 'A'
    if score >= 90:
        return 'A-'
    if score >= 87:
        return 'B+'
    if score >= 83:
        return 'B'
    if score >= 80:
        return 'B-'
    if score >= 75:
        return 'C'
    if score >= 70:
        return 'D'
    return 'F'


def _is_barrel(launch_speed: float, launch_angle: float) -> bool:
    """Return True if a batted ball qualifies as a Statcast barrel.

    Approximates the official definition: EV >= 98 mph with a launch angle
    window centred on 26-30 degrees that expands by ~1 degree per additional
    mph of exit velocity.
    """
    if launch_speed < BARREL_MIN_EV:
        return False
    extra = launch_speed - BARREL_MIN_EV
    min_angle = max(BARREL_BASE_MIN_ANGLE - extra, 8)
    max_angle = min(BARREL_BASE_MAX_ANGLE + extra, 50)
    return min_angle <= launch_angle <= max_angle


def _offense_score(
    avg_ev: float,
    hard_hit_pct: float,
    barrel_pct: float,
    xwoba: float,
    whiff_pct: float,
    walk_pct: float,
    tb_per_pa: float,
) -> float:
    """Compute offense component on a 0-100 scale from available Statcast metrics.

    Raw sub-components (sum to 100) are then lifted via a floor transformation
    so that league-average metrics produce a score in the D grade range (70-74)
    rather than the raw midpoint of ~45.

    Raw components: EV 0-20 | Hard-hit 0-20 | Barrel 0-15 | xwOBA 0-20 | Whiff 0-10 | Walk 0-10 | TB/PA 0-5
    """
    ev_cmp = min(max((avg_ev - 85) / 10, 0), 1) * 20
    hh_cmp = min(hard_hit_pct / HARD_HIT_CEILING, 1) * 20
    barrel_cmp = min(barrel_pct / BARREL_PCT_CEILING, 1) * 15
    xwoba_cmp = min(max((xwoba - 0.280) / 0.160, 0), 1) * 20
    whiff_cmp = max(0, 1 - whiff_pct / 100) * 10
    walk_cmp = min(walk_pct / 15, 1) * 10
    tb_cmp = min(tb_per_pa / 0.5, 1) * 5
    raw = ev_cmp + hh_cmp + barrel_cmp + xwoba_cmp + whiff_cmp + walk_cmp + tb_cmp
    # Lift into practical MLB range using calibration constants
    return round(min(GRADE_FLOOR_OFFSET + raw * GRADE_FLOOR_SCALE, 100), 1)


def _baserunning_score(events_series: pd.Series) -> float:
    """Compute baserunning component on a 0-100 scale.

    Returns a neutral 75 when no baserunning events exist (representing an
    average MLB baserunner rather than penalising for inactivity).
    Stolen-base success/failure adjusts the score between 25 and 90.
    """
    if events_series is None or events_series.empty:
        return 75.0
    sb_ok = int(events_series.isin({'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home'}).sum())
    cs = int(events_series.isin({'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home'}).sum())
    attempts = sb_ok + cs
    if attempts == 0:
        return 75.0
    sb_pct = sb_ok / attempts * 100
    # 100% SB success ≈ 90, 70% ≈ 70.5, 0% ≈ 25
    return round(min(max(25.0 + sb_pct * 0.65, 25.0), 90.0), 1)


def _defense_score(has_fielding_chances: bool = False, is_dh: bool = False) -> float:
    """Return defense component on a 0-100 scale.

    Currently returns a neutral 75 in all cases (representing an average MLB
    fielder).  The parameters are retained so callers can be upgraded to pass
    actual fielding data (OAA, FRV, assists/putouts/errors) when that data
    becomes available from the MLB API.

    Rules enforced:
    - DH is neutral (75), not penalised.
    - No fielding chances in a game blends toward the season baseline (75).
    - Never scored as zero just because a player had no opportunities.
    """
    return 75.0


def _season_baseline_score(
    avg_ev: float,
    hard_hit_pct: float,
    barrel_pct: float,
    xwoba: float,
) -> float:
    """Compute rolling season Statcast baseline on a 0-100 scale.

    Focuses on quality-of-contact metrics that stabilise over larger samples.
    Applies the same floor transformation as the offense component so that
    league-average contact quality maps to the 70-75 range.

    Raw components (sum to 100): EV 0-30 | Hard-hit 0-30 | Barrel 0-20 | xwOBA 0-20
    """
    ev_cmp = min(max((avg_ev - 85) / 10, 0), 1) * 30
    hh_cmp = min(hard_hit_pct / HARD_HIT_CEILING, 1) * 30
    barrel_cmp = min(barrel_pct / BARREL_PCT_CEILING, 1) * 20
    xwoba_cmp = min(max((xwoba - 0.280) / 0.160, 0), 1) * 20
    raw = ev_cmp + hh_cmp + barrel_cmp + xwoba_cmp
    return round(min(GRADE_FLOOR_OFFSET + raw * GRADE_FLOOR_SCALE, 100), 1)


def _war_normalized(war: float) -> float:
    """Normalize WAR to a 0-100 score for use as a bounded stabilising modifier.

    Scale: WAR -2 → 30, WAR 2 → 50 (average starter), WAR 5 → 80, WAR 8 → 100.
    WAR is centred at WAR_AVERAGE_STARTER so a typical MLB starter contributes a
    neutral WAR_BASELINE_SCORE, keeping it from distorting grades for players
    whose WAR data is unavailable.
    """
    return round(min(max(WAR_BASELINE_SCORE + (war - WAR_AVERAGE_STARTER) * WAR_SCALE_FACTOR, 0), 100), 1)


def _hitter_blended_score(
    offense: float,
    baserunning: float,
    defense: float,
    season_baseline: float,
    war: float,
) -> float:
    """Blend hitter component scores into a single 0-100 grade.

    Weights: 45% offense | 15% baserunning | 15% defense | 20% season baseline | 5% WAR.
    WAR is used only as a bounded stabilising modifier, not to dominate the grade.
    """
    war_score = _war_normalized(war)
    raw = (
        WEIGHT_OFFENSE * offense
        + WEIGHT_BASERUNNING * baserunning
        + WEIGHT_DEFENSE * defense
        + WEIGHT_SEASON_BASELINE * season_baseline
        + WEIGHT_WAR * war_score
    )
    return round(min(max(raw, 0), 100), 1)


def build_batter_grades_df(
    statcast_batter_df: pd.DataFrame,
    war_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-batter grade table from Statcast pitch-level data.

    Args:
        statcast_batter_df: Statcast rows for a batter group (one or more games).
        war_df: Optional DataFrame with columns ['Name', 'WAR'] for the season.
                When provided, WAR is used as a bounded stabilising modifier.
    """
    empty_cols = ['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Barrel %', 'Whiff %', 'xwOBA', 'Grade', 'Trend']
    if statcast_batter_df.empty:
        return pd.DataFrame(columns=empty_cols)

    df = statcast_batter_df.copy()
    df['Batter'] = _player_name_series(df)
    for col in ['launch_speed', 'estimated_woba_using_speedangle']:
        if col not in df.columns:
            df[col] = 0.0
    # Use pd.NA for missing launch_angle so barrel computation can skip those rows correctly.
    if 'launch_angle' not in df.columns:
        df['launch_angle'] = pd.NA
    for col in ['description', 'events']:
        if col not in df.columns:
            df[col] = ''

    # Any row with a measured exit velocity has a ball in play; include it.
    contact = df[df['launch_speed'] > 0].copy()
    if contact.empty:
        return pd.DataFrame(columns=empty_cols)

    # Build WAR lookup keyed by lower-cased player name (vectorised for performance)
    war_lookup: dict[str, float] = {}
    if war_df is not None and not war_df.empty and 'Name' in war_df.columns and 'WAR' in war_df.columns:
        war_lookup = {
            str(name).lower().strip(): coerce_float(val, WAR_AVERAGE_STARTER)
            for name, val in war_df.set_index('Name')['WAR'].items()
        }

    rows = []
    for batter, grp in contact.groupby('Batter'):
        bip = len(grp)

        # Exit velocity metrics
        avg_ev = float(grp['launch_speed'].mean())
        hard_hit_pct = float((grp['launch_speed'] >= 95).mean() * 100) if bip else 0.0

        # Barrel rate
        ev_vals = grp['launch_speed'].values
        la_vals = grp['launch_angle'].values
        barrel_count = sum(
            1 for ev, la in zip(ev_vals, la_vals)
            if pd.notna(ev) and pd.notna(la) and _is_barrel(float(ev), float(la))
        )
        barrel_pct = barrel_count / bip * 100 if bip else 0.0

        # xwOBA (exclude zero/null)
        xwoba_raw = grp['estimated_woba_using_speedangle'].replace(0, pd.NA).dropna()
        xwoba = float(xwoba_raw.mean()) if not xwoba_raw.empty else 0.0

        # All pitch rows for this batter (needed for swing/whiff and counting stats)
        all_pa = df[df['Batter'] == batter]
        pa = len(all_pa)

        # Swing / whiff
        swings = int(all_pa['description'].isin(SWING_DESCRIPTIONS).sum())
        whiffs = int(all_pa['description'].isin(WHIFF_DESCRIPTIONS).sum())
        whiff_pct = (whiffs / swings * 100) if swings else 0.0

        # Counting stats from events column
        events_s = all_pa['events'].fillna('')
        singles = int((events_s == 'single').sum())
        doubles = int((events_s == 'double').sum())
        triples = int((events_s == 'triple').sum())
        hrs = int((events_s == 'home_run').sum())
        walks = int(events_s.isin({'walk', 'intent_walk'}).sum())
        total_bases = singles + 2 * doubles + 3 * triples + 4 * hrs
        walk_pct = (walks / pa * 100) if pa else 0.0
        tb_per_pa = (total_bases / pa) if pa else 0.0

        # Component scores
        offense = _offense_score(avg_ev, hard_hit_pct, barrel_pct, xwoba, whiff_pct, walk_pct, tb_per_pa)
        baserunning = _baserunning_score(events_s)
        defense = _defense_score()
        season_baseline = _season_baseline_score(avg_ev, hard_hit_pct, barrel_pct, xwoba)
        war = war_lookup.get(batter.lower().strip(), WAR_AVERAGE_STARTER)

        score = _hitter_blended_score(offense, baserunning, defense, season_baseline, war)

        rows.append({
            'Batter': batter,
            'PA': pa,
            'Avg EV': round(avg_ev, 1),
            'Hard Hit %': round(hard_hit_pct, 1),
            'Barrel %': round(barrel_pct, 1),
            'Whiff %': round(whiff_pct, 1),
            'xwOBA': round(xwoba, 3),
            'Grade': _grade_from_score(score),
            'Trend': stoplight(score - 70, neutral_band=5),
            'Score': score,
        })

    out = pd.DataFrame(rows).sort_values(['Score', 'PA'], ascending=[False, False]).reset_index(drop=True)
    return out[['Batter', 'PA', 'Avg EV', 'Hard Hit %', 'Barrel %', 'Whiff %', 'xwOBA', 'Grade', 'Trend']]


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


def build_opponent_heat_df(
    upcoming_df: pd.DataFrame,
    season_df: pd.DataFrame,
    team_name: str,
    max_opponents: int = 3,
) -> pd.DataFrame:
    """Build next N unique opponents with heat check based on their last 10 games.

    Heat check stoplight:
    🟢 Hot  = 80 %+ wins in last 10
    🟡 Warm = 50-79 %
    🔴 Cold = 0-49 %
    """
    cols = ['Date', 'Opponent', 'Location', 'Opp L10 Record', 'Opp L10 Win%', 'Heat']
    if upcoming_df.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    seen: set[str] = set()
    for _, game in upcoming_df.iterrows():
        if len(rows) >= max_opponents:
            break
        if game.get('away') == team_name:
            opponent = game.get('home', '-')
            location = 'Away'
        else:
            opponent = game.get('away', '-')
            location = 'Home'

        if opponent in seen or opponent == '-':
            continue
        seen.add(opponent)

        opp_games = _team_games(season_df, opponent)
        opp_finals = opp_games[opp_games['is_final']].tail(10) if not opp_games.empty else pd.DataFrame()

        if not opp_finals.empty:
            opp_wins = int((opp_finals['result'] == 'W').sum())
            opp_total = len(opp_finals)
            opp_losses = opp_total - opp_wins
            win_pct = opp_wins / opp_total * 100 if opp_total else 0
            record = format_record(opp_wins, opp_losses)
        else:
            win_pct = 0.0
            record = '0-0'

        if win_pct >= HEAT_HOT_THRESHOLD:
            heat = '🟢 Hot'
        elif win_pct >= HEAT_WARM_THRESHOLD:
            heat = '🟡 Warm'
        else:
            heat = '🔴 Cold'

        rows.append({
            'Date': game.get('officialDate', '-'),
            'Opponent': opponent,
            'Location': location,
            'Opp L10 Record': record,
            'Opp L10 Win%': f'{win_pct:.0f}%',
            'Heat': heat,
        })
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_runs_per_inning_df(
    linescore_games: list[dict],
    team_name: str,
) -> pd.DataFrame:
    """Aggregate total runs scored for and against by inning across all completed games."""
    cols = ['Inning', 'Runs For', 'Runs Against', 'Differential', 'Heat']
    if not linescore_games:
        return pd.DataFrame(columns=cols)

    inning_data: dict[int, dict[str, int]] = {}
    for game in linescore_games:
        teams_obj = game.get('teams', {})
        away_team = (teams_obj.get('away') or {}).get('team', {}).get('name', '')
        home_team = (teams_obj.get('home') or {}).get('team', {}).get('name', '')
        is_away = away_team == team_name

        innings = (game.get('linescore') or {}).get('innings', [])
        for inn in innings:
            num = coerce_int(inn.get('num'), 0)
            if num <= 0:
                continue
            if num not in inning_data:
                inning_data[num] = {'for': 0, 'against': 0}
            if is_away:
                inning_data[num]['for'] += coerce_int((inn.get('away') or {}).get('runs'), 0)
                inning_data[num]['against'] += coerce_int((inn.get('home') or {}).get('runs'), 0)
            else:
                inning_data[num]['for'] += coerce_int((inn.get('home') or {}).get('runs'), 0)
                inning_data[num]['against'] += coerce_int((inn.get('away') or {}).get('runs'), 0)

    if not inning_data:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for num in sorted(inning_data):
        d = inning_data[num]
        diff = d['for'] - d['against']
        rows.append({
            'Inning': str(num),
            'Runs For': d['for'],
            'Runs Against': d['against'],
            'Differential': diff,
            'Heat': _diff_emoji(diff),
        })
    return pd.DataFrame(rows, columns=cols)


def build_change_since_last_game_df(season_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Compute metric changes between the last two completed games with stoplight."""
    cols = ['Metric', 'Last Game', 'Prev Game', 'Change', 'Signal']
    games = _team_games(season_df, team_name)
    if games.empty:
        return pd.DataFrame(columns=cols)
    finals = games[games['is_final']].copy()
    if len(finals) < 2:
        return pd.DataFrame(columns=cols)

    last = finals.iloc[-1]
    prev = finals.iloc[-2]

    rows: list[dict] = []
    delta_rs = int(last['team_runs']) - int(prev['team_runs'])
    rows.append({
        'Metric': 'Runs Scored',
        'Last Game': int(last['team_runs']),
        'Prev Game': int(prev['team_runs']),
        'Change': signed(delta_rs, 0),
        'Signal': stoplight(delta_rs, neutral_band=0),
    })

    delta_ra = int(last['opp_runs']) - int(prev['opp_runs'])
    rows.append({
        'Metric': 'Runs Allowed',
        'Last Game': int(last['opp_runs']),
        'Prev Game': int(prev['opp_runs']),
        'Change': signed(delta_ra, 0),
        'Signal': stoplight(-delta_ra, neutral_band=0),
    })

    last_diff = int(last['team_runs']) - int(last['opp_runs'])
    prev_diff = int(prev['team_runs']) - int(prev['opp_runs'])
    delta_diff = last_diff - prev_diff
    rows.append({
        'Metric': 'Run Differential',
        'Last Game': last_diff,
        'Prev Game': prev_diff,
        'Change': signed(delta_diff, 0),
        'Signal': stoplight(delta_diff, neutral_band=0),
    })

    rows.append({
        'Metric': 'Result',
        'Last Game': last['result'],
        'Prev Game': prev['result'],
        'Change': '-',
        'Signal': '🟢' if last['result'] == 'W' else '🔴',
    })
    return pd.DataFrame(rows, columns=cols)


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