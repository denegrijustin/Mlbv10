import streamlit as st
import pandas as pd
from datetime import date, timedelta

from mlb_api import (
    load_teams, build_schedule_df, build_season_df,
    choose_live_game_pk, get_live_summary, get_wildcard_standings,
    get_statcast_team_df, fetch_war_df, build_upcoming_schedule_df,
    build_season_linescore, get_division_standings,
    get_team_division, get_team_league_id, get_team_logo,
)
from data_helpers import (
    safe_team_row, build_team_snapshot, build_summary_df,
    build_trend_df, build_recent_games_df,
    build_live_box_df, build_team_rolling_df,
    build_pitch_mix_df, build_statcast_summary_df,
    build_opponent_heat_df, build_runs_per_inning_df,
    build_change_since_last_game_df,
    build_division_standings_display, build_wildcard_top5_display,
    build_war_display_df, _player_name_series,
    build_player_impact_table, compare_teams,
    get_team_trend_snapshot,
)
from charts import (
    render_recent_trend_chart,
    render_run_diff_chart, render_rolling_chart,
    render_pitch_mix_chart,
    render_spray_chart, render_runs_per_inning_chart,
)

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')

st.title('Live MLB Analytics Dashboard')
st.caption('Real-time MLB analytics powered by live API data.')

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Controls')

    @st.cache_data(ttl=3600)
    def _cached_teams():
        return load_teams()

    teams_df, teams_err = _cached_teams()
    if teams_err:
        st.warning(teams_err)

    # Default to Kansas City Royals on first load
    team_names = teams_df['name'].tolist()
    default_idx = 0
    if 'Kansas City Royals' in team_names:
        default_idx = team_names.index('Kansas City Royals')

    if 'selected_team' not in st.session_state:
        st.session_state['selected_team'] = team_names[default_idx]

    selected_team = st.selectbox(
        'Select team', team_names,
        index=team_names.index(st.session_state['selected_team'])
        if st.session_state['selected_team'] in team_names else default_idx,
        key='selected_team',
    )
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)

    if st.button('Refresh Data'):
        st.cache_data.clear()
        st.rerun()

    st.caption('All data pulled from MLB Stats API and Statcast.')

# ── Resolve selected team ────────────────────────────────────────────────────

team_row = safe_team_row(teams_df, selected_team)
team_id = (team_row or {}).get('id', 0)
team_abbr = (team_row or {}).get('abbreviation', '')
team_name = (team_row or {}).get('name', selected_team)
season = selected_date.year
date_str = selected_date.strftime('%Y-%m-%d')

# ── Team logo ────────────────────────────────────────────────────────────────

if team_id:
    logo_url = f'https://www.mlbstatic.com/team-logos/{team_id}.svg'
    logo_col, title_col = st.columns([0.12, 0.88])
    with logo_col:
        st.image(logo_url, width=80)
    with title_col:
        st.subheader(team_name)

# ── Cached API loaders ───────────────────────────────────────────────────────


@st.cache_data(ttl=900)
def _cached_daily(tid, ds):
    return build_schedule_df(tid, ds)


@st.cache_data(ttl=1800)
def _cached_season(tid, s, ds):
    return build_season_df(tid, s, ds)


@st.cache_data(ttl=1800)
def _cached_upcoming(tid, ds):
    return build_upcoming_schedule_df(tid, ds)


@st.cache_data(ttl=1800)
def _cached_linescore(tid, s, ds):
    return build_season_linescore(tid, s, ds)


@st.cache_data(ttl=3600)
def _cached_statcast(abbr, start, end, player_type='batter'):
    return get_statcast_team_df(abbr, start, end, player_type)


@st.cache_data(ttl=3600)
def _cached_war(s):
    return fetch_war_df(s)


@st.cache_data(ttl=1800)
def _cached_standings(s):
    return get_wildcard_standings(s)


@st.cache_data(ttl=1800)
def _cached_division_standings(s, league_id):
    return get_division_standings(s, league_id)


# ── Fetch core data ──────────────────────────────────────────────────────────

daily_df, daily_err = _cached_daily(team_id, date_str)
season_df, season_err = _cached_season(team_id, season, date_str)
upcoming_df, upcoming_err = _cached_upcoming(team_id, date_str)
linescore_games, linescore_err = _cached_linescore(team_id, season, date_str)

# Statcast date windows
sc_end = date_str
sc_start = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
sc_season_start = f'{season}-01-01'

# L10 window: use actual last-10-game dates when possible
recent_for_window = build_recent_games_df(season_df, team_name, count=10)
if not recent_for_window.empty and len(recent_for_window) > 0:
    l10_start = str(recent_for_window['Date'].iloc[0])
    l10_end = str(recent_for_window['Date'].iloc[-1])
else:
    l10_start = sc_start
    l10_end = sc_end

# Statcast batter / pitcher data for L10 and season
sc_batter_l10, _ = _cached_statcast(team_abbr, l10_start, l10_end, 'batter')
sc_pitcher_l10, _ = _cached_statcast(team_abbr, l10_start, l10_end, 'pitcher')
sc_batter_season, _ = _cached_statcast(team_abbr, sc_season_start, sc_end, 'batter')
sc_pitcher_season, _ = _cached_statcast(team_abbr, sc_season_start, sc_end, 'pitcher')

war_df, war_err = _cached_war(season)

# ── Compute derived data ─────────────────────────────────────────────────────

snapshot = build_team_snapshot(team_row, season_df, daily_df)
summary_df = build_summary_df(snapshot)
trend_df = build_trend_df(season_df, team_name)
recent_games_df = build_recent_games_df(season_df, team_name, count=10)
rolling_df = build_team_rolling_df(recent_games_df)
change_df = build_change_since_last_game_df(season_df, team_name)

# Opponent heat check
opponent_heat_df = build_opponent_heat_df(upcoming_df, season_df, team_name)

# Runs per inning
rpi_df = build_runs_per_inning_df(linescore_games, team_name)

# Pitch mix (from L10 window)
pitch_mix_df = build_pitch_mix_df(sc_pitcher_l10)

# Statcast summary
statcast_summary_df = build_statcast_summary_df(sc_batter_l10, sc_pitcher_l10)

# Live feed
game_pk = choose_live_game_pk(daily_df)
live_summary, live_err = get_live_summary(game_pk)
live_box_df = build_live_box_df(live_summary)

# Wildcard standings
standings_df, standings_err = _cached_standings(season)
wc_rank = '-'
if not standings_df.empty and team_id:
    team_stand = standings_df[standings_df['team_id'] == team_id]
    if not team_stand.empty:
        wc_rank = team_stand.iloc[0].get('wildcard_rank', '-')

# Division standings (use correct league for selected team)
team_league_id = get_team_league_id(team_id)
team_division = get_team_division(team_id)
division_standings_df, div_err = _cached_division_standings(season, team_league_id)

# WAR display – extract player names from the team's Statcast data
team_player_names: list[str] = []
if not sc_batter_season.empty:
    team_player_names = _player_name_series(sc_batter_season).unique().tolist()
war_display_df = build_war_display_df(war_df, team_player_names if team_player_names else None)

# ── KPI Cards (2 per row) ────────────────────────────────────────────────────

# Row 1: Record + Season Run Diff
row1_left, row1_right = st.columns(2)
row1_left.metric('Record', snapshot.get('record', '0-0'), f"{snapshot.get('win_pct', 0.0)}% win pct")
run_diff = snapshot.get('run_diff', 0)
row1_right.metric('Season Run Diff', run_diff,
                   f'+{run_diff}' if run_diff >= 0 else str(run_diff))

# Row 2: Season Avg RF + Season Avg RA
lookup = {row['Metric']: row for _, row in trend_df.iterrows()} if not trend_df.empty else {}
row2_left, row2_right = st.columns(2)
row2_left.metric('Season Avg RF', snapshot.get('avg_runs_for', 0.0),
                  lookup.get('Last 5 Avg Runs For', {}).get('Trend', '🟡 0.00'))
row2_right.metric('Season Avg RA', snapshot.get('avg_runs_against', 0.0),
                   lookup.get('Last 5 Avg Runs Against', {}).get('Trend', '🟡 0.00'))

# Row 3: Wildcard Rank
wc_col, _ = st.columns(2)
wc_col.metric('Wildcard Rank', f'#{wc_rank}' if wc_rank and wc_rank != '-' else '-')

# Show any data-load warnings
for err_msg in [daily_err, season_err, upcoming_err, linescore_err]:
    if err_msg:
        st.warning(err_msg)

# ── Tabs ─────────────────────────────────────────────────────────────────────

summary_tab, trends_tab, deep_tab, spray_tab, compare_tab, live_tab = st.tabs([
    'Summary', 'Trends & Trackers', 'Deep Trends', 'Spray Charts',
    'Team Comparison', 'Live Feed',
])

# ---------- Summary ----------------------------------------------------------

with summary_tab:
    st.subheader('Team Summary')
    st.dataframe(summary_df, hide_index=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        if not recent_games_df.empty:
            st.dataframe(recent_games_df, hide_index=True)
            render_recent_trend_chart(recent_games_df)
        else:
            st.info('No completed games found yet this season.')
    with right:
        st.subheader('Trend Indicators')
        if not trend_df.empty:
            st.dataframe(trend_df, hide_index=True)
        else:
            st.info('Not enough data for trend analysis yet.')

    # Division Standings (correct division for selected team)
    div_short = team_division.split()[-1] if team_division != 'Unknown' else 'Central'
    div_label = team_division if team_division != 'Unknown' else 'AL Central'
    st.subheader(f'{div_label} Division Standings')
    al_central_display = build_division_standings_display(
        division_standings_df, div_short, highlight_team=team_name,
    )
    if not al_central_display.empty:
        # Render with logos as images
        if 'Logo' in al_central_display.columns:
            st.dataframe(
                al_central_display,
                hide_index=True,
                column_config={
                    'Logo': st.column_config.ImageColumn('Logo', width='small'),
                },
            )
        else:
            st.dataframe(al_central_display, hide_index=True)
    else:
        if div_err:
            st.warning(f'Could not load division standings: {div_err}')
        else:
            st.info('Division standings not available yet.')

    # Wildcard Top 5
    st.subheader('Wildcard Top 5')
    wc_top5_display = build_wildcard_top5_display(standings_df)
    if not wc_top5_display.empty:
        st.dataframe(wc_top5_display, hide_index=True)
    else:
        if standings_err:
            st.warning(f'Could not load wildcard standings: {standings_err}')
        else:
            st.info('Wildcard standings not available yet.')

    # Next 3 Opponents with logos and stoplights
    st.subheader('Next 3 Opponents – Heat Check')
    st.caption('🟢 Hot (60%+ wins L10) · 🟡 Warm (40-59%) · 🔴 Cold (<40%)')
    if not opponent_heat_df.empty:
        if 'Logo' in opponent_heat_df.columns:
            st.dataframe(
                opponent_heat_df,
                hide_index=True,
                column_config={
                    'Logo': st.column_config.ImageColumn('Logo', width='small'),
                },
            )
        else:
            st.dataframe(opponent_heat_df, hide_index=True)
    else:
        st.info('No upcoming opponents found in the schedule window.')

# ---------- Trends & Trackers ------------------------------------------------

with trends_tab:
    st.subheader('Trends & Trackers')

    st.markdown('#### Change Since Last Game')
    if not change_df.empty:
        st.dataframe(change_df, hide_index=True)
    else:
        st.info('Need at least two completed games for change tracking.')

    if not trend_df.empty:
        st.markdown('#### Season Trend Indicators')
        st.dataframe(trend_df, hide_index=True)

    if not rolling_df.empty:
        render_rolling_chart(rolling_df)

    if not recent_games_df.empty:
        render_run_diff_chart(recent_games_df)

    st.markdown('#### Runs Per Inning Tracker')
    if not rpi_df.empty:
        st.dataframe(rpi_df, hide_index=True)
        render_runs_per_inning_chart(rpi_df)
    else:
        st.info('No runs-per-inning data available yet.')

# ---------- Deep Trends -------------------------------------------------------

with deep_tab:
    st.subheader('Deep Trend Analytics')
    st.caption(f'L10 window: {l10_start} → {l10_end} · Season: {sc_season_start} → {sc_end}')

    # Player Impact Scores – Hitters
    st.markdown('#### Hitter Impact Scores')
    st.caption('Custom impact score (0-100) based on hits, XBH, run creation, contact quality, and strikeout penalty. WAR shown as season context only.')
    hitter_impact_df = build_player_impact_table(sc_batter_l10, 'batter', war_df)
    if not hitter_impact_df.empty:
        st.dataframe(hitter_impact_df, hide_index=True)
    else:
        st.info('No hitter impact data available for this window.')

    # Player Impact Scores – Pitchers
    st.markdown('#### Pitcher Impact Scores')
    st.caption('Custom impact score (0-100) based on IP, K, whiff rate, and penalty for ER/BB/H. WAR shown as season context only.')
    pitcher_impact_df = build_player_impact_table(sc_pitcher_l10, 'pitcher', war_df)
    if not pitcher_impact_df.empty:
        st.dataframe(pitcher_impact_df, hide_index=True)
    else:
        st.info('No pitcher impact data available for this window.')

    # WAR Table (season context)
    st.markdown('#### Player WAR (Season Context)')
    st.caption('WAR is a season-level stat and is used only as context, not game-level impact.')
    if not war_display_df.empty:
        st.dataframe(war_display_df, hide_index=True)
    else:
        if war_err:
            st.warning(f'WAR data unavailable: {war_err}')
        else:
            st.info('No WAR data available for this team.')

    st.markdown('#### Pitch Type Mix')
    if not pitch_mix_df.empty:
        render_pitch_mix_chart(pitch_mix_df)
    else:
        st.info('No pitch mix data available.')

    st.markdown('#### Statcast Overview')
    if not statcast_summary_df.empty:
        st.dataframe(statcast_summary_df, hide_index=True)

# ---------- Spray Charts ------------------------------------------------------

with spray_tab:
    st.subheader(f'Spray Charts – Last {statcast_window} Days')
    st.caption('Hit locations showing offensive production and defensive performance.')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Team Hits (Offensive)')
        render_spray_chart(sc_batter_l10, 'offensive')
    with col2:
        st.markdown('#### Hits Allowed (Defensive)')
        render_spray_chart(sc_pitcher_l10, 'defensive')

# ---------- Team Comparison ----------------------------------------------------

with compare_tab:
    st.subheader('Team Comparison')
    st.caption('Compare two teams across rolling windows of completed games.')

    cmp_left, cmp_right, cmp_window = st.columns([1, 1, 1])
    with cmp_left:
        cmp_team_a = st.selectbox(
            'Team A', team_names,
            index=team_names.index(selected_team) if selected_team in team_names else default_idx,
            key='cmp_team_a',
        )
    with cmp_right:
        other_idx = 0 if default_idx != 0 else min(1, len(team_names) - 1)
        cmp_team_b = st.selectbox(
            'Team B', team_names,
            index=other_idx,
            key='cmp_team_b',
        )
    with cmp_window:
        cmp_games_window = st.selectbox(
            'Rolling Window (games)',
            [7, 30, 60, 82, 162],
            index=1,
            key='cmp_window',
        )

    if cmp_team_a and cmp_team_b:
        comparison_df = compare_teams(season_df, cmp_team_a, cmp_team_b, cmp_games_window)
        if not comparison_df.empty:
            st.dataframe(comparison_df, hide_index=True)
        else:
            st.info('Not enough data to compare these teams yet.')

        # Multi-window trend view
        st.markdown('#### Multi-Window Trend Overview')
        st.caption('Shows both teams across all windows to spot recent hot streaks vs. long-term strength.')
        windows = [7, 30, 60, 82, 162]
        trend_rows: list[dict] = []
        for w in windows:
            snap_a = get_team_trend_snapshot(season_df, cmp_team_a, w)
            snap_b = get_team_trend_snapshot(season_df, cmp_team_b, w)
            trend_rows.append({
                'Window': f'Last {w}',
                f'{cmp_team_a} Win%': f"{snap_a.get('win_pct', 0):.3f}" if snap_a else 'N/A',
                f'{cmp_team_a} R/G': f"{snap_a.get('runs_per_game', 0):.2f}" if snap_a else 'N/A',
                f'{cmp_team_a} RA/G': f"{snap_a.get('runs_allowed_per_game', 0):.2f}" if snap_a else 'N/A',
                f'{cmp_team_a} Games': str(snap_a.get('sample_size', 0)) if snap_a else '0',
                f'{cmp_team_b} Win%': f"{snap_b.get('win_pct', 0):.3f}" if snap_b else 'N/A',
                f'{cmp_team_b} R/G': f"{snap_b.get('runs_per_game', 0):.2f}" if snap_b else 'N/A',
                f'{cmp_team_b} RA/G': f"{snap_b.get('runs_allowed_per_game', 0):.2f}" if snap_b else 'N/A',
                f'{cmp_team_b} Games': str(snap_b.get('sample_size', 0)) if snap_b else '0',
            })
        if trend_rows:
            trend_overview_df = pd.DataFrame(trend_rows)
            for c in trend_overview_df.columns:
                trend_overview_df[c] = trend_overview_df[c].astype(str)
            st.dataframe(trend_overview_df, hide_index=True)

# ---------- Live Feed ---------------------------------------------------------

with live_tab:
    st.subheader('Live Game Feed')
    if game_pk and live_summary:
        st.dataframe(live_box_df, hide_index=True)
        if st.button('Refresh Live Feed'):
            st.cache_data.clear()
            st.rerun()
    else:
        st.info('No active in-progress live game feed is available. Check back during game time!')
    if live_err:
        st.warning(f'Live feed issue: {live_err}')

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    'Version v24: Real-time data from MLB Stats API and Statcast. '
    'Player impact scoring, team comparison, corrected division standings, '
    'spray chart field view, team logos, and rolling trend analytics.'
)