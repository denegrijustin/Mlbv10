import streamlit as st
import pandas as pd
from datetime import date, timedelta

from mlb_api import (
    load_teams, build_schedule_df, build_season_df,
    choose_live_game_pk, get_live_summary, get_wildcard_standings,
    get_statcast_team_df, fetch_war_df, build_upcoming_schedule_df,
    build_season_linescore,
)
from data_helpers import (
    safe_team_row, build_team_snapshot, build_summary_df,
    build_trend_df, build_recent_games_df, build_schedule_table,
    build_live_box_df, build_kpi_cards, build_team_rolling_df,
    build_batter_grades_df, build_pitcher_grades_df,
    build_pitch_mix_df, build_statcast_summary_df,
    build_opponent_heat_df, build_runs_per_inning_df,
    build_change_since_last_game_df,
)
from charts import (
    render_schedule_chart, render_recent_trend_chart,
    render_run_diff_chart, render_rolling_chart,
    render_pitch_mix_chart, render_statcast_scatter,
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

    selected_team = st.selectbox('Select team', teams_df['name'].tolist(), index=0)
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)

    if st.button('Refresh Data', use_container_width=True):
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
kpi_cards = build_kpi_cards(snapshot, trend_df)
change_df = build_change_since_last_game_df(season_df, team_name)

# Player grades – last 10 games
batter_grades_l10 = build_batter_grades_df(sc_batter_l10, war_df)
pitcher_grades_l10 = build_pitcher_grades_df(sc_pitcher_l10)

# Player grades – cumulative season
batter_grades_season = build_batter_grades_df(sc_batter_season, war_df)
pitcher_grades_season = build_pitcher_grades_df(sc_pitcher_season)

# Schedule heat check
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

# ── KPI Cards ────────────────────────────────────────────────────────────────

cols = st.columns(len(kpi_cards) + 1)
for col, card in zip(cols, kpi_cards):
    col.metric(card['label'], card['value'], card.get('delta', ''))
cols[-1].metric('Wildcard Rank', f'#{wc_rank}' if wc_rank and wc_rank != '-' else '-')

# Show any data-load warnings
for err_msg in [daily_err, season_err, upcoming_err, linescore_err]:
    if err_msg:
        st.warning(err_msg)

# ── Tabs ─────────────────────────────────────────────────────────────────────

summary_tab, schedule_tab, trends_tab, deep_tab, spray_tab, live_tab = st.tabs([
    'Summary', 'Schedule', 'Trends & Trackers', 'Deep Trends', 'Spray Charts', 'Live Feed',
])

# ---------- Summary ----------------------------------------------------------

with summary_tab:
    st.subheader('Team Summary')
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        if not recent_games_df.empty:
            st.dataframe(recent_games_df, use_container_width=True, hide_index=True)
            render_recent_trend_chart(recent_games_df)
        else:
            st.info('No completed games found yet this season.')
    with right:
        st.subheader('Trend Indicators')
        if not trend_df.empty:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
        else:
            st.info('Not enough data for trend analysis yet.')

# ---------- Schedule ----------------------------------------------------------

with schedule_tab:
    st.subheader("Today's Games")
    schedule_table = build_schedule_table(daily_df, team_name)
    if not schedule_table.empty:
        st.dataframe(schedule_table, use_container_width=True, hide_index=True)
        render_schedule_chart(daily_df)
    else:
        st.info('No games scheduled for the selected date.')

    st.subheader('Next 3 Opponents – Heat Check')
    st.caption('🟢 Hot (80%+ wins L10) · 🟡 Warm (50-79%) · 🔴 Cold (0-49%)')
    if not opponent_heat_df.empty:
        st.dataframe(opponent_heat_df, use_container_width=True, hide_index=True)
    else:
        st.info('No upcoming opponents found in the schedule window.')

# ---------- Trends & Trackers ------------------------------------------------

with trends_tab:
    st.subheader('Trends & Trackers')

    st.markdown('#### Change Since Last Game')
    if not change_df.empty:
        st.dataframe(change_df, use_container_width=True, hide_index=True)
    else:
        st.info('Need at least two completed games for change tracking.')

    if not trend_df.empty:
        st.markdown('#### Season Trend Indicators')
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

    if not rolling_df.empty:
        render_rolling_chart(rolling_df)

    if not recent_games_df.empty:
        render_run_diff_chart(recent_games_df)

    st.markdown('#### Runs Per Inning Tracker')
    if not rpi_df.empty:
        st.dataframe(rpi_df, use_container_width=True, hide_index=True)
        render_runs_per_inning_chart(rpi_df)
    else:
        st.info('No runs-per-inning data available yet.')

# ---------- Deep Trends -------------------------------------------------------

with deep_tab:
    st.subheader('Deep Trend Analytics')
    st.caption(f'L10 window: {l10_start} → {l10_end} · Season: {sc_season_start} → {sc_end}')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Batter Grades – Last 10 Games')
        if not batter_grades_l10.empty:
            st.dataframe(batter_grades_l10, use_container_width=True, hide_index=True)
            render_statcast_scatter(batter_grades_l10)
        else:
            st.info('No batter Statcast data available for this window.')
    with col2:
        st.markdown('#### Batter Grades – Cumulative (Season)')
        if not batter_grades_season.empty:
            st.dataframe(batter_grades_season, use_container_width=True, hide_index=True)
        else:
            st.info('No season batter data available.')

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('#### Pitcher Grades – Last 10 Games')
        if not pitcher_grades_l10.empty:
            st.dataframe(pitcher_grades_l10, use_container_width=True, hide_index=True)
        else:
            st.info('No pitcher Statcast data available for this window.')
    with col4:
        st.markdown('#### Pitcher Grades – Cumulative (Season)')
        if not pitcher_grades_season.empty:
            st.dataframe(pitcher_grades_season, use_container_width=True, hide_index=True)
        else:
            st.info('No season pitcher data available.')

    st.markdown('#### Pitch Type Mix')
    if not pitch_mix_df.empty:
        render_pitch_mix_chart(pitch_mix_df)
    else:
        st.info('No pitch mix data available.')

    st.markdown('#### Statcast Overview')
    if not statcast_summary_df.empty:
        st.dataframe(statcast_summary_df, use_container_width=True, hide_index=True)

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

# ---------- Live Feed ---------------------------------------------------------

with live_tab:
    st.subheader('Live Game Feed')
    if game_pk and live_summary:
        st.dataframe(live_box_df, use_container_width=True, hide_index=True)
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
    'Version v22: Real-time data from MLB Stats API and Statcast. '
    'Record as W-L, player grades for L10 and cumulative, functional live feed, '
    'schedule heat check, runs-per-inning tracker, no lazy loading.'
)