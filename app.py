import streamlit as st
import pandas as pd
from datetime import date, timedelta

from mlb_api import (
    load_teams,
    build_schedule_df,
    build_season_df,
    choose_live_game_pk,
    get_live_summary,
    get_wildcard_standings,
    get_statcast_team_df,
)
from data_helpers import (
    safe_team_row,
    build_team_snapshot,
    build_summary_df,
    build_trend_df,
    build_recent_games_df,
    build_schedule_table,
    build_live_box_df,
    build_kpi_cards,
    build_team_rolling_df,
    build_batter_grades_df,
    build_pitcher_grades_df,
    build_pitch_mix_df,
    build_statcast_summary_df,
)
from charts import (
    render_schedule_chart,
    render_recent_trend_chart,
    render_run_diff_chart,
    render_rolling_chart,
    render_pitch_mix_chart,
    render_statcast_scatter,
    render_spray_chart,
)

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')

st.title('Live MLB Analytics Dashboard')
st.caption('Real-time MLB data powered by the MLB Stats API and Baseball Savant Statcast.')


# ---------------------------------------------------------------------------
# Cached data-loading helpers (low TTL for near real-time freshness)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _cached_teams() -> tuple:
    return load_teams()


@st.cache_data(ttl=120)
def _cached_schedule(team_id: int, date_str: str) -> tuple:
    return build_schedule_df(team_id, date_str)


@st.cache_data(ttl=600)
def _cached_season(team_id: int, season: int, end_date: str) -> tuple:
    return build_season_df(team_id, season, end_date)


@st.cache_data(ttl=900)
def _cached_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str) -> tuple:
    return get_statcast_team_df(team_abbr, start_date, end_date, player_type)


@st.cache_data(ttl=600)
def _cached_standings(season: int) -> tuple:
    return get_wildcard_standings(season)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header('Controls')
    teams_df, teams_err = _cached_teams()
    if teams_err:
        st.warning(teams_err)
    team_names = teams_df['name'].tolist() if not teams_df.empty else []
    default_idx = team_names.index('Kansas City Royals') if 'Kansas City Royals' in team_names else 0
    selected_team = st.selectbox('Select team', team_names, index=default_idx)
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)

    if st.button('Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption('All tabs load real MLB data automatically.')

# ---------------------------------------------------------------------------
# Resolve selected team
# ---------------------------------------------------------------------------

team_row = safe_team_row(teams_df, selected_team)
team_id: int = int((team_row or {}).get('id', 0))
team_abbr: str = str((team_row or {}).get('abbreviation', ''))
today_str = selected_date.strftime('%Y-%m-%d')
current_season = selected_date.year

# Load schedule and season data
schedule_df, schedule_err = _cached_schedule(team_id, today_str)
season_df, season_err = _cached_season(team_id, current_season, today_str)
standings_df, standings_err = _cached_standings(current_season)

if schedule_err:
    st.warning(f'Schedule: {schedule_err}')
if season_err:
    st.warning(f'Season data: {season_err}')

# Build derived data
snapshot = build_team_snapshot(team_row, season_df, schedule_df)
trend_df = build_trend_df(season_df, selected_team)
recent_games_df = build_recent_games_df(season_df, selected_team)
rolling_df = build_team_rolling_df(recent_games_df)
kpi_cards = build_kpi_cards(snapshot, trend_df)

# KPI header row
kpi_cols = st.columns(len(kpi_cards) + 1)
for col, card in zip(kpi_cols, kpi_cards):
    col.metric(card['label'], card['value'], card['delta'])

# Wildcard rank
wc_rank_raw = None
if not standings_df.empty and team_id:
    match = standings_df[standings_df['team_id'] == team_id]
    if not match.empty:
        wc_rank_raw = match.iloc[0].get('wildcard_rank')
wc_display = f'#{wc_rank_raw}' if (wc_rank_raw is not None and not pd.isna(wc_rank_raw)) else '-'
kpi_cols[-1].metric('Wildcard Rank', wc_display)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

summary_tab, schedule_tab, trends_tab, deep_tab, spray_tab, live_tab = st.tabs([
    'Summary', 'Schedule', 'Trends', 'Deep Trends', 'Spray Charts', 'Live Feed'
])

# --- Summary ---
with summary_tab:
    st.subheader('Team Summary')
    summary_df = build_summary_df(snapshot)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        st.dataframe(recent_games_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader('Trend Indicators')
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

# --- Schedule ---
with schedule_tab:
    st.subheader(f'Schedule for {today_str}')
    schedule_table = build_schedule_table(schedule_df, selected_team)
    if schedule_table.empty:
        st.info('No games found for the selected date.')
    else:
        st.dataframe(schedule_table, use_container_width=True, hide_index=True)
    render_schedule_chart(schedule_df)

# --- Trends ---
with trends_tab:
    st.subheader('Recent Runs Trend')
    render_recent_trend_chart(recent_games_df)
    render_run_diff_chart(recent_games_df)
    st.subheader('Rolling Averages')
    render_rolling_chart(rolling_df)

# --- Deep Trends (Statcast) ---
with deep_tab:
    st.subheader('Deep Trend Analytics')
    statcast_start = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
    st.caption(f'Statcast window: {statcast_start} through {today_str}')

    with st.spinner('Loading Statcast batter data...'):
        batter_df, batter_err = _cached_statcast(team_abbr, statcast_start, today_str, 'batter')
    with st.spinner('Loading Statcast pitcher data...'):
        pitcher_df, pitcher_err = _cached_statcast(team_abbr, statcast_start, today_str, 'pitcher')

    if batter_err:
        st.warning(f'Batter Statcast: {batter_err}')
    if pitcher_err:
        st.warning(f'Pitcher Statcast: {pitcher_err}')

    batter_grades = build_batter_grades_df(batter_df)
    pitcher_grades = build_pitcher_grades_df(pitcher_df)
    pitch_mix = build_pitch_mix_df(pitcher_df)
    statcast_summary = build_statcast_summary_df(batter_df, pitcher_df)

    st.subheader('Statcast Summary')
    st.dataframe(statcast_summary, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Batter Grades')
        st.dataframe(batter_grades, use_container_width=True, hide_index=True)
        render_statcast_scatter(batter_grades)
    with col2:
        st.markdown('#### Pitcher Grades')
        st.dataframe(pitcher_grades, use_container_width=True, hide_index=True)
        st.markdown('#### Pitch Type Mix')
        st.dataframe(pitch_mix, use_container_width=True, hide_index=True)
        render_pitch_mix_chart(pitch_mix)

# --- Spray Charts ---
with spray_tab:
    st.subheader(f'Spray Charts – Last {statcast_window} Days')
    st.caption('Hit locations showing offensive production and defensive performance.')

    with st.spinner('Loading spray chart data...'):
        off_df, off_err = _cached_statcast(team_abbr, statcast_start, today_str, 'batter')
        def_df, def_err = _cached_statcast(team_abbr, statcast_start, today_str, 'pitcher')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Team Hits (Offensive)')
        if off_err:
            st.warning(off_err)
        render_spray_chart(off_df, chart_type='offensive')
    with col2:
        st.markdown('#### Hits Allowed (Defensive)')
        if def_err:
            st.warning(def_err)
        render_spray_chart(def_df, chart_type='defensive')

# --- Live Feed ---
with live_tab:
    st.subheader('Live Game Feed')
    live_game_pk = choose_live_game_pk(schedule_df)
    if live_game_pk:
        with st.spinner('Fetching live game data...'):
            live_summary, live_err = get_live_summary(live_game_pk)
        if live_err:
            st.warning(live_err)
        live_box = build_live_box_df(live_summary)
        if not live_box.empty:
            st.dataframe(live_box, use_container_width=True, hide_index=True)
        else:
            st.info('Live game data not available.')
    else:
        st.info('No in-progress game found for the selected team and date.')

st.divider()
st.caption('Version v14: Real-time MLB Stats API data with auto-loaded charts, Statcast analytics, and live game feed.')