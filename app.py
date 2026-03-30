from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from charts import (
    render_pitch_mix_chart,
    render_recent_trend_chart,
    render_rolling_chart,
    render_run_diff_chart,
    render_schedule_chart,
    render_statcast_scatter,
)
from data_helpers import (
    build_batter_grades_df,
    build_kpi_cards,
    build_live_box_df,
    build_pitch_mix_df,
    build_pitcher_grades_df,
    build_recent_games_df,
    build_schedule_table,
    build_statcast_summary_df,
    build_summary_df,
    build_team_rolling_df,
    build_team_snapshot,
    build_trend_df,
    safe_team_row,
)
from formatting import coerce_int
from mlb_api import (
    build_schedule_df,
    build_season_df,
    choose_live_game_pk,
    get_live_summary,
    get_statcast_team_df,
    load_teams,
)

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')


@st.cache_data(ttl=3600)
def cached_teams():
    return load_teams()


@st.cache_data(ttl=900)
def cached_daily(team_id: int, target_date: str):
    return build_schedule_df(team_id=team_id, target_date=target_date)


@st.cache_data(ttl=1800)
def cached_season(team_id: int, season: int, end_date: str):
    return build_season_df(team_id=team_id, season=season, end_date=end_date)


@st.cache_data(ttl=120)
def cached_live(game_pk: int | None):
    return get_live_summary(game_pk)


@st.cache_data(ttl=1800)
def cached_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str):
    return get_statcast_team_df(team_abbr=team_abbr, start_date=start_date, end_date=end_date, player_type=player_type)


st.title('Live MLB Analytics Dashboard')
st.caption('Stable Streamlit version built for GitHub + Streamlit Cloud deployment.')

with st.container(border=True):
    st.markdown(
        'This build keeps the stable flat-file deployment structure and adds deeper trends, player grades, and optional Statcast analytics. '
        'If MLB or Statcast endpoints are missing or empty, the app still loads and shows the rest of the dashboard.'
    )

teams_df, teams_warning = cached_teams()
if teams_warning:
    st.warning(teams_warning)

if teams_df.empty:
    st.error('Teams could not be loaded. The app shell is running, but no team list is available.')
    st.stop()

team_names = teams_df['name'].tolist()
default_team = 'Kansas City Royals' if 'Kansas City Royals' in team_names else team_names[0]

with st.sidebar:
    st.header('Controls')
    selected_team = st.selectbox('Select team', team_names, index=team_names.index(default_team))
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)
    if st.button('Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption('Deep trend tabs use recent team results plus optional Statcast data for exit velocity, spin, pitch use, and player grades.')

team_row = safe_team_row(teams_df, selected_team)
team_id = coerce_int(team_row.get('id') if team_row is not None else 0, 0)
team_abbr = str(team_row.get('abbreviation') if team_row else '').upper()
season = selected_date.year
selected_date_str = str(selected_date)
statcast_start = str(selected_date - timedelta(days=statcast_window))

# Core data
DailyResult = cached_daily(team_id, selected_date_str)
daily_df, daily_error = DailyResult
season_df, season_error = cached_season(team_id, season, selected_date_str)

if daily_error:
    st.warning(f'Daily schedule call had an issue: {daily_error}')
if season_error:
    st.warning(f'Season schedule call had an issue: {season_error}')

snapshot = build_team_snapshot(team_row, season_df, daily_df)
summary_df = build_summary_df(snapshot)
trend_df = build_trend_df(season_df, selected_team)
recent_games_df = build_recent_games_df(season_df, selected_team, count=10)
rolling_df = build_team_rolling_df(recent_games_df)
schedule_table = build_schedule_table(daily_df, selected_team)
kpi_cards = build_kpi_cards(snapshot, trend_df)

# Live feed
live_game_pk = choose_live_game_pk(daily_df)
live_summary, live_error = cached_live(live_game_pk)
if live_game_pk and live_error:
    st.caption('Live game feed is not available from MLB right now. Other tabs remain active.')

# Statcast optional
statcast_batter_df, statcast_batter_error = cached_statcast(team_abbr, statcast_start, selected_date_str, 'batter')
statcast_pitcher_df, statcast_pitcher_error = cached_statcast(team_abbr, statcast_start, selected_date_str, 'pitcher')

batter_grades_df = build_batter_grades_df(statcast_batter_df)
pitcher_grades_df = build_pitcher_grades_df(statcast_pitcher_df)
pitch_mix_df = build_pitch_mix_df(statcast_pitcher_df)
statcast_summary_df = build_statcast_summary_df(statcast_batter_df, statcast_pitcher_df)

cols = st.columns(4)
for col, item in zip(cols, kpi_cards):
    col.metric(item['label'], item['value'], item['delta'])

summary_tab, schedule_tab, trends_tab, deep_tab, live_tab = st.tabs(['Summary', 'Schedule', 'Trends', 'Deep Trends', 'Live Feed'])

with summary_tab:
    st.subheader('Team Summary')
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        if recent_games_df.empty:
            st.info('No recent completed games are available yet.')
        else:
            st.dataframe(recent_games_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader('Trend Indicators')
        if trend_df.empty:
            st.info('Trend data is not available yet.')
        else:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

with schedule_tab:
    st.subheader('Selected Date Schedule')
    if schedule_table.empty:
        st.info('No games found for the selected team and date.')
    else:
        st.dataframe(schedule_table, use_container_width=True, hide_index=True)
        render_schedule_chart(daily_df)

with trends_tab:
    st.subheader('Season Averages and Recent Trend')
    render_recent_trend_chart(recent_games_df)
    render_run_diff_chart(recent_games_df)
    st.subheader('Rolling Team Trend Table')
    if rolling_df.empty:
        st.info('Rolling trend data is not available yet.')
    else:
        st.dataframe(rolling_df, use_container_width=True, hide_index=True)
        render_rolling_chart(rolling_df)

with deep_tab:
    st.subheader('Deep Trend Analytics')
    st.caption(f'Statcast window: {statcast_start} through {selected_date_str}. These sections stay optional and do not block the app if Statcast is unavailable.')

    if statcast_batter_error and batter_grades_df.empty:
        st.info(f'Batter Statcast data is unavailable right now: {statcast_batter_error}')
    if statcast_pitcher_error and pitcher_grades_df.empty and pitch_mix_df.empty:
        st.info(f'Pitcher Statcast data is unavailable right now: {statcast_pitcher_error}')

    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.markdown('#### Team Statcast Snapshot')
        st.dataframe(statcast_summary_df, use_container_width=True, hide_index=True)
    with top_right:
        st.markdown('#### Pitch Type Use and Success')
        if pitch_mix_df.empty:
            st.info('Pitch mix is not available yet.')
        else:
            st.dataframe(pitch_mix_df, use_container_width=True, hide_index=True)
            render_pitch_mix_chart(pitch_mix_df)

    bottom_left, bottom_right = st.columns([1, 1])
    with bottom_left:
        st.markdown('#### Batter Rating Grades')
        if batter_grades_df.empty:
            st.info('Batter grades are not available yet.')
        else:
            st.dataframe(batter_grades_df, use_container_width=True, hide_index=True)
            render_statcast_scatter(batter_grades_df)
    with bottom_right:
        st.markdown('#### Pitcher Rating Grades')
        if pitcher_grades_df.empty:
            st.info('Pitcher grades are not available yet.')
        else:
            st.dataframe(pitcher_grades_df, use_container_width=True, hide_index=True)

with live_tab:
    st.subheader('Live Feed')
    live_df = build_live_box_df(live_summary)
    if live_df.empty:
        st.info('No active in-progress live game feed is available for the selected team and date.')
    else:
        st.dataframe(live_df, use_container_width=True, hide_index=True)

st.divider()
st.caption('Version v11: stable flat deployment, restored trend tabs, added player grades, stoplight grading, and optional exit velocity / spin / pitch-mix analytics.')
