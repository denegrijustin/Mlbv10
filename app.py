from __future__ import annotations

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
    render_spray_chart,
)

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')

st.title('Live MLB Analytics Dashboard')
st.caption('Real-time data from the MLB Stats API and Baseball Savant (Statcast).')

with st.container(border=True):
    st.markdown(
        'This dashboard provides MLB analytics with Statcast data. '
        'All tabs load efficiently without blocking the main interface.'
    )


@st.cache_data(ttl=3600)
def cached_load_teams() -> tuple[pd.DataFrame, str | None]:
    return load_teams()


@st.cache_data(ttl=900)
def cached_schedule(team_id: int, target_date: str) -> tuple[pd.DataFrame, str | None]:
    return build_schedule_df(team_id, target_date)


@st.cache_data(ttl=1800)
def cached_season(team_id: int, season: int, end_date: str) -> tuple[pd.DataFrame, str | None]:
    return build_season_df(team_id, season, end_date)


@st.cache_data(ttl=60)
def cached_live_summary(game_pk: int | None) -> tuple[dict, str | None]:
    return get_live_summary(game_pk)


@st.cache_data(ttl=3600)
def cached_wildcard(season: int) -> tuple[pd.DataFrame, str | None]:
    return get_wildcard_standings(season)


@st.cache_data(ttl=1800)
def cached_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str) -> tuple[pd.DataFrame, str | None]:
    return get_statcast_team_df(team_abbr, start_date, end_date, player_type)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Controls')
    teams_df, teams_err = cached_load_teams()
    if teams_err:
        st.warning(teams_err)

    team_names = teams_df['name'].tolist() if not teams_df.empty else []
    if not team_names:
        st.error('No teams available. Check your network connection and refresh.')
        st.stop()
    default_idx = team_names.index('Kansas City Royals') if 'Kansas City Royals' in team_names else 0
    selected_team = st.selectbox('Select team', team_names, index=default_idx)
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)

    if st.button('Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption('Deep trend tabs use recent team results plus optional Statcast data.')


# ── Resolve team identifiers ─────────────────────────────────────────────────
team_row = safe_team_row(teams_df, selected_team)
team_id = int(team_row['id']) if team_row else 0
team_abbr = str(team_row.get('abbreviation', '')) if team_row else ''
today_str = selected_date.strftime('%Y-%m-%d')
current_season = selected_date.year

# ── Load data ────────────────────────────────────────────────────────────────
schedule_df, schedule_err = cached_schedule(team_id, today_str)
season_df, season_err = cached_season(team_id, current_season, today_str)
wc_df, wc_err = cached_wildcard(current_season)

game_pk = choose_live_game_pk(schedule_df)
live_summary, live_err = cached_live_summary(game_pk)

# ── Build derived frames ─────────────────────────────────────────────────────
snapshot = build_team_snapshot(team_row, season_df, schedule_df)
summary_df = build_summary_df(snapshot)
trend_df = build_trend_df(season_df, selected_team)
recent_games_df = build_recent_games_df(season_df, selected_team)
schedule_table = build_schedule_table(schedule_df, selected_team)
live_box_df = build_live_box_df(live_summary)
kpi_cards = build_kpi_cards(snapshot, trend_df)
rolling_df = build_team_rolling_df(recent_games_df)

# Wildcard rank for the selected team
wc_rank: str | None = None
if not wc_df.empty and team_id:
    wc_match = wc_df[wc_df['team_id'] == team_id]
    if not wc_match.empty:
        rank_val = wc_match.iloc[0].get('wildcard_rank')
        wc_rank = f'#{rank_val}' if rank_val else None

# ── KPI banner ───────────────────────────────────────────────────────────────
cols = st.columns(len(kpi_cards) + 1)
for i, card in enumerate(kpi_cards):
    cols[i].metric(card['label'], card['value'], card.get('delta'))
cols[len(kpi_cards)].metric('Wildcard Rank', wc_rank or 'N/A')

# Surface any API warnings beneath the KPI row
for err in filter(None, [schedule_err, season_err, wc_err]):
    st.warning(err)

# ── Tabs ─────────────────────────────────────────────────────────────────────
summary_tab, schedule_tab, trends_tab, deep_tab, spray_tab, live_tab = st.tabs([
    'Summary', 'Schedule', 'Trends', 'Deep Trends', 'Spray Charts', 'Live Feed'
])

with summary_tab:
    st.subheader('Team Summary')
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        st.dataframe(recent_games_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader('Trend Indicators')
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

with schedule_tab:
    st.subheader('Selected Date Schedule')
    st.dataframe(schedule_table, use_container_width=True, hide_index=True)
    render_schedule_chart(schedule_df)

with trends_tab:
    st.subheader('Season Averages and Recent Trend')
    render_recent_trend_chart(recent_games_df)
    render_run_diff_chart(recent_games_df)
    render_rolling_chart(rolling_df)

with deep_tab:
    st.subheader('Deep Trend Analytics')
    start_date_deep = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
    st.caption(f'Statcast window: {start_date_deep} through {today_str}')

    if st.button('Load Statcast Data'):
        with st.spinner('Loading Statcast data…'):
            batter_df, batter_err = cached_statcast(team_abbr, start_date_deep, today_str, 'batter')
            pitcher_df, pitcher_err = cached_statcast(team_abbr, start_date_deep, today_str, 'pitcher')

            for err in filter(None, [batter_err, pitcher_err]):
                st.warning(err)

            batter_grades = build_batter_grades_df(batter_df)
            pitcher_grades = build_pitcher_grades_df(pitcher_df)
            pitch_mix = build_pitch_mix_df(pitcher_df)
            statcast_summary = build_statcast_summary_df(batter_df, pitcher_df)

            st.subheader('Statcast Overview')
            st.dataframe(statcast_summary, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('#### Batter Metrics')
                st.dataframe(batter_grades, use_container_width=True, hide_index=True)
            with col2:
                st.markdown('#### Pitch Type Use')
                render_pitch_mix_chart(pitch_mix)
                st.dataframe(pitcher_grades, use_container_width=True, hide_index=True)

with spray_tab:
    st.subheader('Spray Charts – Last 21 Days')
    start_date_spray = (selected_date - timedelta(days=21)).strftime('%Y-%m-%d')
    st.caption(f'Hit locations: {start_date_spray} through {today_str}')

    if st.button('Load Spray Charts'):
        with st.spinner('Loading spray chart data…'):
            off_df, off_err = cached_statcast(team_abbr, start_date_spray, today_str, 'batter')
            def_df, def_err = cached_statcast(team_abbr, start_date_spray, today_str, 'pitcher')

            for err in filter(None, [off_err, def_err]):
                st.warning(err)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('#### Team Hits (Offensive)')
                render_spray_chart(off_df, 'offensive')
            with col2:
                st.markdown('#### Hits Allowed (Defensive)')
                render_spray_chart(def_df, 'defensive')

with live_tab:
    st.subheader('Live Feed')
    if live_err:
        st.warning(f'Live feed error: {live_err}')
    if live_box_df.empty:
        st.info('No active in-progress live game feed is available.')
    else:
        st.dataframe(live_box_df, use_container_width=True, hide_index=True)

    if st.button('Refresh Live Feed'):
        st.cache_data.clear()
        st.rerun()

st.divider()
st.caption('Version v14: Live data from MLB Stats API and Baseball Savant Statcast.')