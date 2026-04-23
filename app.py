from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time

from mlb_api import (
    TEAM_META, DEFAULT_TEAM_ID, ALL_DIVISIONS,
    get_team_meta, get_team_logo, get_team_division, get_teams_in_division,
    load_teams, build_schedule_df, build_season_df,
    get_division_standings, get_wildcard_standings_league,
    get_upcoming_schedule, get_statcast_team_df,
    choose_live_game_pk, get_live_summary, get_wildcard_standings_combined,
    get_current_team_game, get_live_scorecard, get_game_plays, get_game_boxscore,
)
from data_helpers import (
    safe_team_row, build_team_snapshot, build_summary_df, build_trend_df,
    build_recent_games_df, build_schedule_table, build_live_box_df,
    build_kpi_cards, build_team_rolling_df,
    build_batter_grades_df, build_pitcher_grades_df, build_pitch_mix_df,
    build_statcast_summary_df,
    validate_required_columns, safe_display_value,
    compute_hitter_impact, compute_pitcher_impact, classify_pitcher_role,
    get_team_last_n_record, get_next_three_opponents,
    get_team_trend_snapshot, compare_teams,
    get_home_run_distance_summary, build_spray_chart_last_30,
    run_monte_carlo_matchup,
    render_scorecard_html,
    build_live_player_of_game, build_live_play_of_game,
    load_frozen_game_state, save_frozen_game_state,
    should_replace_frozen_state, build_frozen_snapshot,
)
from charts import (
    render_recent_trend_chart, render_run_diff_chart,
    render_rolling_chart, render_pitch_mix_chart,
    render_statcast_scatter, render_spray_chart,
    render_monte_carlo_results, render_division_standings_table,
    render_wildcard_standings_table, render_hr_3d_chart,
)
from formatting import coerce_float, coerce_int, format_record

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

def _fmt_series_date(date_str: str) -> str:
    """Format a YYYY-MM-DD string to 'Mon D' (e.g. 'Apr 5'), or return as-is on failure."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').strftime('%b %-d')
    except (ValueError, TypeError):
        return date_str or 'TBD'


st.set_page_config(page_title='MLB Analytics Dashboard', layout='wide', initial_sidebar_state='expanded')

st.markdown("""<style>
.block-container { padding-top: 2rem; padding-bottom: 1rem; }
.stMetric { background: rgba(255,255,255,0.08); border-radius: 6px; padding: 8px; }
.stMetric .stMetricLabel, .stMetric [data-testid="stMetricLabel"] { color: #CCCCCC !important; }
.stMetric .stMetricValue, .stMetric [data-testid="stMetricValue"] { color: #FFFFFF !important; }
h1 { font-size: 1.6rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.1rem !important; }
.stop-green { color: #27AE60; font-weight: bold; }
.stop-red { color: #E74C3C; font-weight: bold; }
.stop-yellow { color: #F39C12; font-weight: bold; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------

CURRENT_SEASON = date.today().year


@st.cache_data(ttl=3600)
def _cached_load_teams():
    return load_teams()


@st.cache_data(ttl=1800)
def _cached_build_season_df(team_id: int, season: int):
    end_date = date.today().isoformat()
    return build_season_df(team_id, season, end_date)


@st.cache_data(ttl=900)
def _cached_build_schedule_df(team_id: int, target_date: str):
    return build_schedule_df(team_id, target_date)


@st.cache_data(ttl=1800)
def _cached_get_division_standings(division: str, season: int):
    return get_division_standings(division, season)


@st.cache_data(ttl=1800)
def _cached_get_wildcard_standings(league: str, season: int):
    return get_wildcard_standings_league(league, season)


@st.cache_data(ttl=1800)
def _cached_get_upcoming_schedule(team_id: int, from_date: str, n_games: int = 10):
    return get_upcoming_schedule(team_id, from_date, n_games)


@st.cache_data(ttl=3600)
def _cached_get_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str = 'batter'):
    return get_statcast_team_df(team_abbr, start_date, end_date, player_type)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

all_teams = sorted(TEAM_META.values(), key=lambda t: t['full_name'])
team_names = [t['full_name'] for t in all_teams]
team_ids = [t['id'] for t in all_teams]

royals_name = 'Kansas City Royals'
default_idx = team_names.index(royals_name) if royals_name in team_names else 0

with st.sidebar:
    st.markdown('## ⚾ MLB Dashboard')

    selected_team_name = st.selectbox('Select Team', team_names, index=default_idx, key='selected_team')
    selected_team_id = team_ids[team_names.index(selected_team_name)]
    team_meta = get_team_meta(selected_team_id) or {}

    logo_url = team_meta.get('logo_url', '')
    if logo_url:
        st.image(logo_url, width=80)

    st.markdown(f"**{selected_team_name}**  \n{team_meta.get('division', '')}")

    statcast_window = st.slider('Statcast Lookback (days)', 7, 60, 30, 7)

    debug_mode = st.checkbox('Debug Mode', value=False)

    if st.button('🔄 Refresh All Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    today_str = date.today().isoformat()
    st.caption(f'Last updated: {today_str}')

# ---------------------------------------------------------------------------
# Load core season / schedule data
# ---------------------------------------------------------------------------

season_df, season_err = _cached_build_season_df(selected_team_id, CURRENT_SEASON)
daily_df, daily_err = _cached_build_schedule_df(selected_team_id, date.today().isoformat())

teams_df, teams_err = _cached_load_teams()
team_row = safe_team_row(teams_df, selected_team_name) or {
    'name': selected_team_name,
    'id': selected_team_id,
    'abbreviation': team_meta.get('abbreviation', ''),
    'division': team_meta.get('division', ''),
}
snapshot = build_team_snapshot(team_row, season_df, daily_df)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

header_cols = st.columns([1, 4, 2, 2, 2, 2, 2])
with header_cols[0]:
    if logo_url:
        st.image(logo_url, width=60)
with header_cols[1]:
    st.markdown(f"### {selected_team_name}")
    st.caption(team_meta.get('division', ''))
with header_cols[2]:
    st.metric('Record', snapshot.get('record', '0-0'))
with header_cols[3]:
    st.metric('Win %', f"{snapshot.get('win_pct', 0):.1f}%")
with header_cols[4]:
    st.metric('Run Diff', snapshot.get('run_diff', 0))
with header_cols[5]:
    st.metric('Runs/G', f"{snapshot.get('avg_runs_for', 0):.2f}")
with header_cols[6]:
    st.metric('RA/G', f"{snapshot.get('avg_runs_against', 0):.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_summary, tab_standings, tab_opponents, tab_players, tab_hr, tab_spray, tab_compare, tab_live = st.tabs([
    '📊 Summary',
    '🏆 Standings',
    '📅 Next Opponents',
    '👤 Player Impact',
    '💣 HR Tracker',
    '🎯 Spray Chart',
    '⚔️ Team Compare',
    '📡 Live Feed',
])

# ===========================================================================
# TAB 1: Summary
# ===========================================================================

with tab_summary:
    if debug_mode:
        if season_err:
            st.warning(f'Season data: {season_err}')
        if daily_err:
            st.warning(f'Daily schedule: {daily_err}')
        if teams_err:
            st.warning(f'Teams: {teams_err}')

    trend_df = build_trend_df(season_df, selected_team_name)
    recent_df = build_recent_games_df(season_df, selected_team_name, 10)

    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.markdown('#### 📈 Season Trend Metrics')
        if trend_df.empty:
            st.info('Season trend data is not yet available. Check back once games have been played.')
        else:
            with st.expander('ℹ️ Signal Legend'):
                st.markdown(
                    '🟢 **Up** — performing above the comparison baseline  \n'
                    '🔴 **Down** — performing below the comparison baseline  \n'
                    '🟡 **Even** — near the comparison baseline  \n'
                    '🟡 **Reference** — this row *is* the season baseline (no comparison)  \n'
                    '🏠 **Split** — home games subset, no head-to-head comparison  \n'
                    '✈️ **Split** — away games subset, no head-to-head comparison'
                )
            st.dataframe(trend_df.astype(str), use_container_width=True, hide_index=True)

        st.markdown('#### 🗓️ Recent 10 Games')
        if recent_df.empty:
            st.info('No completed games found yet this season.')
        else:
            st.dataframe(recent_df.astype(str), use_container_width=True, hide_index=True)

    with right_col:
        st.markdown('#### 🏃 Runs Trend')
        with st.spinner('Loading trend chart...'):
            render_recent_trend_chart(recent_df)

        st.markdown('#### ±  Run Differential')
        render_run_diff_chart(recent_df)

# ===========================================================================
# TAB 2: Standings
# ===========================================================================

with tab_standings:
    div_sub, wc_sub = st.tabs(['🏅 Division', '🃏 Wild Card'])

    with div_sub:
        team_div = team_meta.get('division', 'AL Central')
        all_div_options = ALL_DIVISIONS
        default_div_idx = all_div_options.index(team_div) if team_div in all_div_options else 0
        selected_division = st.selectbox('Division', all_div_options, index=default_div_idx)

        with st.spinner('Loading division standings...'):
            div_standings, div_err = _cached_get_division_standings(selected_division, CURRENT_SEASON)

        if debug_mode and div_err:
            st.warning(f'Division standings: {div_err}')

        if div_standings.empty:
            st.info(f'Division standings for {selected_division} are not yet available.')
        else:
            render_division_standings_table(div_standings, selected_team_id)

    with wc_sub:
        team_league = team_meta.get('league', 'AL')
        wc_league = st.radio('League', ['AL', 'NL'], index=0 if team_league == 'AL' else 1, horizontal=True)

        with st.spinner('Loading wild card standings...'):
            wc_df, wc_err = _cached_get_wildcard_standings(wc_league, CURRENT_SEASON)

        if debug_mode and wc_err:
            st.warning(f'Wild Card standings: {wc_err}')

        if wc_df.empty:
            st.info(f'{wc_league} Wild Card standings are not yet available.')
        else:
            render_wildcard_standings_table(wc_df, selected_team_id)

# ===========================================================================
# TAB 3: Next 3 Opponents
# ===========================================================================

with tab_opponents:
    with st.spinner('Loading upcoming schedule...'):
        upcoming_df, up_err = _cached_get_upcoming_schedule(selected_team_id, date.today().isoformat(), 20)

    if debug_mode and up_err:
        st.warning(f'Upcoming schedule: {up_err}')

    next_opponents = (
        get_next_three_opponents(upcoming_df, selected_team_name, date.today().isoformat())
        if not upcoming_df.empty else []
    )

    if not next_opponents:
        st.info('No upcoming scheduled games found. Schedule data may not be loaded yet for this date.')
    else:
        opp_cols = st.columns(min(len(next_opponents), 3))
        for i, opp in enumerate(next_opponents[:3]):
            with opp_cols[i]:
                with st.container(border=True):
                    opp_name = opp.get('team_name', '')
                    opp_meta = get_team_meta(opp_name) or {}
                    opp_logo = opp_meta.get('logo_url', '')
                    opp_id = opp_meta.get('id', 0)

                    col_logo, col_info = st.columns([1, 2])
                    with col_logo:
                        if opp_logo:
                            st.image(opp_logo, width=60)
                    with col_info:
                        st.markdown(f"**{opp_name}**")
                        # Format series date range
                        s_start = opp.get('series_start', '')
                        s_end = opp.get('series_end', '')
                        fmt_start = _fmt_series_date(s_start)
                        fmt_end = _fmt_series_date(s_end)
                        date_range = f"{fmt_start} – {fmt_end}" if fmt_end and fmt_end != fmt_start else fmt_start
                        st.caption(f"📅 {date_range}")
                        loc = 'Home' if opp.get('is_home') else 'Away'
                        num_games = opp.get('num_games', 1)
                        st.caption(f"📍 {loc} · {num_games} game{'s' if num_games != 1 else ''}")

                    if opp_id:
                        opp_season_df, _ = _cached_build_season_df(opp_id, CURRENT_SEASON)
                        rec = get_team_last_n_record(opp_season_df, opp_name, 10)
                        # Last game result
                        last_game_df = build_recent_games_df(opp_season_df, opp_name, 1)
                        if not last_game_df.empty:
                            lg = last_game_df.iloc[0]
                            lg_result = lg.get('Result', '')
                            lg_runs = lg.get('Team Runs', '')
                            lg_opp_runs = lg.get('Opp Runs', '')
                            last_game_str = f"{lg_result} {lg_runs}–{lg_opp_runs}" if lg_result else '—'
                        else:
                            last_game_str = '—'
                    else:
                        rec = {'wins': 0, 'losses': 0, 'win_pct': 0.0, 'sample_size': 0, 'games_played': 0}
                        last_game_str = '—'

                    st.caption(f"🏁 Last Game: **{last_game_str}**")

                    w = rec.get('wins', 0)
                    l = rec.get('losses', 0)
                    wp = rec.get('win_pct', 0.0)
                    sample = rec.get('games_played', 0)

                    if sample > 0:
                        if wp >= 0.600:
                            sl = '🟢'
                        elif wp >= 0.400:
                            sl = '🟡'
                        else:
                            sl = '🔴'
                        label = f"Last {sample}" if sample < 10 else "Last 10"
                        st.markdown(f"**🎯 {label} Record**")
                        st.metric(f"{sl} W-L", f"{w}-{l}", f"{wp:.3f}")
                    else:
                        st.info('No completed games found for this opponent yet.')

# ===========================================================================
# TAB 4: Player Impact
# ===========================================================================

with tab_players:
    abbr = team_meta.get('abbreviation', '')
    today = date.today()
    start_sc = (today - timedelta(days=statcast_window)).isoformat()
    end_sc = today.isoformat()

    with st.spinner('Loading Statcast data...'):
        sc_batter_df, sc_bat_err = _cached_get_statcast(abbr, start_sc, end_sc, 'batter')
        sc_pitcher_df, sc_pit_err = _cached_get_statcast(abbr, start_sc, end_sc, 'pitcher')

    hitter_tab, starter_tab, reliever_tab = st.tabs(['⚾ Hitters', '🎯 Starters', '💪 Relievers'])

    with hitter_tab:
        hitter_impact_df = compute_hitter_impact(sc_batter_df)
        if hitter_impact_df.empty:
            st.info('Hitter impact data unavailable. Try a larger lookback window or check back later.')
            if debug_mode and sc_bat_err:
                st.warning(sc_bat_err)
        else:
            st.caption(f'Hitter Impact Scores — last {statcast_window} days | {len(hitter_impact_df)} players')
            display_df = hitter_impact_df.copy()
            display_df['Impact'] = display_df['impact_score'].apply(lambda x: f'{x:.1f}')
            display_df['Stoplight'] = display_df['impact_score'].apply(
                lambda x: '🟢' if x >= 65 else ('🟡' if x >= 45 else '🔴')
            )
            show_cols = ['player_name', 'Stoplight', 'Impact', 'PA', 'avg_ev', 'hard_hit_pct', 'xwoba', 'bb_rate', 'k_rate', 'hr_count']
            available = [c for c in show_cols if c in display_df.columns]
            st.dataframe(
                display_df[available].rename(columns={
                    'player_name': 'Player', 'avg_ev': 'Avg EV', 'hard_hit_pct': 'HH%',
                    'xwoba': 'xwOBA', 'bb_rate': 'BB%', 'k_rate': 'K%', 'hr_count': 'HR',
                }).astype(str),
                use_container_width=True,
                hide_index=True,
            )

    with starter_tab:
        starter_df = compute_pitcher_impact(sc_pitcher_df, 'Starter')
        if starter_df.empty:
            st.info('Starter impact data unavailable.')
            if debug_mode and sc_pit_err:
                st.warning(sc_pit_err)
        else:
            st.caption(f'Starter Impact Scores — last {statcast_window} days')
            display_df = starter_df.copy()
            display_df['Impact'] = display_df['impact_score'].apply(lambda x: f'{x:.1f}')
            display_df['Stoplight'] = display_df['impact_score'].apply(
                lambda x: '🟢' if x >= 65 else ('🟡' if x >= 45 else '🔴')
            )
            show_cols = ['player_name', 'Stoplight', 'Impact', 'pitches', 'avg_velo', 'whiff_pct', 'k_rate', 'bb_rate', 'woba_allowed']
            available = [c for c in show_cols if c in display_df.columns]
            st.dataframe(
                display_df[available].rename(columns={
                    'player_name': 'Pitcher', 'avg_velo': 'Velo', 'whiff_pct': 'Whiff%',
                    'k_rate': 'K%', 'bb_rate': 'BB%', 'woba_allowed': 'wOBA Allow',
                }).astype(str),
                use_container_width=True,
                hide_index=True,
            )

    with reliever_tab:
        reliever_df = compute_pitcher_impact(sc_pitcher_df, 'Reliever')
        if reliever_df.empty:
            st.info('Reliever impact data unavailable.')
            if debug_mode and sc_pit_err:
                st.warning(sc_pit_err)
        else:
            st.caption(f'Reliever Impact Scores — last {statcast_window} days')
            display_df = reliever_df.copy()
            display_df['Impact'] = display_df['impact_score'].apply(lambda x: f'{x:.1f}')
            display_df['Stoplight'] = display_df['impact_score'].apply(
                lambda x: '🟢' if x >= 65 else ('🟡' if x >= 45 else '🔴')
            )
            show_cols = ['player_name', 'Stoplight', 'Impact', 'pitches', 'avg_velo', 'whiff_pct', 'k_rate', 'bb_rate', 'woba_allowed']
            available = [c for c in show_cols if c in display_df.columns]
            st.dataframe(
                display_df[available].rename(columns={
                    'player_name': 'Pitcher', 'avg_velo': 'Velo', 'whiff_pct': 'Whiff%',
                    'k_rate': 'K%', 'bb_rate': 'BB%', 'woba_allowed': 'wOBA Allow',
                }).astype(str),
                use_container_width=True,
                hide_index=True,
            )

# ===========================================================================
# TAB 5: HR Tracker
# ===========================================================================

with tab_hr:
    # Re-use batter statcast data loaded in tab_players context
    abbr = team_meta.get('abbreviation', '')
    today = date.today()
    start_sc = (today - timedelta(days=statcast_window)).isoformat()
    end_sc = today.isoformat()

    with st.spinner('Loading HR data...'):
        sc_batter_df_hr, sc_bat_err_hr = _cached_get_statcast(abbr, start_sc, end_sc, 'batter')

    hr_summary, hr_df = get_home_run_distance_summary(sc_batter_df_hr)

    col_home, col_away, col_total = st.columns(3)
    with col_home:
        with st.container(border=True):
            st.markdown('#### 🏠 Home HRs')
            n = hr_summary.get('home_hr_count', 0)
            dist = hr_summary.get('home_avg_distance', None)
            ev = hr_summary.get('home_avg_ev', None)
            st.metric('Count', n if n else 'N/A')
            st.metric('Avg Distance', f"{dist:.0f} ft" if dist else 'N/A')
            st.metric('Avg EV', f"{ev:.1f} mph" if ev else 'N/A')

    with col_away:
        with st.container(border=True):
            st.markdown('#### ✈️ Away HRs')
            n = hr_summary.get('away_hr_count', 0)
            dist = hr_summary.get('away_avg_distance', None)
            ev = hr_summary.get('away_avg_ev', None)
            st.metric('Count', n if n else 'N/A')
            st.metric('Avg Distance', f"{dist:.0f} ft" if dist else 'N/A')
            st.metric('Avg EV', f"{ev:.1f} mph" if ev else 'N/A')

    with col_total:
        with st.container(border=True):
            st.markdown('#### ⚾ All HRs')
            n = hr_summary.get('total_hr_count', 0)
            dist = hr_summary.get('total_avg_distance', None)
            max_h = hr_summary.get('max_height_ft', None)
            avg_h = hr_summary.get('avg_height_ft', None)
            st.metric('Total HRs', n if n else 'N/A')
            st.metric('Avg Distance', f"{dist:.0f} ft" if dist else 'N/A')
            st.metric('🏔️ Max Height', f"{max_h:.0f} ft" if max_h else 'N/A')
            st.metric('Avg Height', f"{avg_h:.0f} ft" if avg_h else 'N/A')

    if hr_summary.get('total_hr_count', 0) == 0:
        st.info('No home run data found in the selected Statcast window. Try increasing the lookback period.')

    # 3D trajectory chart
    st.markdown('---')
    st.markdown('#### 🚀 3D Home Run Flight Paths')
    render_hr_3d_chart(hr_df)

# ===========================================================================
# TAB 6: Spray Chart
# ===========================================================================

with tab_spray:
    abbr = team_meta.get('abbreviation', '')
    today = date.today()
    start_sc = (today - timedelta(days=statcast_window)).isoformat()
    end_sc = today.isoformat()

    with st.spinner('Loading spray chart data...'):
        sc_batter_df_spray, sc_bat_err_spray = _cached_get_statcast(abbr, start_sc, end_sc, 'batter')

    spray_df = build_spray_chart_last_30(sc_batter_df_spray, days=30)
    render_spray_chart(spray_df, title=f'{selected_team_name} — Spray Chart (Last 30 Games)')
    if spray_df.empty:
        if debug_mode and sc_bat_err_spray:
            st.warning(sc_bat_err_spray)

# ===========================================================================
# TAB 7: Team Comparison
# ===========================================================================

with tab_compare:
    all_team_names_sorted = sorted([m['full_name'] for m in TEAM_META.values()])
    team_a_default_idx = (
        all_team_names_sorted.index(selected_team_name) if selected_team_name in all_team_names_sorted else 0
    )
    team_b_default_idx = 0 if team_a_default_idx != 0 else 1

    col_a, col_window, col_b = st.columns([2, 1, 2])
    with col_a:
        team_a_name = st.selectbox('Team A', all_team_names_sorted, index=team_a_default_idx, key='compare_a')
    with col_b:
        team_b_name = st.selectbox('Team B', all_team_names_sorted, index=team_b_default_idx, key='compare_b')
    with col_window:
        games_window = st.selectbox('Window (games)', [7, 30, 60, 82, 162], index=1, key='games_window')

    team_a_meta_cmp = get_team_meta(team_a_name) or {}
    team_b_meta_cmp = get_team_meta(team_b_name) or {}
    team_a_id_cmp = team_a_meta_cmp.get('id', 0)
    team_b_id_cmp = team_b_meta_cmp.get('id', 0)

    with st.spinner('Loading comparison data...'):
        season_a, _ = _cached_build_season_df(team_a_id_cmp, CURRENT_SEASON) if team_a_id_cmp else (pd.DataFrame(), 'No ID')
        season_b, _ = _cached_build_season_df(team_b_id_cmp, CURRENT_SEASON) if team_b_id_cmp else (pd.DataFrame(), 'No ID')

    logo_col_a, vs_col, logo_col_b = st.columns([1, 1, 1])
    with logo_col_a:
        if team_a_meta_cmp.get('logo_url'):
            st.image(team_a_meta_cmp['logo_url'], width=80)
        st.markdown(f"**{team_a_name}**")
    with vs_col:
        st.markdown('### VS')
    with logo_col_b:
        if team_b_meta_cmp.get('logo_url'):
            st.image(team_b_meta_cmp['logo_url'], width=80)
        st.markdown(f"**{team_b_name}**")

    combined_season = (
        pd.concat([season_a, season_b], ignore_index=True)
        if not season_a.empty or not season_b.empty
        else pd.DataFrame()
    )
    cmp_df = compare_teams(combined_season, team_a_name, team_b_name, games_window)
    cmp_df = cmp_df.rename(columns={'Team_A': team_a_name, 'Team_B': team_b_name})

    if not cmp_df.empty:
        st.dataframe(cmp_df.astype(str), use_container_width=True, hide_index=True)
    else:
        st.info(f'Comparison data unavailable for {team_a_name} vs {team_b_name}.')

    st.markdown('---')
    st.markdown('#### 🎲 Monte Carlo Simulation')

    mc_col1, mc_col2 = st.columns([2, 1])
    with mc_col1:
        mc_window = st.selectbox('Simulation Window', [7, 30, 60, 82, 162], index=1, key='mc_window')
        home_options = ['Neutral', f'{team_a_name} Home', f'{team_b_name} Home']
        home_choice = st.selectbox('Home Field', home_options, index=0, key='mc_home')
    with mc_col2:
        n_sims = st.selectbox('Simulations', [1000, 5000, 10000], index=1, key='mc_sims')
        run_sim = st.button('Run Simulation ▶', use_container_width=True)

    if run_sim:
        snap_a = get_team_trend_snapshot(season_a, team_a_name, mc_window) if not season_a.empty else {}
        snap_b = get_team_trend_snapshot(season_b, team_b_name, mc_window) if not season_b.empty else {}

        if not snap_a or not snap_b:
            st.warning('Insufficient data for one or both teams. Try a different window or team.')
        else:
            home_team = None
            if home_choice == f'{team_a_name} Home':
                home_team = team_a_name
            elif home_choice == f'{team_b_name} Home':
                home_team = team_b_name

            with st.spinner(f'Running {n_sims:,} simulations...'):
                mc_result = run_monte_carlo_matchup(snap_a, snap_b, n_sims, home_team)

            render_monte_carlo_results(mc_result, team_a_meta_cmp, team_b_meta_cmp)

# ===========================================================================
# TAB 8: Live Feed — Full Scorecard + POTG + PLOTG
# ===========================================================================

_LIVE_REFRESH_INTERVAL = 30  # seconds

with tab_live:
    # -----------------------------------------------------------------------
    # Auto-refresh countdown
    # -----------------------------------------------------------------------
    refresh_placeholder = st.empty()

    # -----------------------------------------------------------------------
    # Fetch current game for the selected team
    # -----------------------------------------------------------------------
    with st.spinner('Fetching game data...'):
        current_game, game_err = get_current_team_game(selected_team_id)

    game_phase = current_game.get('game_phase', 'none') if current_game else 'none'
    game_pk = coerce_int(current_game.get('gamePk', 0), 0) if current_game else 0

    # -----------------------------------------------------------------------
    # Load frozen state
    # -----------------------------------------------------------------------
    frozen = load_frozen_game_state(selected_team_id)

    # -----------------------------------------------------------------------
    # Live/Final path: fetch full scorecard, plays, boxscore
    # -----------------------------------------------------------------------
    live_scorecard: dict = {}
    plays: list = []
    boxscore: dict = {}
    potg: dict = {}
    plotg: dict = {}
    fetch_err: str = ''

    if game_pk and game_phase in ('live', 'final'):
        with st.spinner('Loading full game data...'):
            live_scorecard, sc_err = get_live_scorecard(game_pk)
            if sc_err:
                fetch_err = sc_err
            else:
                plays, play_err = get_game_plays(game_pk)
                boxscore, box_err = get_game_boxscore(game_pk)
                if plays and boxscore:
                    game_feed_raw: dict = {'gameData': {'teams': {
                        'away': {'name': live_scorecard.get('away_team', ''), 'id': live_scorecard.get('away_id', 0)},
                        'home': {'name': live_scorecard.get('home_team', ''), 'id': live_scorecard.get('home_id', 0)},
                    }}}
                    potg = build_live_player_of_game(game_feed_raw, boxscore, plays)
                    plotg = build_live_play_of_game(game_feed_raw, plays)

        # Persist / update frozen state
        if live_scorecard and should_replace_frozen_state(frozen, current_game):
            snap = build_frozen_snapshot(selected_team_id, current_game, live_scorecard, potg, plotg)
            save_frozen_game_state(selected_team_id, snap)
            frozen = snap

    # -----------------------------------------------------------------------
    # Determine what to show
    # -----------------------------------------------------------------------
    # For 'scheduled' phase with no live scorecard, fall back to frozen
    show_scorecard = live_scorecard if live_scorecard else (frozen.get('scorecard') if frozen else {})
    show_potg = potg if potg and potg.get('name', 'N/A') != 'N/A' else (frozen.get('player_of_game') if frozen else {})
    show_plotg = plotg if plotg and plotg.get('batter', 'N/A') != 'N/A' else (frozen.get('play_of_game') if frozen else {})

    # -----------------------------------------------------------------------
    # Game header
    # -----------------------------------------------------------------------
    if show_scorecard:
        away = show_scorecard.get('away_team', 'Away')
        home = show_scorecard.get('home_team', 'Home')
        away_r = show_scorecard.get('away_runs', 0)
        home_r = show_scorecard.get('home_runs', 0)
        sc_status = show_scorecard.get('status', '')
        sc_abstract = show_scorecard.get('abstract_status', '')
        if sc_abstract == 'Live' or 'Progress' in sc_status:
            phase_badge = '🔴 LIVE'
        elif sc_abstract == 'Final' or 'Final' in sc_status:
            phase_badge = '✅ FINAL'
        else:
            phase_badge = '🗓️ SCHEDULED'

        h_col1, h_col2, h_col3 = st.columns([3, 2, 3])
        with h_col1:
            away_logo = f"https://www.mlbstatic.com/team-logos/{show_scorecard.get('away_id', 0)}.svg"
            st.image(away_logo, width=48)
            st.markdown(f"**{away}**")
        with h_col2:
            st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:bold'>{away_r} — {home_r}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;color:#aaa;font-size:0.85rem'>{phase_badge}</div>", unsafe_allow_html=True)
        with h_col3:
            home_logo = f"https://www.mlbstatic.com/team-logos/{show_scorecard.get('home_id', 0)}.svg"
            st.image(home_logo, width=48)
            st.markdown(f"**{home}**")
    elif current_game and game_phase == 'scheduled':
        st.markdown(f"### 🗓️ Next Game: {current_game.get('away', 'Away')} at {current_game.get('home', 'Home')}")
        st.caption(f"Date: {current_game.get('officialDate', current_game.get('gameDate', ''))[:10]}")
        if frozen and frozen.get('scorecard'):
            st.info('Showing frozen state from previous game.')
            show_scorecard = frozen.get('scorecard', {})
    else:
        st.info(f'No game data available for {selected_team_name}.')
        if debug_mode and game_err:
            st.warning(game_err)

    # -----------------------------------------------------------------------
    # Custom Scorecard
    # -----------------------------------------------------------------------
    if show_scorecard:
        st.markdown('#### 📊 Inning-by-Inning Scorecard')
        sc_html = render_scorecard_html(show_scorecard)
        st.markdown(sc_html, unsafe_allow_html=True)

        # Recent plays
        recent = show_scorecard.get('recent_plays') or []
        if recent:
            with st.expander('📋 Recent Plays', expanded=(game_phase == 'live')):
                for play_desc in reversed(recent[-8:]):
                    st.markdown(f'- {play_desc}')

        # Scoring plays (from plays list)
        if plays:
            scoring = [p for p in plays if p.get('is_scoring_play') and p.get('description')]
            if scoring:
                with st.expander('🏃 Scoring Plays', expanded=False):
                    for sp in scoring:
                        inn = sp.get('inning', 0)
                        half = 'Top' if sp.get('half_inning') == 'top' else 'Bot'
                        desc = sp.get('description', '')
                        after_a = sp.get('away_score_after', 0)
                        after_h = sp.get('home_score_after', 0)
                        st.markdown(f'- **{half} {inn}** ({after_a}–{after_h}): {desc}')

    # -----------------------------------------------------------------------
    # Player of the Game
    # -----------------------------------------------------------------------
    if show_potg and show_potg.get('name', 'N/A') != 'N/A':
        st.markdown('---')
        st.markdown('#### 🏆 Player of the Game')
        potg_col1, potg_col2 = st.columns([1, 3])
        with potg_col1:
            hs_url = show_potg.get('headshot_url', '')
            if hs_url:
                try:
                    st.image(hs_url, width=120)
                except Exception:
                    pass
        with potg_col2:
            potg_name = show_potg.get('name', 'N/A')
            potg_team = show_potg.get('team', '')
            potg_role = show_potg.get('role', '')
            potg_gis = coerce_float(show_potg.get('game_impact_score', 0), 0)
            potg_wpa = coerce_float(show_potg.get('wpa', 0), 0)
            potg_re24 = coerce_float(show_potg.get('re24', 0), 0)
            potg_summary = show_potg.get('summary', '')
            potg_stats = show_potg.get('stat_line') or {}

            st.markdown(f"**{potg_name}** — {potg_team}")
            st.caption(f"Role: {potg_role}")
            st.markdown(f"*{potg_summary}*")

            m1, m2, m3 = st.columns(3)
            m1.metric('Game Impact Score', f'{potg_gis:.1f}')
            m2.metric('WPA', f'{potg_wpa:+.3f}')
            m3.metric('RE24', f'{potg_re24:+.2f}')

            if potg_stats:
                stat_str = '&nbsp;&nbsp;'.join([f'**{k}:** {v}' for k, v in potg_stats.items()])
                st.markdown(f'<div style="font-size:0.85rem;color:#ccc">{stat_str}</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Play of the Game
    # -----------------------------------------------------------------------
    if show_plotg and show_plotg.get('batter', 'N/A') != 'N/A':
        st.markdown('---')
        st.markdown('#### 🎯 Play of the Game')

        inning_n = show_plotg.get('inning', 0)
        half_str = 'Top' if show_plotg.get('half_inning') == 'top' else 'Bot'
        outs_n = show_plotg.get('outs', 0)
        runners_list = show_plotg.get('start_runners') or []
        runners_map = {'1B': '1st', '2B': '2nd', '3B': '3rd'}
        runners_str = ', '.join([runners_map.get(r, r) for r in runners_list]) or 'Bases empty'
        sc_before = show_plotg.get('score_before', '0-0')
        sc_after = show_plotg.get('score_after', '0-0')
        plotg_batter = show_plotg.get('batter', 'N/A')
        plotg_pitcher = show_plotg.get('pitcher', 'N/A')
        plotg_event = show_plotg.get('event', '')
        plotg_desc = show_plotg.get('description', '')
        plotg_wpa = coerce_float(show_plotg.get('wpa', 0), 0)
        plotg_re24 = coerce_float(show_plotg.get('re24', 0), 0)
        plotg_narrative = show_plotg.get('narrative', '')

        with st.container(border=True):
            st.markdown(f"**{half_str} {inning_n}** &nbsp;|&nbsp; {outs_n} out(s) &nbsp;|&nbsp; {runners_str}")
            st.markdown(f"**{plotg_batter}** vs {plotg_pitcher}")
            st.markdown(f"**{plotg_event}**")
            if plotg_desc:
                st.markdown(f"*{plotg_desc}*")
            st.markdown(f"Score: {sc_before} → **{sc_after}**")

            st.markdown(f'<div style="font-size:0.8rem;color:#aaa;margin-top:8px">Why this was the Play of the Game:<br>'
                        f'WPA swing: {plotg_wpa:+.3f} &nbsp;|&nbsp; RE24 swing: {plotg_re24:+.2f}</div>',
                        unsafe_allow_html=True)

        # Runner-ups
        runner_ups = show_plotg.get('runner_ups') or []
        if runner_ups:
            with st.expander('Runner-up Plays', expanded=False):
                for i, ru in enumerate(runner_ups[:2], start=2):
                    tier = 'Second' if i == 2 else 'Third'
                    ru_inn = ru.get('inning', 0)
                    ru_half = 'Top' if ru.get('half_inning') == 'top' else 'Bot'
                    ru_event = ru.get('event', '')
                    ru_batter = ru.get('batter', '')
                    ru_wpa = coerce_float(ru.get('wpa', 0), 0)
                    ru_re24 = coerce_float(ru.get('re24', 0), 0)
                    st.markdown(f"**{tier} tier** — {ru_half} {ru_inn}: {ru_batter} — {ru_event}")
                    st.caption(f"WPA: {ru_wpa:+.3f} | RE24: {ru_re24:+.2f}")

    # -----------------------------------------------------------------------
    # Debug / error info
    # -----------------------------------------------------------------------
    if debug_mode:
        if fetch_err:
            st.warning(f'Scorecard fetch error: {fetch_err}')
        if game_err:
            st.warning(f'Game lookup error: {game_err}')
        if frozen:
            st.caption(f"Frozen state: gamePk={frozen.get('gamePk')} | frozen_at={frozen.get('frozen_at', '')[:19]}")

    # -----------------------------------------------------------------------
    # Auto-refresh for live games
    # -----------------------------------------------------------------------
    if game_phase == 'live':
        refresh_placeholder.caption(f'⟳ Auto-refreshing every {_LIVE_REFRESH_INTERVAL}s — Next refresh in {_LIVE_REFRESH_INTERVAL}s')
        time.sleep(_LIVE_REFRESH_INTERVAL)
        st.rerun()
    else:
        refresh_placeholder.caption('📡 Live Feed · Updates automatically during active games.')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(f'MLB Analytics Dashboard v27 · Season {CURRENT_SEASON} · Data from MLB Stats API & Baseball Savant')
