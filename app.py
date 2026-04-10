"""Live MLB Analytics Dashboard – powered by real-time MLB Stats API data."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from mlb_api import (
    load_teams,
    build_schedule_df,
    build_season_df,
    choose_live_game_pk,
    get_live_summary,
    get_wildcard_standings,
    get_statcast_team_df,
    get_league_team_hitting_mlb,
    get_league_team_pitching_mlb,
    get_league_team_fielding_mlb,
    get_fg_team_batting,
    get_fg_team_fielding,
    get_fg_pitching_individual,
    get_fg_batting_individual,
    get_division_standings,
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
    build_offensive_rankings,
    build_defensive_rankings,
    build_starter_rankings,
    build_reliever_rankings,
    build_war_leaderboard,
    top_n_summary,
    build_runs_by_inning,
    build_standings_table,
)
from charts import (
    render_schedule_chart,
    render_recent_trend_chart,
    render_run_diff_chart,
    render_rolling_chart,
    render_pitch_mix_chart,
    render_statcast_scatter,
    render_spray_chart,
    render_ranking_bar,
    render_runs_by_inning,
)
from formatting import signed

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')
st.title('Live MLB Analytics Dashboard')
st.caption('Real-time data from the MLB Stats API, FanGraphs, and Baseball Savant.')

# ---------------------------------------------------------------------------
# Cached data loaders (shared data layer)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _load_teams():
    return load_teams()

@st.cache_data(ttl=900, show_spinner=False)
def _load_schedule(team_id: int, target_date: str):
    return build_schedule_df(team_id, target_date)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_season(team_id: int, season: int, end_date: str):
    return build_season_df(team_id, season, end_date)

@st.cache_data(ttl=60, show_spinner=False)
def _load_live_summary(game_pk: int):
    return get_live_summary(game_pk)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_wildcard(season: int):
    return get_wildcard_standings(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_statcast(team_abbr: str, start_date: str, end_date: str, player_type: str):
    return get_statcast_team_df(team_abbr, start_date, end_date, player_type)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_division_standings(season: int):
    return get_division_standings(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg_team_batting(season: int):
    return get_fg_team_batting(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_mlb_team_hitting(season: int):
    return get_league_team_hitting_mlb(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_mlb_team_fielding(season: int):
    return get_league_team_fielding_mlb(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg_team_fielding(season: int):
    return get_fg_team_fielding(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_mlb_team_pitching(season: int):
    return get_league_team_pitching_mlb(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg_pitching_individual(season: int):
    return get_fg_pitching_individual(season)

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg_batting_individual(season: int):
    return get_fg_batting_individual(season)

@st.cache_data(ttl=1800, show_spinner=False)
def _load_runs_by_inning(season_tuple, team_name: str):
    """Wrapper to make season_df hashable via tuple conversion."""
    season_df = pd.DataFrame(list(season_tuple[1:]), columns=season_tuple[0]) if season_tuple else pd.DataFrame()
    return build_runs_by_inning(season_df, team_name)

# ---------------------------------------------------------------------------
# Sidebar – team selection & controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header('Controls')
    teams_df, teams_err = _load_teams()
    if teams_err:
        st.warning(teams_err)

    if teams_df.empty:
        st.error('Unable to load MLB teams. Please try refreshing.')
        st.stop()

    team_names = teams_df['name'].tolist()
    default_idx = team_names.index('Kansas City Royals') if 'Kansas City Royals' in team_names else 0
    selected_team = st.selectbox('Select team', team_names, index=default_idx, key='team_sel')
    selected_date = st.date_input('Schedule date', value=date.today(), key='date_sel')
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7, key='sc_window')

    if st.button('Refresh Data', use_container_width=True, key='refresh_btn'):
        st.cache_data.clear()
        st.rerun()

    st.caption('All data is fetched live from the MLB Stats API, FanGraphs, and Baseball Savant.')

# ---------------------------------------------------------------------------
# Shared data orchestration – fetch once, reuse everywhere
# ---------------------------------------------------------------------------

team_row = safe_team_row(teams_df, selected_team)
team_id = (team_row or {}).get('id', 0)
team_abbr = (team_row or {}).get('abbreviation', '')
current_season = selected_date.year
today_str = selected_date.strftime('%Y-%m-%d')
season_end = today_str

# Core data loads
schedule_df, schedule_err = _load_schedule(team_id, today_str)
season_df, season_err = _load_season(team_id, current_season, season_end)

# Live game detection
live_game_pk = choose_live_game_pk(schedule_df)
live_summary, live_err = _load_live_summary(live_game_pk) if live_game_pk else ({}, None)

# Derived objects from shared data
snapshot = build_team_snapshot(team_row, season_df, schedule_df)
trend_df = build_trend_df(season_df, selected_team)
recent_df = build_recent_games_df(season_df, selected_team, count=10)
kpi_cards = build_kpi_cards(snapshot, trend_df)

# Team logo
LOGO_URL = f'https://www.mlbstatic.com/team-logos/{team_id}.svg' if team_id else ''

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

with st.container(border=True):
    logo_col, *kpi_cols = st.columns([0.6] + [1] * len(kpi_cards))
    with logo_col:
        if LOGO_URL:
            st.image(LOGO_URL, width=64)
        else:
            st.write('🏟️')
    for i, card in enumerate(kpi_cards):
        with kpi_cols[i]:
            st.metric(card['label'], card['value'], card.get('delta'))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(summary_tab, live_tab, standings_tab, hitting_tab, pitching_tab,
 spray_tab, trends_tab, rankings_tab) = st.tabs([
    '📊 Summary', '🔴 Live Feed', '🏆 Standings',
    '⚾ Hitting', '🎯 Pitching', '📍 Spray Charts',
    '📈 Trends', '📋 Rankings',
])

# ========================== SUMMARY TAB ====================================
with summary_tab:
    st.subheader(f'{selected_team} – Team Summary')
    if schedule_err and season_err:
        st.warning('Some data sources are temporarily unavailable.')
        if schedule_err:
            st.caption(f'Schedule: {schedule_err}')
        if season_err:
            st.caption(f'Season: {season_err}')

    try:
        summary = build_summary_df(snapshot)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f'Error building summary: {exc}')

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        if recent_df.empty:
            st.info('No completed games found for this season yet.')
        else:
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader('Trend Indicators')
        if trend_df.empty:
            st.info('Not enough games played for trend analysis.')
        else:
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

    # Today's schedule section
    if not schedule_df.empty:
        st.subheader("Today's Games")
        try:
            sched_table = build_schedule_table(schedule_df, selected_team)
            st.dataframe(sched_table, use_container_width=True, hide_index=True)
            render_schedule_chart(schedule_df, key='summary_schedule_chart')
        except Exception as exc:
            st.error(f'Error rendering schedule: {exc}')

    # Live game context
    if live_summary:
        st.subheader('Current Live Game')
        try:
            live_box = build_live_box_df(live_summary)
            st.dataframe(live_box, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f'Error rendering live box: {exc}')

# ========================== LIVE FEED TAB ==================================
with live_tab:
    st.subheader('Live Game Feed')

    if not live_game_pk:
        # Check for recent final games
        if not schedule_df.empty:
            final_today = schedule_df[schedule_df['status'].str.contains('Final', case=False, na=False)]
            if not final_today.empty:
                st.info('No game currently in progress. Most recent today:')
                sched_table = build_schedule_table(final_today, selected_team)
                st.dataframe(sched_table, use_container_width=True, hide_index=True)
            else:
                pregame = schedule_df[schedule_df['status'].str.contains('Pre-Game|Scheduled|Warmup', case=False, na=False)]
                if not pregame.empty:
                    st.info('Game not yet started. Scheduled:')
                    sched_table = build_schedule_table(pregame, selected_team)
                    st.dataframe(sched_table, use_container_width=True, hide_index=True)
                else:
                    st.info('No active in-progress live game feed is available for today.')
        else:
            st.info('No games scheduled for the selected date.')
    else:
        if live_err:
            st.warning(f'Live feed error: {live_err}')
        elif live_summary:
            st.success(f"🔴 LIVE: {live_summary.get('away_team', '-')} at {live_summary.get('home_team', '-')}")
            live_box = build_live_box_df(live_summary)
            st.dataframe(live_box, use_container_width=True, hide_index=True)

            # Manual refresh for live games
            if st.button('Refresh Live Feed', key='live_refresh_btn'):
                st.cache_data.clear()
                st.rerun()

# ========================== STANDINGS TAB ==================================
with standings_tab:
    st.subheader('MLB Standings')

    try:
        standings_data, standings_err = _load_division_standings(current_season)
        if standings_err:
            st.warning(f'Standings unavailable: {standings_err}')
        else:
            league_filter = st.radio(
                'League', ['All', 'AL', 'NL'],
                horizontal=True, key='standings_league',
            )
            filter_map = {'All': 'all', 'AL': 'AL', 'NL': 'NL'}
            standings_tbl = build_standings_table(standings_data, filter_map[league_filter])
            if standings_tbl.empty:
                st.info('No standings data available for this season.')
            else:
                for div_name, div_df in standings_tbl.groupby('Division', sort=False):
                    st.markdown(f'**{div_name}**')
                    display_cols = [c for c in div_df.columns if c not in ('Division', 'League')]
                    st.dataframe(
                        div_df[display_cols].reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )
    except Exception as exc:
        st.error(f'Error loading standings: {exc}')

    # Wildcard standings
    try:
        wc_df, wc_err = _load_wildcard(current_season)
        if wc_err:
            st.caption(f'Wildcard data: {wc_err}')
        elif not wc_df.empty:
            st.subheader('Wild Card Race')
            wc_display = wc_df[wc_df['wildcard_rank'].notna()].sort_values('wildcard_rank').head(12)
            if not wc_display.empty:
                st.dataframe(wc_display, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.caption(f'Wildcard error: {exc}')

# ========================== HITTING TAB ====================================
with hitting_tab:
    st.subheader(f'{selected_team} – Hitting Analysis')
    st.caption(f'Statcast data from the last {statcast_window} days.')

    try:
        sc_start = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
        sc_end = today_str
        bat_df, bat_err = _load_statcast(team_abbr, sc_start, sc_end, 'batter')

        if bat_err:
            st.warning(f'Hitting data: {bat_err}')
        elif bat_df.empty:
            st.info('No Statcast batting data available for this window.')
        else:
            grades_df = build_batter_grades_df(bat_df)
            if not grades_df.empty:
                st.markdown('#### Batter Quality of Contact')
                st.dataframe(grades_df, use_container_width=True, hide_index=True)
                render_statcast_scatter(grades_df, key='hitting_scatter')
            else:
                st.info('Not enough contact data for batter grades.')

            # Team hitting summary from MLB API
            try:
                mlb_hit, mlb_hit_err = _load_mlb_team_hitting(current_season)
                if not mlb_hit.empty and 'Team' in mlb_hit.columns:
                    team_hitting = mlb_hit[mlb_hit['Team'] == selected_team]
                    if not team_hitting.empty:
                        st.markdown('#### Season Hitting Stats (MLB Stats API)')
                        st.dataframe(team_hitting, use_container_width=True, hide_index=True)
            except Exception:
                pass

    except Exception as exc:
        st.error(f'Error loading hitting data: {exc}')

# ========================== PITCHING TAB ===================================
with pitching_tab:
    st.subheader(f'{selected_team} – Pitching Analysis')
    st.caption(f'Statcast data from the last {statcast_window} days.')

    try:
        sc_start = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
        sc_end = today_str
        pitch_df, pitch_err = _load_statcast(team_abbr, sc_start, sc_end, 'pitcher')

        if pitch_err:
            st.warning(f'Pitching data: {pitch_err}')
        elif pitch_df.empty:
            st.info('No Statcast pitching data available for this window.')
        else:
            grades_df = build_pitcher_grades_df(pitch_df)
            mix_df = build_pitch_mix_df(pitch_df)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('#### Pitcher Grades')
                if not grades_df.empty:
                    st.dataframe(grades_df, use_container_width=True, hide_index=True)
                else:
                    st.info('Not enough data for pitcher grades.')
            with col2:
                st.markdown('#### Pitch Mix')
                if not mix_df.empty:
                    st.dataframe(mix_df, use_container_width=True, hide_index=True)
                    render_pitch_mix_chart(mix_df, key='pitching_mix_chart')
                else:
                    st.info('Pitch mix data unavailable.')

            # Statcast team summary
            bat_for_summary, _ = _load_statcast(team_abbr, sc_start, sc_end, 'batter')
            sc_summary = build_statcast_summary_df(bat_for_summary, pitch_df)
            if not sc_summary.empty:
                st.markdown('#### Team Statcast Summary')
                st.dataframe(sc_summary, use_container_width=True, hide_index=True)

            # Team pitching stats from MLB API
            try:
                mlb_pitch, mlb_pitch_err = _load_mlb_team_pitching(current_season)
                if not mlb_pitch.empty and 'Team' in mlb_pitch.columns:
                    team_pitching = mlb_pitch[mlb_pitch['Team'] == selected_team]
                    if not team_pitching.empty:
                        st.markdown('#### Season Pitching Stats (MLB Stats API)')
                        st.dataframe(team_pitching, use_container_width=True, hide_index=True)
            except Exception:
                pass

    except Exception as exc:
        st.error(f'Error loading pitching data: {exc}')

# ========================== SPRAY CHARTS TAB ===============================
with spray_tab:
    st.subheader(f'{selected_team} – Spray Charts')
    st.caption(f'Hit locations from the last {statcast_window} days via Baseball Savant.')

    try:
        sc_start = (selected_date - timedelta(days=statcast_window)).strftime('%Y-%m-%d')
        sc_end = today_str

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Team Hits (Offensive)')
            bat_spray, bat_spray_err = _load_statcast(team_abbr, sc_start, sc_end, 'batter')
            if bat_spray_err:
                st.warning(f'Offensive spray data: {bat_spray_err}')
            else:
                render_spray_chart(bat_spray, 'offensive', key='spray_offense')

        with col2:
            st.markdown('#### Hits Allowed (Defensive)')
            pitch_spray, pitch_spray_err = _load_statcast(team_abbr, sc_start, sc_end, 'pitcher')
            if pitch_spray_err:
                st.warning(f'Defensive spray data: {pitch_spray_err}')
            else:
                render_spray_chart(pitch_spray, 'defensive', key='spray_defense')

    except Exception as exc:
        st.error(f'Error loading spray chart data: {exc}')

# ========================== TRENDS TAB =====================================
with trends_tab:
    st.subheader(f'{selected_team} – Season Trends')

    if recent_df.empty:
        st.info('No completed games yet for trend analysis.')
    else:
        try:
            render_recent_trend_chart(recent_df, key='trends_recent')
            render_run_diff_chart(recent_df, key='trends_run_diff')

            rolling = build_team_rolling_df(recent_df)
            render_rolling_chart(rolling, key='trends_rolling')
        except Exception as exc:
            st.error(f'Error rendering trends: {exc}')

    # Runs by inning section
    st.subheader('Season Runs by Inning')
    try:
        if not season_df.empty:
            # Convert to tuple for caching
            cols = tuple(season_df.columns.tolist())
            rows_tuple = tuple(season_df.itertuples(index=False, name=None))
            cache_key = (cols,) + rows_tuple if len(rows_tuple) <= 100 else (cols,) + rows_tuple[:50]
            # Direct call (already cached at season level)
            rf_inning, ra_inning = build_runs_by_inning(season_df, selected_team)
            if rf_inning or ra_inning:
                render_runs_by_inning(rf_inning, ra_inning, key='trends_runs_by_inning')
            else:
                st.info('Runs-by-inning data not yet available.')
        else:
            st.info('No season data available for runs-by-inning analysis.')
    except Exception as exc:
        st.error(f'Error loading runs by inning: {exc}')

# ========================== RANKINGS TAB ===================================
with rankings_tab:
    st.subheader('MLB Rankings')
    st.caption('League-wide team rankings across key offensive, defensive, and pitching categories.')

    _MIN_RANKING_SEASON = 2020
    ranking_season = st.selectbox(
        'Season', options=list(range(date.today().year, _MIN_RANKING_SEASON - 1, -1)),
        index=0, key='ranking_season',
    )

    off_sub, def_sub, sp_sub, rp_sub, war_sub = st.tabs([
        '⚾ Offense', '🛡️ Defense', '🎯 Starting Pitching',
        '💪 Relief Pitching', '🏆 WAR Leaders',
    ])

    # Shared data loads (cached)
    with st.spinner('Loading rankings data…'):
        fg_bat, fg_bat_err = _load_fg_team_batting(ranking_season)
        mlb_hit, mlb_hit_err = _load_mlb_team_hitting(ranking_season)
        mlb_field, mlb_field_err = _load_mlb_team_fielding(ranking_season)
        fg_field, fg_field_err = _load_fg_team_fielding(ranking_season)
        fg_pitch_ind, fg_pitch_ind_err = _load_fg_pitching_individual(ranking_season)
        mlb_pitch, mlb_pitch_err = _load_mlb_team_pitching(ranking_season)
        fg_bat_ind, fg_bat_ind_err = _load_fg_batting_individual(ranking_season)

    # Metrics where lower values are better
    _LOWER_IS_BETTER = frozenset({
        'ERA', 'WHIP', 'BB%', 'HR/9', 'FIP', 'Opp AVG', 'BS', 'E', 'K%',
    })

    def _sort_ascending(metric: str) -> bool:
        return metric in _LOWER_IS_BETTER

    def _display_cols(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c != 'Source']

    def _render_leader_cards(df: pd.DataFrame, metric: str, label: str = 'Team',
                              top_title: str = 'Top 5', bot_title: str = 'Bottom 5',
                              ascending_best: bool = True) -> None:
        left, right = st.columns(2)
        with left:
            st.markdown(f'**{top_title}**')
            top5 = top_n_summary(df, metric, n=5, ascending=ascending_best, label_col=label)
            if not top5.empty:
                st.dataframe(top5, use_container_width=True, hide_index=True)
            else:
                st.caption('No data')
        with right:
            st.markdown(f'**{bot_title}**')
            bot5 = top_n_summary(df, metric, n=5, ascending=not ascending_best, label_col=label)
            if not bot5.empty:
                st.dataframe(bot5, use_container_width=True, hide_index=True)
            else:
                st.caption('No data')

    # ---- Offense subtab
    with off_sub:
        st.markdown('#### Team Offensive Rankings')
        try:
            off_df = build_offensive_rankings(fg_bat, mlb_hit)
            if off_df.empty:
                st.warning('Offensive rankings data is currently unavailable.')
                if fg_bat_err:
                    st.caption(f'FanGraphs: {fg_bat_err}')
                if mlb_hit_err:
                    st.caption(f'MLB API: {mlb_hit_err}')
            else:
                src = off_df['Source'].iloc[0] if 'Source' in off_df.columns else 'Unknown'
                st.caption(f'Source: {src}')
                _render_leader_cards(off_df, 'OPS', top_title='Top 5 Offenses (OPS)',
                                      bot_title='Bottom 5 Offenses (OPS)', ascending_best=False)
                st.markdown('---')
                sort_off = st.selectbox(
                    'Sort by', [c for c in off_df.columns if c not in ('Rank', 'Team', 'Source')],
                    index=0, key='off_sort',
                )
                asc_off = _sort_ascending(sort_off)
                sorted_off = off_df.sort_values(sort_off, ascending=asc_off).reset_index(drop=True)
                sorted_off['Rank'] = range(1, len(sorted_off) + 1)
                st.dataframe(sorted_off[_display_cols(sorted_off)], use_container_width=True, hide_index=True)
                render_ranking_bar(off_df, sort_off, title=f'Top 10 – {sort_off}',
                                    ascending=not asc_off, key=f'off_bar_{sort_off}')
        except Exception as exc:
            st.error(f'Error loading offensive rankings: {exc}')

    # ---- Defense subtab
    with def_sub:
        st.markdown('#### Team Defensive Rankings')
        try:
            def_df = build_defensive_rankings(mlb_field, fg_field)
            if def_df.empty:
                st.warning('Defensive rankings data is currently unavailable.')
                if mlb_field_err:
                    st.caption(f'MLB API: {mlb_field_err}')
                if fg_field_err:
                    st.caption(f'FanGraphs: {fg_field_err}')
            else:
                src = def_df['Source'].iloc[0] if 'Source' in def_df.columns else 'Unknown'
                st.caption(f'Source: {src}')
                best_sort = 'Def' if 'Def' in def_df.columns else 'Fielding %'
                _render_leader_cards(def_df, best_sort,
                                      top_title='Best Defensive Teams',
                                      bot_title='Worst Defensive Teams',
                                      ascending_best=False)
                st.markdown('---')
                sort_def = st.selectbox(
                    'Sort by', [c for c in def_df.columns if c not in ('Rank', 'Team', 'Source')],
                    index=0, key='def_sort',
                )
                asc_def = _sort_ascending(sort_def)
                sorted_def = def_df.sort_values(sort_def, ascending=asc_def).reset_index(drop=True)
                sorted_def['Rank'] = range(1, len(sorted_def) + 1)
                st.dataframe(sorted_def[_display_cols(sorted_def)], use_container_width=True, hide_index=True)
                render_ranking_bar(def_df, sort_def, title=f'Top 10 – {sort_def}',
                                    ascending=not asc_def, key=f'def_bar_{sort_def}')
        except Exception as exc:
            st.error(f'Error loading defensive rankings: {exc}')

    # ---- Starting Pitching subtab
    with sp_sub:
        st.markdown('#### Team Starting Pitching Rankings')
        try:
            sp_df = build_starter_rankings(fg_pitch_ind, mlb_pitch)
            if sp_df.empty:
                st.warning('Starting pitching rankings data is currently unavailable.')
                if fg_pitch_ind_err:
                    st.caption(f'FanGraphs: {fg_pitch_ind_err}')
                if mlb_pitch_err:
                    st.caption(f'MLB API: {mlb_pitch_err}')
            else:
                src = sp_df['Source'].iloc[0] if 'Source' in sp_df.columns else 'Unknown'
                st.caption(f'Source: {src}')
                _render_leader_cards(sp_df, 'ERA',
                                      top_title='Best Starter Groups (ERA)',
                                      bot_title='Worst Starter Groups (ERA)',
                                      ascending_best=True)
                st.markdown('---')
                sort_sp = st.selectbox(
                    'Sort by', [c for c in sp_df.columns if c not in ('Rank', 'Team', 'Source')],
                    index=0, key='sp_sort',
                )
                asc_sp = _sort_ascending(sort_sp)
                sorted_sp = sp_df.sort_values(sort_sp, ascending=asc_sp).reset_index(drop=True)
                sorted_sp['Rank'] = range(1, len(sorted_sp) + 1)
                st.dataframe(sorted_sp[_display_cols(sorted_sp)], use_container_width=True, hide_index=True)
                render_ranking_bar(sp_df, sort_sp, title=f'Top 10 – {sort_sp}',
                                    ascending=not asc_sp, key=f'sp_bar_{sort_sp}')
        except Exception as exc:
            st.error(f'Error loading starter rankings: {exc}')

    # ---- Relief Pitching subtab
    with rp_sub:
        st.markdown('#### Team Relief Pitching Rankings')
        try:
            rp_df = build_reliever_rankings(fg_pitch_ind, mlb_pitch)
            if rp_df.empty:
                st.warning('Relief pitching rankings data is currently unavailable.')
                if fg_pitch_ind_err:
                    st.caption(f'FanGraphs: {fg_pitch_ind_err}')
                if mlb_pitch_err:
                    st.caption(f'MLB API: {mlb_pitch_err}')
            else:
                src = rp_df['Source'].iloc[0] if 'Source' in rp_df.columns else 'Unknown'
                st.caption(f'Source: {src}')
                _render_leader_cards(rp_df, 'ERA',
                                      top_title='Best Bullpens (ERA)',
                                      bot_title='Worst Bullpens (ERA)',
                                      ascending_best=True)
                st.markdown('---')
                sort_rp = st.selectbox(
                    'Sort by', [c for c in rp_df.columns if c not in ('Rank', 'Team', 'Source')],
                    index=0, key='rp_sort',
                )
                asc_rp = _sort_ascending(sort_rp)
                sorted_rp = rp_df.sort_values(sort_rp, ascending=asc_rp).reset_index(drop=True)
                sorted_rp['Rank'] = range(1, len(sorted_rp) + 1)
                st.dataframe(sorted_rp[_display_cols(sorted_rp)], use_container_width=True, hide_index=True)
                render_ranking_bar(rp_df, sort_rp, title=f'Top 10 – {sort_rp}',
                                    ascending=not asc_rp, key=f'rp_bar_{sort_rp}')
        except Exception as exc:
            st.error(f'Error loading reliever rankings: {exc}')

    # ---- WAR Leaders subtab
    with war_sub:
        st.markdown('#### WAR Leaders')
        try:
            war_filter = st.radio(
                'Filter', ['Overall', 'Hitters', 'Pitchers'],
                horizontal=True, key='war_filter',
            )
            _filter_map = {'Overall': 'all', 'Hitters': 'hitters', 'Pitchers': 'pitchers'}
            war_df = build_war_leaderboard(fg_bat_ind, fg_pitch_ind,
                                            filter_type=_filter_map[war_filter])
            if war_df.empty:
                st.warning('WAR leaderboard data is currently unavailable.')
                if fg_bat_ind_err:
                    st.caption(f'FanGraphs batting: {fg_bat_ind_err}')
                if fg_pitch_ind_err:
                    st.caption(f'FanGraphs pitching: {fg_pitch_ind_err}')
            else:
                src = war_df['Source'].iloc[0] if 'Source' in war_df.columns else 'Unknown'
                st.caption(f'Source: {src}')

                top_all = war_df.head(1)
                if not top_all.empty:
                    strip_cols = st.columns(3)
                    row0 = top_all.iloc[0]
                    strip_cols[0].metric('Current WAR Leader',
                                          f"{row0.get('Player', '-')} ({row0.get('Team', '-')})",
                                          f"{row0.get('WAR', 0):.1f} WAR")

                    hitter_war = build_war_leaderboard(fg_bat_ind, fg_pitch_ind, 'hitters')
                    if not hitter_war.empty:
                        h0 = hitter_war.iloc[0]
                        strip_cols[1].metric('Top Hitter WAR',
                                              f"{h0.get('Player', '-')} ({h0.get('Team', '-')})",
                                              f"{h0.get('WAR', 0):.1f} WAR")

                    pitcher_war = build_war_leaderboard(fg_bat_ind, fg_pitch_ind, 'pitchers')
                    if not pitcher_war.empty:
                        p0 = pitcher_war.iloc[0]
                        strip_cols[2].metric('Top Pitcher WAR',
                                              f"{p0.get('Player', '-')} ({p0.get('Team', '-')})",
                                              f"{p0.get('WAR', 0):.1f} WAR")

                st.markdown('---')
                n_show = st.slider('Show top N players', 10, 100, 25, 5, key='war_n')
                display_war = war_df.head(n_show)
                st.dataframe(display_war[_display_cols(display_war)],
                              use_container_width=True, hide_index=True)
                render_ranking_bar(display_war, 'WAR', label_col='Player',
                                    title=f'Top {min(10, n_show)} by WAR', n=min(10, n_show),
                                    ascending=False, key='war_bar')
        except Exception as exc:
            st.error(f'Error loading WAR leaderboard: {exc}')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    'Live MLB Analytics Dashboard – Data sources: MLB Stats API, FanGraphs (via pybaseball), '
    'Baseball Savant. All data is fetched live at runtime.'
)
