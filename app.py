"""Live MLB Analytics Dashboard — fully live-data powered.

Tab layout
----------
1. Summary     — KPI cards, stoplights, recent games, runs by inning
2. Standings   — Division standings (all 6) + wildcard overview
3. Rankings    — Selected team rank board + MLB-wide stats tables
4. WAR Leaders — Top WAR leaders (FanGraphs primary, BR fallback)
5. Spray Chart — Player-selectable spray chart from Statcast
6. Home Runs   — Team HR tracker with home/away split and 2D/3D view
7. Live Feed   — Live game state for today
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

import mlb_api
import data_helpers as dh
import charts

# ─── App config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Live MLB Analytics Dashboard',
    page_icon='⚾',
    layout='wide',
    initial_sidebar_state='expanded',
)

SEASON = date.today().year
TODAY_STR = date.today().isoformat()
SEASON_START = f'{SEASON}-03-01'

# ─── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_teams() -> tuple[pd.DataFrame, str | None]:
    return mlb_api.load_teams()


@st.cache_data(ttl=900, show_spinner=False)
def _load_schedule(team_id: int, today: str) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.build_schedule_df(team_id, today)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_season(team_id: int, season: int, end: str) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.build_season_df(team_id, season, end)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_linescore(team_id: int, season: int, end: str) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.build_season_linescore_df(team_id, season, end)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_standings(season: int) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.get_division_standings(season)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_wildcard(season: int) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.get_wildcard_standings(season)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_all_team_stats(season: int) -> tuple[dict, str | None]:
    return mlb_api.get_all_teams_stats(season)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_war(season: int) -> tuple[pd.DataFrame, str | None]:
    return mlb_api.get_war_leaders(season, n=50)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_statcast_season(team_abbr: str, season: int) -> tuple[pd.DataFrame, str | None]:
    """Full-season Statcast batter data — lazy, cached per team."""
    return mlb_api.get_statcast_season_df(team_abbr, season, player_type='batter')


@st.cache_data(ttl=60, show_spinner=False)
def _load_live(game_pk: int | None) -> tuple[dict, str | None]:
    return mlb_api.get_live_summary(game_pk)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚾ Controls')

    teams_df, teams_err = _load_teams()
    team_names = teams_df['name'].tolist() if not teams_df.empty else ['Kansas City Royals']
    default_idx = team_names.index('Kansas City Royals') if 'Kansas City Royals' in team_names else 0
    selected_team = st.selectbox('Select Team', team_names, index=default_idx, key='sidebar_team')

    selected_date = st.date_input('Schedule Date', value=date.today(), key='sidebar_date')

    if st.button('🔄 Refresh All Data', use_container_width=True, key='sidebar_refresh'):
        st.cache_data.clear()
        st.rerun()

    if teams_err:
        st.caption(f'ℹ️ {teams_err}')
    st.caption(f'Season: {SEASON} | Data: MLB Stats API + Baseball Savant')

# ─── Team context ─────────────────────────────────────────────────────────────
team_row = dh.safe_team_row(teams_df, selected_team)
team_id = int(team_row['id']) if team_row else 118
team_abbr = str(team_row.get('abbreviation', 'KC')) if team_row else 'KC'
team_name = str(team_row.get('name', 'Kansas City Royals')) if team_row else 'Kansas City Royals'

logo_url = f'https://www.mlbstatic.com/team-logos/{team_id}.svg'

# ─── Load shared / fast data ──────────────────────────────────────────────────
today_str_for_date = selected_date.isoformat()
schedule_df, sched_err = _load_schedule(team_id, today_str_for_date)
season_df, season_err = _load_season(team_id, SEASON, TODAY_STR)
linescore_df, ls_err = _load_linescore(team_id, SEASON, TODAY_STR)
standings_df, stand_err = _load_standings(SEASON)
wildcard_df, wc_err = _load_wildcard(SEASON)
all_stats, stats_err = _load_all_team_stats(SEASON)

hitting_df = all_stats.get('hitting', pd.DataFrame())
pitching_df = all_stats.get('pitching', pd.DataFrame())
fielding_df = all_stats.get('fielding', pd.DataFrame())

# Pre-compute rankings (only when all_stats loaded)
hitting_rank_df = dh.build_mlb_hitting_rankings(hitting_df)
pitching_rank_df = dh.build_mlb_pitching_rankings(pitching_df)
fielding_rank_df = dh.build_mlb_fielding_rankings(fielding_df)
team_rank_df = dh.build_team_ranking_summary(team_id, hitting_rank_df, pitching_rank_df, fielding_rank_df)

# Team game snapshot
snapshot = dh.build_team_snapshot(team_row, season_df, schedule_df)
trend_df = dh.build_trend_df(season_df, team_name)
recent_df = dh.build_recent_games_df(season_df, team_name, count=15)
rolling_df = dh.build_team_rolling_df(recent_df)
kpi_cards = dh.build_kpi_cards(snapshot, trend_df)

# Live game
live_game_pk = mlb_api.choose_live_game_pk(schedule_df)

# ─── Header ───────────────────────────────────────────────────────────────────
hdr_col1, hdr_col2 = st.columns([0.08, 0.92])
with hdr_col1:
    try:
        st.image(logo_url, width=60)
    except Exception:
        pass
with hdr_col2:
    st.title(f'{team_name} — Live MLB Analytics')
    st.caption(f'Season {SEASON} | {TODAY_STR}')

# ─── KPI cards ────────────────────────────────────────────────────────────────
kpi_cols = st.columns(len(kpi_cards))
for i, card in enumerate(kpi_cards):
    kpi_cols[i].metric(
        label=card['label'],
        value=card['value'],
        delta=card['delta'],
    )

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
(
    tab_summary, tab_standings, tab_rankings,
    tab_war, tab_spray, tab_hr, tab_live,
) = st.tabs([
    '📊 Summary', '🏆 Standings', '📈 Rankings',
    '📋 WAR Leaders', '🎯 Spray Chart', '💥 Home Runs', '📡 Live Feed',
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if season_err:
        st.warning(f'Season data: {season_err}')

    # Trend table (stoplights only — no arrows)
    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        st.subheader('Recent Games')
        if not recent_df.empty:
            disp = recent_df.copy()
            for c in disp.columns:
                disp[c] = disp[c].astype(str)
            st.dataframe(disp, use_container_width=True, hide_index=True, key='recent_games_tbl')
        else:
            st.info('No completed games found yet this season.')

    with col_r:
        st.subheader('Season Trend Indicators')
        if not trend_df.empty:
            disp = trend_df[['Metric', 'Value', 'Trend']].copy()
            disp['Value'] = disp['Value'].astype(str)
            st.dataframe(disp, use_container_width=True, hide_index=True, key='trend_tbl')
        else:
            st.info('Not enough games played yet for trend analysis.')

    st.divider()

    # Runs by inning
    st.subheader('Season Runs by Inning')
    if linescore_df.empty and ls_err:
        st.warning(f'Inning data: {ls_err}')
    elif linescore_df.empty:
        st.info('Inning data not yet available — no completed games found with linescore data.')
    else:
        rbi_col1, rbi_col2 = st.columns([2, 1])
        with rbi_col1:
            loc_filter = st.radio(
                'Filter by location',
                options=['All', 'Home', 'Away'],
                horizontal=True,
                key='rbi_location_filter',
            )
        with rbi_col2:
            last_n_map = {'Full Season': None, 'Last 30 Games': 30, 'Last 15 Games': 15}
            last_n_label = st.selectbox('Games window', list(last_n_map.keys()), index=0, key='rbi_games_window')
            last_n = last_n_map[last_n_label]

        inning_agg = dh.build_runs_by_inning(linescore_df, location_filter=loc_filter, last_n=last_n)
        if inning_agg.empty:
            st.info('No inning data for the selected filter.')
        else:
            chart_mode = st.radio('Chart type', ['Bar Chart', 'Heatmap'], horizontal=True, key='rbi_chart_mode')
            if chart_mode == 'Bar Chart':
                charts.render_runs_by_inning_bar(inning_agg, key='rbi_bar_main')
            else:
                charts.render_runs_by_inning_heatmap(inning_agg, key='rbi_heatmap_main')

            # Summary table
            with st.expander('View inning data table'):
                disp = inning_agg.copy()
                for c in disp.columns:
                    disp[c] = disp[c].astype(str)
                st.dataframe(disp, use_container_width=True, hide_index=True, key='rbi_tbl')

    st.divider()

    # Rolling trend
    st.subheader('Rolling Scoring Trends')
    charts.render_recent_trend_chart(recent_df, key='summary_recent_trend')
    charts.render_run_diff_chart(recent_df, key='summary_run_diff')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_standings:
    if stand_err:
        st.warning(f'Standings: {stand_err}')

    if standings_df.empty:
        st.info('Division standings not available.')
    else:
        divisions = sorted(standings_df['division'].unique().tolist())
        al_divs = [d for d in divisions if d.startswith('AL')]
        nl_divs = [d for d in divisions if d.startswith('NL')]

        st.subheader('American League')
        al_cols = st.columns(len(al_divs)) if al_divs else []
        for i, div in enumerate(al_divs):
            with al_cols[i]:
                st.markdown(f'**{div}**')
                charts.render_standings_table(standings_df, div, key=f'stand_{div.replace(" ", "_")}')

        st.divider()
        st.subheader('National League')
        nl_cols = st.columns(len(nl_divs)) if nl_divs else []
        for i, div in enumerate(nl_divs):
            with nl_cols[i]:
                st.markdown(f'**{div}**')
                charts.render_standings_table(standings_df, div, key=f'stand_{div.replace(" ", "_")}')

    st.divider()
    st.subheader('Wild Card Overview')
    if wc_err:
        st.warning(f'Wildcard: {wc_err}')
    if not wildcard_df.empty:
        wc_disp = wildcard_df.copy()
        for c in wc_disp.columns:
            wc_disp[c] = wc_disp[c].astype(str)
        st.dataframe(wc_disp, use_container_width=True, hide_index=True, key='wc_tbl')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_rankings:
    if stats_err:
        st.warning(f'Team stats: {stats_err}')

    # ── Selected team summary ──
    st.subheader(f'📍 {team_name} — Rank Summary')
    if team_rank_df.empty:
        st.info('Team ranking data not available. Stats may not yet be published for this season.')
    else:
        charts.render_team_rank_cards(team_rank_df, team_name)

    st.divider()

    # ── MLB-wide rankings ──
    st.subheader('MLB-Wide Rankings')
    rank_tab_hit, rank_tab_pit, rank_tab_fld = st.tabs(['⚾ Hitting', '🥎 Pitching', '🧤 Fielding'])

    with rank_tab_hit:
        if hitting_rank_df.empty:
            st.info('Hitting rankings not available.')
        else:
            sort_col_h = st.selectbox(
                'Sort by',
                [c for c in ['OPS', 'AVG', 'Runs', 'HR', 'SB', 'OBP', 'SLG'] if c in hitting_rank_df.columns],
                index=0,
                key='rankings_hit_sort',
            )
            df_h = hitting_rank_df.copy().sort_values(sort_col_h, ascending=False)
            show_cols_h = [c for c in ['Team', 'Runs', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'SB', 'BB', 'K'] if c in df_h.columns]
            for c in show_cols_h:
                df_h[c] = df_h[c].astype(str)
            st.dataframe(df_h[show_cols_h], use_container_width=True, hide_index=True, key='rankings_hit_tbl')

    with rank_tab_pit:
        if pitching_rank_df.empty:
            st.info('Pitching rankings not available.')
        else:
            sort_col_p = st.selectbox(
                'Sort by',
                [c for c in ['ERA', 'WHIP', 'K', 'Saves'] if c in pitching_rank_df.columns],
                index=0,
                key='rankings_pit_sort',
            )
            ascending_p = sort_col_p in ('ERA', 'WHIP')
            df_p = pitching_rank_df.copy().sort_values(sort_col_p, ascending=ascending_p)
            show_cols_p = [c for c in ['Team', 'ERA', 'WHIP', 'K', 'BB', 'HR Allowed', 'Saves'] if c in df_p.columns]
            for c in show_cols_p:
                df_p[c] = df_p[c].astype(str)
            st.dataframe(df_p[show_cols_p], use_container_width=True, hide_index=True, key='rankings_pit_tbl')

    with rank_tab_fld:
        if fielding_rank_df.empty:
            st.info('Fielding rankings not available.')
        else:
            df_f = fielding_rank_df.copy().sort_values('FLD%', ascending=False) if 'FLD%' in fielding_rank_df.columns else fielding_rank_df.copy()
            show_cols_f = [c for c in ['Team', 'FLD%', 'Errors', 'DP'] if c in df_f.columns]
            for c in show_cols_f:
                df_f[c] = df_f[c].astype(str)
            st.dataframe(df_f[show_cols_f], use_container_width=True, hide_index=True, key='rankings_fld_tbl')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WAR LEADERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_war:
    st.subheader('WAR Leaders')
    war_raw, war_err = _load_war(SEASON)
    if war_err:
        st.info(f'ℹ️ Source note: {war_err}')

    if war_raw.empty:
        st.warning('WAR data is currently unavailable from all sources. Try refreshing later.')
    else:
        war_df = dh.build_war_leaderboard_df(war_raw, top_n=40)
        if not war_df.empty:
            source = war_df['Source'].iloc[0] if 'Source' in war_df.columns else 'Unknown'
            st.caption(f'Data source: **{source}**')

            # Filter by type if available
            if 'Type' in war_df.columns:
                type_options = ['All'] + sorted(war_df['Type'].unique().tolist())
                war_type_filter = st.radio('Player type', type_options, horizontal=True, key='war_type_filter')
                if war_type_filter != 'All':
                    war_df = war_df[war_df['Type'] == war_type_filter]

            charts.render_war_chart(war_df, key='war_chart_main')

            with st.expander('View WAR data table'):
                disp = war_df.copy()
                for c in disp.columns:
                    disp[c] = disp[c].astype(str)
                st.dataframe(disp, use_container_width=True, hide_index=True, key='war_tbl')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SPRAY CHART  (lazy-loaded Statcast)
# ══════════════════════════════════════════════════════════════════════════════
with tab_spray:
    st.subheader(f'Spray Chart — {team_name}')
    st.caption('Current-season batted-ball data from Baseball Savant (Statcast).')

    if st.button('Load Spray Chart Data', key='spray_load_btn', use_container_width=True):
        st.session_state['spray_loaded'] = True

    if st.session_state.get('spray_loaded', False):
        with st.spinner('Loading Statcast season data…'):
            statcast_df, sc_err = _load_statcast_season(team_abbr, SEASON)

        if sc_err:
            st.warning(f'Statcast note: {sc_err}')

        if statcast_df.empty:
            st.error(
                'No Statcast data returned for this team and season. '
                'Baseball Savant may be temporarily unavailable, or the season may not have started yet.'
            )
        else:
            # Player selector
            player_list = dh.get_spray_player_list(statcast_df)
            if not player_list:
                st.warning('No player names found in Statcast data.')
            else:
                spray_col1, spray_col2 = st.columns([2, 1])
                with spray_col1:
                    selected_player = st.selectbox(
                        'Select Batter',
                        player_list,
                        key='spray_player_select',
                    )
                with spray_col2:
                    event_filter = st.selectbox(
                        'Filter Events',
                        ['All', 'Hits', 'Extra-Base Hits', 'Home Runs'],
                        key='spray_event_filter',
                    )

                spray_df = dh.build_spray_df(statcast_df, player_name=selected_player, event_filter=event_filter)

                if spray_df.empty:
                    st.info(f'No batted-ball events found for {selected_player} with filter "{event_filter}".')
                else:
                    n_events = len(spray_df)
                    st.caption(f'Showing {n_events} events for {selected_player}')
                    charts.render_player_spray_chart(spray_df, selected_player, key='spray_chart_player')

                    with st.expander('View raw event data'):
                        disp = spray_df.drop(columns=['spray_x', 'spray_y'], errors='ignore')
                        for c in disp.columns:
                            disp[c] = disp[c].astype(str)
                        st.dataframe(disp, use_container_width=True, hide_index=True, key='spray_raw_tbl')
    else:
        st.info('Click "Load Spray Chart Data" to fetch current-season Statcast data. This may take 15–30 seconds.')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — HOME RUNS  (lazy-loaded Statcast)
# ══════════════════════════════════════════════════════════════════════════════
with tab_hr:
    st.subheader(f'Home Runs — {team_name} ({SEASON})')
    st.caption('Current-season home run events from Baseball Savant (Statcast).')

    if st.button('Load Home Run Data', key='hr_load_btn', use_container_width=True):
        st.session_state['hr_loaded'] = True

    if st.session_state.get('hr_loaded', False):
        with st.spinner('Loading Statcast season data…'):
            # Reuse cached data if already loaded by spray tab
            statcast_df_hr, sc_err_hr = _load_statcast_season(team_abbr, SEASON)

        if sc_err_hr:
            st.warning(f'Statcast note: {sc_err_hr}')

        if statcast_df_hr.empty:
            st.error(
                'No Statcast data returned. '
                'Baseball Savant may be temporarily unavailable, or the season may not have started yet.'
            )
        else:
            hr_loc_filter = st.radio(
                'Home/Away split',
                ['All', 'Home', 'Away'],
                horizontal=True,
                key='hr_location_filter',
            )

            full_hr_df = dh.build_hr_df(statcast_df_hr, location_filter='All')
            hr_df = dh.build_hr_df(statcast_df_hr, location_filter=hr_loc_filter)

            if hr_df.empty:
                st.info(f'No home runs found for filter "{hr_loc_filter}".')
            else:
                # Summary cards
                summary_cards = dh.build_hr_summary_cards(hr_df, statcast_df_hr)
                card_cols = st.columns(len(summary_cards))
                for i, card in enumerate(summary_cards):
                    card_cols[i].metric(label=card['label'], value=card['value'])

                st.divider()

                # Visualization mode
                viz_mode = st.radio(
                    'Visualization',
                    ['2D Field Chart', '3D Apex Chart'],
                    horizontal=True,
                    key='hr_viz_mode',
                )

                if viz_mode == '3D Apex Chart':
                    if charts._has_3d_fields(hr_df):
                        charts.render_team_hr_3d(hr_df, key='hr_3d_main')
                    else:
                        st.info('Insufficient fields for 3D view (need exit velocity, launch angle, and coordinates). Showing 2D chart.')
                        charts.render_team_hr_2d(hr_df, key='hr_2d_fallback')
                else:
                    charts.render_team_hr_2d(hr_df, key='hr_2d_main')

                # Distance/angle distributions
                st.divider()
                st.subheader('Distributions')
                charts.render_hr_distribution(hr_df, key='hr_dist_main')

                # Detail table
                with st.expander('View HR event table'):
                    disp = hr_df.copy()
                    for c in disp.columns:
                        disp[c] = disp[c].astype(str)
                    st.dataframe(disp, use_container_width=True, hide_index=True, key='hr_detail_tbl')
    else:
        st.info('Click "Load Home Run Data" to fetch current-season Statcast HR events.')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader('Live Game Feed')

    # Today's schedule
    if not schedule_df.empty:
        sch_display = dh.build_schedule_table(schedule_df, team_name)
        for c in sch_display.columns:
            sch_display[c] = sch_display[c].astype(str)
        st.dataframe(sch_display, use_container_width=True, hide_index=True, key='live_sched_tbl')
    else:
        st.info(f'No games scheduled for {today_str_for_date}.')

    if live_game_pk:
        st.success(f'🔴 Live game detected — Game PK: {live_game_pk}')
        with st.spinner('Fetching live game state…'):
            live_summary, live_err = _load_live(live_game_pk)
        if live_err:
            st.warning(f'Live feed error: {live_err}')
        if live_summary:
            live_box = dh.build_live_box_df(live_summary)
            for c in live_box.columns:
                live_box[c] = live_box[c].astype(str)
            st.dataframe(live_box, use_container_width=True, hide_index=True, key='live_box_tbl')

            # Recent plays
            recent_plays = live_summary.get('recent_plays', [])
            if recent_plays:
                st.subheader('Recent Play-by-Play')
                plays_df = pd.DataFrame(recent_plays)
                for c in plays_df.columns:
                    plays_df[c] = plays_df[c].astype(str)
                st.dataframe(plays_df, use_container_width=True, hide_index=True, key='live_plays_tbl')
    else:
        st.info(f'No live game in progress for {team_name} on {today_str_for_date}. Live updates activate when a game is in progress.')

    # Auto-refresh note
    if live_game_pk:
        st.caption('⏱ Live data — click Refresh to update.')

st.divider()
st.caption(
    'Data: MLB Stats API (schedule, standings, team stats) | '
    'Baseball Savant / Statcast (spray chart, HR events) | '
    'FanGraphs / Baseball-Reference (WAR). '
    'All data is current-season live data. No placeholder values.'
)
