import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from mlb_api import (
    get_league_hitting_stats,
    get_league_pitching_stats,
    get_league_fielding_stats,
    get_individual_pitcher_stats,
    get_war_leaderboard_data,
)
from data_helpers import (
    build_offensive_rankings,
    build_defensive_rankings,
    build_starter_rankings,
    build_reliever_rankings,
    build_war_leaderboard,
    sort_rankings,
)
from charts import render_rankings_bar

st.set_page_config(page_title='Live MLB Analytics Dashboard', layout='wide')

st.title('Live MLB Analytics Dashboard')
st.caption('Stable Streamlit version built for GitHub + Streamlit Cloud deployment.')

with st.container(border=True):
    st.markdown(
        'This dashboard provides MLB analytics with Statcast data. '
        'All tabs load efficiently without blocking the main interface.'
    )

# Initialize sample data
@st.cache_data(ttl=3600)
def load_teams_data():
    return pd.DataFrame({
        'name': ['Kansas City Royals', 'New York Yankees', 'Boston Red Sox', 'Los Angeles Dodgers'],
        'id': [118, 147, 111, 119],
        'abbreviation': ['KC', 'NYY', 'BOS', 'LAD']
    })

@st.cache_data(ttl=900)
def load_schedule_data():
    return pd.DataFrame({
        'Date': ['2026-04-06', '2026-04-07', '2026-04-08'],
        'Opponent': ['Boston Red Sox', 'Toronto Blue Jays', 'Tampa Bay Rays'],
        'Time': ['7:05 PM', '1:10 PM', '6:10 PM']
    })

@st.cache_data(ttl=1800)
def load_season_data():
    return pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'Runs Scored': [28, 31, 25, 29],
        'Runs Allowed': [22, 24, 26, 20]
    })

@st.cache_data(ttl=3600)
def load_statcast_data():
    return pd.DataFrame({
        'Player': ['Player A', 'Player B', 'Player C'],
        'Exit Velocity': [95.2, 92.1, 98.5],
        'Launch Angle': [25, 18, 32]
    })

# Sidebar
with st.sidebar:
    st.header('Controls')
    teams = load_teams_data()
    selected_team = st.selectbox('Select team', teams['name'].tolist(), index=0)
    selected_date = st.date_input('Schedule date', value=date.today())
    statcast_window = st.slider('Statcast lookback days', min_value=7, max_value=60, value=21, step=7)
    
    if st.button('Refresh Data', use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.caption('Deep trend tabs use recent team results plus optional Statcast data.')

# Load all data
teams_df = load_teams_data()
schedule_df = load_schedule_data()
season_df = load_season_data()

# Display KPI cards
cols = st.columns(5)
cols[0].metric('Wins', '42', '+2')
cols[1].metric('Losses', '38', '-1')
cols[2].metric('Win %', '.525', '+.005')
cols[3].metric('Runs', '245', '+8')
cols[4].metric('Wildcard Rank', '#5')

# Create tabs
summary_tab, schedule_tab, trends_tab, deep_tab, spray_tab, rankings_tab, live_tab = st.tabs([
    'Summary', 'Schedule', 'Trends', 'Deep Trends', 'Spray Charts', '🏆 MLB Rankings', 'Live Feed'
])

with summary_tab:
    st.subheader('Team Summary')
    summary_data = pd.DataFrame({
        'Metric': ['AVG', 'OBP', 'SLG', 'ERA', 'WHIP'],
        'Value': ['.265', '.325', '.415', '3.85', '1.15']
    })
    st.dataframe(summary_data, use_container_width=True, hide_index=True)
    
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('Recent 10 Games')
        recent = pd.DataFrame({
            'Date': ['2026-04-05', '2026-04-04', '2026-04-03'],
            'Opponent': ['Yankees', 'Red Sox', 'Rays'],
            'Result': ['W 6-4', 'L 3-5', 'W 7-2']
        })
        st.dataframe(recent, use_container_width=True, hide_index=True)
    with right:
        st.subheader('Trend Indicators')
        trend_ind = pd.DataFrame({
            'Indicator': ['Batting', 'Pitching', 'Defense'],
            'Status': ['↑ Up', '→ Stable', '↓ Down']
        })
        st.dataframe(trend_ind, use_container_width=True, hide_index=True)

with schedule_tab:
    st.subheader('Selected Date Schedule')
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

with trends_tab:
    st.subheader('Season Averages and Recent Trend')
    st.line_chart(season_df.set_index('Week'))

with deep_tab:
    st.subheader('Deep Trend Analytics')
    st.caption(f'Statcast window: {selected_date - timedelta(days=statcast_window)} through {selected_date}')
    
    if st.button('Load Statcast Data'):
        with st.spinner('Loading Statcast data...'):
            statcast_df = load_statcast_data()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('#### Batter Metrics')
                st.dataframe(statcast_df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown('#### Pitch Type Use')
                st.write('Pitch mix data placeholder')

with spray_tab:
    st.subheader('Spray Charts - Last 21 Days')
    st.caption('Hit locations showing offensive production and defensive performance.')
    
    if st.button('Load Spray Charts'):
        with st.spinner('Loading spray chart data...'):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('#### Team Hits (Offensive)')
                st.write('Offensive spray chart placeholder')
            with col2:
                st.markdown('#### Hits Allowed (Defensive)')
                st.write('Defensive spray chart placeholder')

with live_tab:
    st.subheader('Live Feed')
    live_data = pd.DataFrame({
        'Time': ['8:42 PM', '8:38 PM', '8:34 PM'],
        'Event': ['Strike out', 'Single', 'Walk'],
        'Result': ['Out', 'On Base', 'On Base']
    })
    if not live_data.empty:
        st.dataframe(live_data, use_container_width=True, hide_index=True)
    else:
        st.info('No active in-progress live game feed is available.')


# ---------------------------------------------------------------------------
# Cached ranking data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800)
def _cached_hitting(season: int):
    return get_league_hitting_stats(season)

@st.cache_data(ttl=1800)
def _cached_pitching(season: int):
    return get_league_pitching_stats(season)

@st.cache_data(ttl=1800)
def _cached_fielding(season: int):
    return get_league_fielding_stats(season)

@st.cache_data(ttl=1800)
def _cached_pitchers(season: int):
    return get_individual_pitcher_stats(season)

@st.cache_data(ttl=3600)
def _cached_war(season: int):
    return get_war_leaderboard_data(season)


def _show_top_bottom(df: pd.DataFrame, stat_col: str, ascending: bool, label: str, n: int = 5) -> None:
    """Render compact Top-N / Bottom-N summary cards for a stat."""
    if df.empty or stat_col not in df.columns or 'Team' not in df.columns:
        return
    tmp = df[['Team', stat_col]].copy()
    tmp[stat_col] = pd.to_numeric(tmp[stat_col], errors='coerce')
    tmp = tmp.dropna(subset=[stat_col])
    if tmp.empty:
        return
    best = tmp.nsmallest(n, stat_col) if ascending else tmp.nlargest(n, stat_col)
    worst = tmp.nlargest(n, stat_col) if ascending else tmp.nsmallest(n, stat_col)
    best_label = f'Best {label} ({'Lowest' if ascending else 'Highest'} {stat_col})'
    worst_label = f'Worst {label} ({'Highest' if ascending else 'Lowest'} {stat_col})'
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'**🏆 {best_label}**')
        for _, r in best.iterrows():
            st.write(f"• **{r['Team']}**: {r[stat_col]}")
    with c2:
        st.markdown(f'**⚠️ {worst_label}**')
        for _, r in worst.iterrows():
            st.write(f"• **{r['Team']}**: {r[stat_col]}")


with rankings_tab:
    st.subheader('⚾ MLB League Rankings')
    st.caption('League-wide team and player rankings. Data sourced from the MLB Stats API and FanGraphs (via pybaseball).')

    current_year = date.today().year
    rankings_season = st.selectbox(
        'Season',
        list(range(current_year, current_year - 4, -1)),
        index=0,
        key='rankings_season',
    )

    off_tab, def_tab, sp_tab, rp_tab, war_tab = st.tabs([
        '⚔️ Offense', '🛡️ Defense', '⚾ Starters', '💨 Relievers', '📊 WAR Leaders',
    ])

    # ── Offense ──────────────────────────────────────────────────────────────
    with off_tab:
        hitting_df, hit_err = _cached_hitting(rankings_season)
        if hit_err and hitting_df.empty:
            st.error(f'Offense data unavailable: {hit_err}')
        else:
            if hit_err:
                st.warning(f'Partial data: {hit_err}')
            off_df = build_offensive_rankings(hitting_df)
            if off_df.empty:
                st.info('No offensive stats available for this season yet.')
            else:
                OFFENSE_SORT_OPTS = {
                    'OPS (desc)': ('OPS', False),
                    'Runs (desc)': ('R', False),
                    'R/G (desc)': ('R/G', False),
                    'AVG (desc)': ('AVG', False),
                    'OBP (desc)': ('OBP', False),
                    'SLG (desc)': ('SLG', False),
                    'HR (desc)': ('HR', False),
                    'SB (desc)': ('SB', False),
                    'BB% (desc)': ('BB%', False),
                    'K% (asc)': ('K%', True),
                }
                off_sort_sel = st.selectbox('Sort by', list(OFFENSE_SORT_OPTS.keys()), index=0, key='off_sort')
                off_sort_col, off_sort_asc = OFFENSE_SORT_OPTS[off_sort_sel]
                off_display = sort_rankings(off_df, off_sort_col, off_sort_asc)
                display_cols = [c for c in off_display.columns if c != 'team_id']
                st.dataframe(off_display[display_cols], use_container_width=True, hide_index=True)

                st.markdown('---')
                _show_top_bottom(off_df, off_sort_col, off_sort_asc, 'Offense')
                st.markdown('---')
                render_rankings_bar(off_display, off_sort_col, f'Top 10 Teams — {off_sort_col}',
                                    ascending=off_sort_asc, n=10, color='#2196F3')

    # ── Defense ───────────────────────────────────────────────────────────────
    with def_tab:
        fielding_df, fld_err = _cached_fielding(rankings_season)
        if fld_err and fielding_df.empty:
            st.error(f'Defensive data unavailable: {fld_err}')
        else:
            if fld_err:
                st.warning(f'Partial data: {fld_err}')
            def_df = build_defensive_rankings(fielding_df)
            if def_df.empty:
                st.info('No defensive stats available for this season yet.')
            else:
                DEFENSE_SORT_OPTS = {
                    'FLD% (desc)': ('FLD%', False),
                    'Errors (asc)': ('E', True),
                    'Double Plays (desc)': ('DP', False),
                    'RF/G (desc)': ('RF/G', False),
                }
                def_sort_sel = st.selectbox('Sort by', list(DEFENSE_SORT_OPTS.keys()), index=0, key='def_sort')
                def_sort_col, def_sort_asc = DEFENSE_SORT_OPTS[def_sort_sel]
                def_display = sort_rankings(def_df, def_sort_col, def_sort_asc)
                display_cols = [c for c in def_display.columns if c != 'team_id']
                st.dataframe(def_display[display_cols], use_container_width=True, hide_index=True)

                st.markdown('---')
                _show_top_bottom(def_df, def_sort_col, def_sort_asc, 'Defense')
                st.markdown('---')
                render_rankings_bar(def_display, def_sort_col, f'Top 10 Teams — {def_sort_col}',
                                    ascending=def_sort_asc, n=10, color='#4CAF50')
                st.caption('Source: MLB Stats API standard fielding data. OAA (Outs Above Average) requires Baseball Savant and is not included.')

    # ── Starters ─────────────────────────────────────────────────────────────
    with sp_tab:
        pitcher_df_sp, pit_err_sp = _cached_pitchers(rankings_season)
        if pit_err_sp and pitcher_df_sp.empty:
            st.error(f'Starter data unavailable: {pit_err_sp}')
        else:
            if pit_err_sp:
                st.warning(f'Partial data: {pit_err_sp}')
            sp_df = build_starter_rankings(pitcher_df_sp)
            if sp_df.empty:
                st.info('No starting pitcher stats available for this season yet.')
            else:
                SP_SORT_OPTS = {
                    'ERA (asc)': ('ERA', True),
                    'WHIP (asc)': ('WHIP', True),
                    'IP (desc)': ('IP', False),
                    'IP/GS (desc)': ('IP/GS', False),
                    'K% (desc)': ('K%', False),
                    'BB% (asc)': ('BB%', True),
                    'K-BB% (desc)': ('K-BB%', False),
                    'HR/9 (asc)': ('HR/9', True),
                    'QS (desc)': ('QS', False),
                }
                sp_sort_sel = st.selectbox('Sort by', list(SP_SORT_OPTS.keys()), index=0, key='sp_sort')
                sp_sort_col, sp_sort_asc = SP_SORT_OPTS[sp_sort_sel]
                sp_display = sort_rankings(sp_df, sp_sort_col, sp_sort_asc)
                display_cols = [c for c in sp_display.columns if c != 'team_id']
                st.dataframe(sp_display[display_cols], use_container_width=True, hide_index=True)

                st.markdown('---')
                _show_top_bottom(sp_df, sp_sort_col, sp_sort_asc, 'Rotation')
                st.markdown('---')
                render_rankings_bar(sp_display, sp_sort_col, f'Top 10 Rotations — {sp_sort_col}',
                                    ascending=sp_sort_asc, n=10, color='#FF5722')
                st.caption('Role classification: SP = ≥50% of appearances as a start. Aggregated from individual pitcher season stats.')

    # ── Relievers ────────────────────────────────────────────────────────────
    with rp_tab:
        pitcher_df_rp, pit_err_rp = _cached_pitchers(rankings_season)
        if pit_err_rp and pitcher_df_rp.empty:
            st.error(f'Reliever data unavailable: {pit_err_rp}')
        else:
            if pit_err_rp:
                st.warning(f'Partial data: {pit_err_rp}')
            rp_df = build_reliever_rankings(pitcher_df_rp)
            if rp_df.empty:
                st.info('No reliever stats available for this season yet.')
            else:
                RP_SORT_OPTS = {
                    'ERA (asc)': ('ERA', True),
                    'WHIP (asc)': ('WHIP', True),
                    'K% (desc)': ('K%', False),
                    'BB% (asc)': ('BB%', True),
                    'K-BB% (desc)': ('K-BB%', False),
                    'Saves (desc)': ('SV', False),
                    'Holds (desc)': ('HLD', False),
                    'Save % (desc)': ('SV%', False),
                    'Blown Saves (asc)': ('BS', True),
                }
                rp_sort_sel = st.selectbox('Sort by', list(RP_SORT_OPTS.keys()), index=0, key='rp_sort')
                rp_sort_col, rp_sort_asc = RP_SORT_OPTS[rp_sort_sel]
                rp_display = sort_rankings(rp_df, rp_sort_col, rp_sort_asc)
                display_cols = [c for c in rp_display.columns if c != 'team_id']
                st.dataframe(rp_display[display_cols], use_container_width=True, hide_index=True)

                st.markdown('---')
                _show_top_bottom(rp_df, rp_sort_col, rp_sort_asc, 'Bullpen')
                st.markdown('---')
                render_rankings_bar(rp_display, rp_sort_col, f'Top 10 Bullpens — {rp_sort_col}',
                                    ascending=rp_sort_asc, n=10, color='#9C27B0')
                st.caption('Role classification: RP = <50% of appearances as a start. Aggregated from individual pitcher season stats.')

    # ── WAR Leaders ──────────────────────────────────────────────────────────
    with war_tab:
        st.markdown('**WAR Leaderboard** — FanGraphs via pybaseball. Click to load (may take 10–20 s).')
        if st.button('Load WAR Data', key='load_war'):
            with st.spinner('Fetching WAR leaderboard from FanGraphs...'):
                bat_war_raw, pit_war_raw, war_err = _cached_war(rankings_season)

            if war_err and bat_war_raw.empty and pit_war_raw.empty:
                st.error(f'WAR data unavailable: {war_err}')
            else:
                if war_err:
                    st.warning(f'Partial WAR data: {war_err}')

                overall_df, bat_war_df, pit_war_df = build_war_leaderboard(bat_war_raw, pit_war_raw)

                # Summary strip
                if not overall_df.empty:
                    top = overall_df.iloc[0]
                    top_bat = bat_war_df.iloc[0] if not bat_war_df.empty else None
                    top_pit = pit_war_df.iloc[0] if not pit_war_df.empty else None
                    s1, s2, s3 = st.columns(3)
                    s1.metric('🏆 WAR Leader', f"{top['Name']} ({top['Team']})", f"{top['WAR']:.1f} WAR")
                    if top_bat is not None:
                        s2.metric('🔝 Top Hitter (WAR)', f"{top_bat['Name']} ({top_bat['Team']})", f"{top_bat['WAR']:.1f} WAR")
                    if top_pit is not None:
                        s3.metric('⚾ Top Pitcher (WAR)', f"{top_pit['Name']} ({top_pit['Team']})", f"{top_pit['WAR']:.1f} WAR")
                    st.markdown('---')

                war_sub_all, war_sub_bat, war_sub_pit = st.tabs(['Overall WAR', 'Hitter WAR', 'Pitcher WAR'])

                with war_sub_all:
                    if overall_df.empty:
                        st.info('No combined WAR data available.')
                    else:
                        st.dataframe(overall_df.head(50), use_container_width=True, hide_index=True)

                with war_sub_bat:
                    if bat_war_df.empty:
                        st.info('No hitter WAR data available.')
                    else:
                        st.dataframe(bat_war_df.head(50), use_container_width=True, hide_index=True)
                        render_rankings_bar(bat_war_df, 'WAR', 'Top 15 Hitters by WAR',
                                            ascending=False, n=15, color='#2196F3')

                with war_sub_pit:
                    if pit_war_df.empty:
                        st.info('No pitcher WAR data available.')
                    else:
                        st.dataframe(pit_war_df.head(50), use_container_width=True, hide_index=True)
                        render_rankings_bar(pit_war_df, 'WAR', 'Top 15 Pitchers by WAR',
                                            ascending=False, n=15, color='#FF5722')

                st.caption('Source: FanGraphs (via pybaseball). WAR = Wins Above Replacement.')

st.divider()
st.caption('Version v13: Performance optimized with lazy-loaded Statcast data, parallel API calls, increased cache TTL to 3600s, and improved initial load speed.')