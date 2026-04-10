import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from mlb_api import (
    get_league_team_hitting_mlb,
    get_league_team_pitching_mlb,
    get_league_team_fielding_mlb,
    get_fg_team_batting,
    get_fg_team_fielding,
    get_fg_pitching_individual,
    get_fg_batting_individual,
)
from data_helpers import (
    build_offensive_rankings,
    build_defensive_rankings,
    build_starter_rankings,
    build_reliever_rankings,
    build_war_leaderboard,
    top_n_summary,
)
from charts import render_ranking_bar

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
summary_tab, schedule_tab, trends_tab, deep_tab, spray_tab, live_tab, rankings_tab = st.tabs([
    'Summary', 'Schedule', 'Trends', 'Deep Trends', 'Spray Charts', 'Live Feed', 'MLB Rankings'
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
# MLB Rankings tab
# ---------------------------------------------------------------------------

# Cached data loaders – each source is fetched once per hour max.

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


def _render_leader_cards(df: pd.DataFrame, metric: str, label: str = 'Team',
                         top_title: str = 'Top 5', bot_title: str = 'Bottom 5',
                         ascending_best: bool = True) -> None:
    """Show compact Top-5 / Bottom-5 summary cards side-by-side."""
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


def _display_cols(df: pd.DataFrame) -> list[str]:
    """Return display columns, dropping the internal Source column."""
    return [c for c in df.columns if c != 'Source']


with rankings_tab:
    st.subheader('MLB Rankings')
    st.caption('League-wide team rankings across key offensive, defensive, and pitching categories.')

    ranking_season = st.selectbox(
        'Season', options=list(range(date.today().year, 2019, -1)),
        index=0, key='ranking_season',
    )

    off_sub, def_sub, sp_sub, rp_sub, war_sub = st.tabs([
        '⚾ Offense', '🛡️ Defense', '🎯 Starting Pitching',
        '💪 Relief Pitching', '🏆 WAR Leaders',
    ])

    # -- shared data loads (cached) --
    with st.spinner('Loading rankings data…'):
        fg_bat, fg_bat_err = _load_fg_team_batting(ranking_season)
        mlb_hit, mlb_hit_err = _load_mlb_team_hitting(ranking_season)
        mlb_field, mlb_field_err = _load_mlb_team_fielding(ranking_season)
        fg_field, fg_field_err = _load_fg_team_fielding(ranking_season)
        fg_pitch_ind, fg_pitch_ind_err = _load_fg_pitching_individual(ranking_season)
        mlb_pitch, mlb_pitch_err = _load_mlb_team_pitching(ranking_season)
        fg_bat_ind, fg_bat_ind_err = _load_fg_batting_individual(ranking_season)

    # ---- Offense subtab ---------------------------------------------------
    with off_sub:
        st.markdown('#### Team Offensive Rankings')
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
            asc_off = sort_off in ('K%',)  # lower K% is better
            sorted_off = off_df.copy()
            sorted_off = sorted_off.sort_values(sort_off, ascending=asc_off).reset_index(drop=True)
            sorted_off['Rank'] = range(1, len(sorted_off) + 1)
            st.dataframe(sorted_off[_display_cols(sorted_off)], use_container_width=True, hide_index=True)
            render_ranking_bar(off_df, sort_off, title=f'Top 10 – {sort_off}',
                               ascending=not asc_off)

    # ---- Defense subtab ---------------------------------------------------
    with def_sub:
        st.markdown('#### Team Defensive Rankings')
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
            asc_def = sort_def in ('E',)  # lower errors is better
            sorted_def = def_df.copy()
            sorted_def = sorted_def.sort_values(sort_def, ascending=asc_def).reset_index(drop=True)
            sorted_def['Rank'] = range(1, len(sorted_def) + 1)
            st.dataframe(sorted_def[_display_cols(sorted_def)], use_container_width=True, hide_index=True)
            render_ranking_bar(def_df, sort_def, title=f'Top 10 – {sort_def}',
                               ascending=not asc_def)

    # ---- Starting Pitching subtab -----------------------------------------
    with sp_sub:
        st.markdown('#### Team Starting Pitching Rankings')
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
            asc_sp = sort_sp in ('ERA', 'WHIP', 'BB%', 'HR/9', 'FIP', 'Opp AVG')
            sorted_sp = sp_df.copy()
            sorted_sp = sorted_sp.sort_values(sort_sp, ascending=asc_sp).reset_index(drop=True)
            sorted_sp['Rank'] = range(1, len(sorted_sp) + 1)
            st.dataframe(sorted_sp[_display_cols(sorted_sp)], use_container_width=True, hide_index=True)
            render_ranking_bar(sp_df, sort_sp, title=f'Top 10 – {sort_sp}',
                               ascending=not asc_sp)

    # ---- Relief Pitching subtab -------------------------------------------
    with rp_sub:
        st.markdown('#### Team Relief Pitching Rankings')
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
            asc_rp = sort_rp in ('ERA', 'WHIP', 'BB%', 'HR/9', 'FIP', 'BS', 'Opp AVG')
            sorted_rp = rp_df.copy()
            sorted_rp = sorted_rp.sort_values(sort_rp, ascending=asc_rp).reset_index(drop=True)
            sorted_rp['Rank'] = range(1, len(sorted_rp) + 1)
            st.dataframe(sorted_rp[_display_cols(sorted_rp)], use_container_width=True, hide_index=True)
            render_ranking_bar(rp_df, sort_rp, title=f'Top 10 – {sort_rp}',
                               ascending=not asc_rp)

    # ---- WAR Leaders subtab -----------------------------------------------
    with war_sub:
        st.markdown('#### WAR Leaders')
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

            # Summary strip
            top_all = war_df.head(1)
            if not top_all.empty:
                strip_cols = st.columns(3)
                row0 = top_all.iloc[0]
                strip_cols[0].metric('Current WAR Leader',
                                     f"{row0.get('Player', '-')} ({row0.get('Team', '-')})",
                                     f"{row0.get('WAR', 0):.1f} WAR")

                # Top hitter by WAR
                hitter_war = build_war_leaderboard(fg_bat_ind, fg_pitch_ind, 'hitters')
                if not hitter_war.empty:
                    h0 = hitter_war.iloc[0]
                    strip_cols[1].metric('Top Hitter WAR',
                                         f"{h0.get('Player', '-')} ({h0.get('Team', '-')})",
                                         f"{h0.get('WAR', 0):.1f} WAR")

                # Top pitcher by WAR
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
                               ascending=False)

st.divider()
st.caption('Version v14: Added MLB Rankings tab with league-wide Offense, Defense, Starting Pitching, Relief Pitching, and WAR Leaders boards.')