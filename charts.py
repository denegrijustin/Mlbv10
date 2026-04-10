from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _get_plotly_config() -> dict:
    """Return Plotly config with zoom disabled."""
    return {
        'scrollZoom': False,
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False,
    }


def render_schedule_chart(schedule_df: pd.DataFrame) -> None:
    if schedule_df.empty:
        st.info('No schedule data available for charting.')
        return
    df = schedule_df.copy()
    df['matchup'] = df['away'] + ' at ' + df['home']
    fig = px.bar(df, x='matchup', y=['away_score', 'home_score'], barmode='group', title="Today's Score Snapshot")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_recent_trend_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        st.info('No completed recent games are available yet.')
        return
    df = recent_games_df.copy()
    fig = px.line(df, x='Date', y=['Team Runs', 'Opp Runs'], markers=True, title='Recent Runs Trend')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_run_diff_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        return
    fig = px.bar(recent_games_df, x='Date', y='Run Diff', color='Result', title='Recent Game Run Differential')
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_rolling_chart(rolling_df: pd.DataFrame) -> None:
    if rolling_df.empty:
        st.info('Not enough completed games for rolling trend lines yet.')
        return
    fig = px.line(rolling_df, x='Date', y=['Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'], markers=True, title='Rolling Team Trend Lines')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_pitch_mix_chart(pitch_mix_df: pd.DataFrame) -> None:
    if pitch_mix_df.empty:
        st.info('Pitch-mix data is not available yet.')
        return
    fig = px.bar(pitch_mix_df, x='Pitch Type', y='Usage %', color='Success', title='Pitch Type Usage')
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_statcast_scatter(batter_df: pd.DataFrame) -> None:
    if batter_df.empty:
        st.info('Exit velocity data is not available yet.')
        return
    fig = px.scatter(batter_df, x='Avg EV', y='Hard Hit %', size='BIP', color='Grade', hover_name='Batter', title='Batter Quality of Contact')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_spray_chart(statcast_df: pd.DataFrame, chart_type: str = 'offensive') -> None:
    """
    Render a spray chart showing hit locations on a baseball field.
    
    Args:
        statcast_df: Statcast data with hc_x, hc_y coordinates
        chart_type: 'offensive' for team hits or 'defensive' for hits allowed
    """
    if statcast_df.empty or 'hc_x' not in statcast_df.columns or 'hc_y' not in statcast_df.columns:
        st.info(f'{chart_type.capitalize()} spray chart data is not available yet.')
        return
    
    df = statcast_df.copy()
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=['hc_x', 'hc_y'])
    if df.empty:
        st.info(f'No hit location data available for {chart_type} spray chart.')
        return
    
    # Determine hit vs out
    HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
    if 'events' in df.columns:
        df['hit_type'] = df['events'].isin(HIT_EVENTS)
        df['result'] = df['hit_type'].map({True: 'Hit', False: 'Out'})
    else:
        df['result'] = 'Unknown'
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add hits (green circles)
    hits = df[df['result'] == 'Hit']
    if not hits.empty:
        fig.add_trace(go.Scatter(
            x=hits['hc_x'],
            y=hits['hc_y'],
            mode='markers',
            marker=dict(size=8, color='green', opacity=0.7),
            name='Hits',
            hovertext=hits.get('description', ''),
            hovertemplate='<b>Hit</b><br>Location: (%{x}, %{y})<extra></extra>',
        ))
    
    # Add outs (red X's)
    outs = df[df['result'] == 'Out']
    if not outs.empty:
        fig.add_trace(go.Scatter(
            x=outs['hc_x'],
            y=outs['hc_y'],
            mode='markers',
            marker=dict(size=8, color='red', symbol='x', opacity=0.6),
            name='Outs',
            hovertext=outs.get('description', ''),
            hovertemplate='<b>Out</b><br>Location: (%{x}, %{y})<extra></extra>',
        ))
    
    # Add baseball field outline (approximate)
    # Foul lines
    fig.add_trace(go.Scatter(
        x=[0, 88, 127.3],
        y=[0, 88, 0],
        mode='lines',
        line=dict(color='white', width=2),
        name='Foul Lines',
        hoverinfo='skip',
        showlegend=False,
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, -88, -127.3],
        y=[0, 88, 0],
        mode='lines',
        line=dict(color='white', width=2),
        name='Foul Lines',
        hoverinfo='skip',
        showlegend=False,
    ))
    
    # Infield diamond
    fig.add_trace(go.Scatter(
        x=[0, 63.6, 0, -63.6, 0],
        y=[0, 63.6, 127.3, 63.6, 0],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        name='Infield',
        hoverinfo='skip',
        showlegend=False,
    ))
    
    fig.update_layout(
        title=f'Spray Chart - {chart_type.capitalize()}',
        xaxis_title='Horizontal Distance (ft)',
        yaxis_title='Vertical Distance (ft)',
        height=500,
        width=500,
        hovermode='closest',
        plot_bgcolor='rgba(34, 139, 34, 0.3)',  # Dark green field
        paper_bgcolor='white',
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_ranking_bar(
    df: pd.DataFrame,
    value_col: str,
    label_col: str = 'Team',
    title: str = 'Leaderboard',
    n: int = 10,
    ascending: bool = False,
) -> None:
    """Horizontal bar chart showing a ranking metric for *n* teams."""
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        st.info('No data available for chart.')
        return
    plot_df = df.sort_values(value_col, ascending=ascending).head(n).copy()
    plot_df = plot_df.iloc[::-1]
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation='h',
        title=title,
        text=value_col,
    )
    fig.update_layout(
        height=max(300, n * 35),
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title='',
        xaxis_title=value_col,
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())