from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
    st.plotly_chart(fig, config=_get_plotly_config())


def render_recent_trend_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        st.info('No completed recent games are available yet.')
        return
    df = recent_games_df.copy()
    fig = px.line(df, x='Date', y=['Team Runs', 'Opp Runs'], markers=True, title='Recent Runs Trend')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, config=_get_plotly_config())


def render_run_diff_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        return
    fig = px.bar(recent_games_df, x='Date', y='Run Diff', color='Result', title='Recent Game Run Differential')
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, config=_get_plotly_config())


def render_rolling_chart(rolling_df: pd.DataFrame) -> None:
    if rolling_df.empty:
        st.info('Not enough completed games for rolling trend lines yet.')
        return
    fig = px.line(rolling_df, x='Date', y=['Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'], markers=True, title='Rolling Team Trend Lines')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, config=_get_plotly_config())


def render_pitch_mix_chart(pitch_mix_df: pd.DataFrame) -> None:
    if pitch_mix_df.empty:
        st.info('Pitch-mix data is not available yet.')
        return
    fig = px.bar(pitch_mix_df, x='Pitch Type', y='Usage %', color='Success', title='Pitch Type Usage')
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, config=_get_plotly_config())


def render_statcast_scatter(batter_df: pd.DataFrame) -> None:
    if batter_df.empty:
        st.info('Exit velocity data is not available yet.')
        return
    required = {'Avg EV', 'Hard Hit %', 'Batter'}
    missing = required - set(batter_df.columns)
    if missing:
        st.warning(f'Scatter chart missing columns: {", ".join(sorted(missing))}')
        return
    size_col = 'PA' if 'PA' in batter_df.columns else None
    fig = px.scatter(batter_df, x='Avg EV', y='Hard Hit %', size=size_col, color='Grade' if 'Grade' in batter_df.columns else None, hover_name='Batter', title='Batter Quality of Contact')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, key='statcast_scatter', config=_get_plotly_config())


def render_spray_chart(statcast_df: pd.DataFrame, chart_type: str = 'offensive') -> None:
    """Render a 2D spray chart on a baseball field diagram.

    Plots batted ball landing locations using Statcast hc_x/hc_y coordinates
    on a field with home plate, foul lines, infield arc, bases, and outfield
    boundary.  Points are coloured by hit result (single/double/triple/HR/out).

    Args:
        statcast_df: Statcast data with hc_x, hc_y coordinates.
        chart_type: 'offensive' for team hits or 'defensive' for hits allowed.
    """
    if statcast_df.empty or 'hc_x' not in statcast_df.columns or 'hc_y' not in statcast_df.columns:
        st.info(f'No spray chart data available for {chart_type} view.')
        return

    df = statcast_df.dropna(subset=['hc_x', 'hc_y']).copy()
    if df.empty:
        st.info(f'No hit location coordinates available for {chart_type} spray chart.')
        return

    # -- Convert Statcast coordinates to field coordinates --
    # Statcast hc_x/hc_y use a 250x250 pixel coordinate system where home
    # plate sits at approximately (125.42, 198.27).  These values are the
    # standard Statcast spray-chart origin used across public analyses (see
    # e.g. baseballsavant.mlb.com spray chart overlay).
    # We re-centre so home plate = (0, 0) and +y = toward centre field.
    HP_X, HP_Y = 125.42, 198.27
    # The 250-px image covers roughly 250 ft of field width.  The resulting
    # scale of 2.5 ft/px maps batted-ball dots to real-world distances that
    # approximate MLB field dimensions (330 ft down the lines, ~400 ft to CF).
    SCALE = 2.5

    df['field_x'] = (df['hc_x'] - HP_X) * SCALE
    df['field_y'] = (HP_Y - df['hc_y']) * SCALE  # flip y so up = outfield

    # Classify hit results
    RESULT_MAP = {
        'single': 'Single', 'double': 'Double', 'triple': 'Triple',
        'home_run': 'Home Run',
    }
    COLOR_MAP = {
        'Single': '#2ecc71', 'Double': '#3498db', 'Triple': '#9b59b6',
        'Home Run': '#e74c3c', 'Out': '#95a5a6',
    }
    if 'events' in df.columns:
        df['result'] = df['events'].map(RESULT_MAP).fillna('Out')
    else:
        df['result'] = 'Out'

    fig = go.Figure()

    # -- Draw field background shapes --
    # Outfield boundary arc (approx 330 ft down lines, 400 ft to center)
    theta_arc = np.linspace(-np.pi / 4, np.pi / 4 + np.pi / 2, 120)
    # Elliptical arc: wider to CF (~400ft), shorter down lines (~330ft)
    r_outfield = np.array([330 + 70 * np.cos(2 * (t - np.pi / 4)) for t in theta_arc])
    arc_x = r_outfield * np.sin(theta_arc)
    arc_y = r_outfield * np.cos(theta_arc)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[0], arc_x, [0]]).tolist(),
        y=np.concatenate([[0], arc_y, [0]]).tolist(),
        fill='toself', fillcolor='rgba(34, 120, 34, 0.25)',
        line=dict(color='rgba(34, 120, 34, 0.5)', width=2),
        name='Outfield', hoverinfo='skip', showlegend=False,
    ))

    # Infield dirt arc (approximate 95 ft radius)
    theta_inf = np.linspace(-np.pi / 4, np.pi / 4 + np.pi / 2, 60)
    inf_x = (95 * np.sin(theta_inf)).tolist()
    inf_y = (95 * np.cos(theta_inf)).tolist()
    fig.add_trace(go.Scatter(
        x=[0] + inf_x + [0], y=[0] + inf_y + [0],
        fill='toself', fillcolor='rgba(180, 140, 90, 0.3)',
        line=dict(color='rgba(140, 100, 60, 0.5)', width=1),
        name='Infield', hoverinfo='skip', showlegend=False,
    ))

    # Foul lines
    fig.add_trace(go.Scatter(
        x=[0, -330 * np.sin(np.pi / 4)], y=[0, 330 * np.cos(np.pi / 4)],
        mode='lines', line=dict(color='white', width=2),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[0, 330 * np.sin(np.pi / 4)], y=[0, 330 * np.cos(np.pi / 4)],
        mode='lines', line=dict(color='white', width=2),
        hoverinfo='skip', showlegend=False,
    ))

    # Bases diamond (90 ft basepaths)
    BASE_DIST = 90 / np.sqrt(2)  # ~63.6 ft diagonal distance
    bases_x = [0, BASE_DIST, 0, -BASE_DIST, 0]
    bases_y = [0, BASE_DIST, 2 * BASE_DIST, BASE_DIST, 0]
    fig.add_trace(go.Scatter(
        x=bases_x, y=bases_y, mode='lines+markers',
        line=dict(color='white', width=1.5),
        marker=dict(size=8, color='white', symbol='diamond'),
        hoverinfo='skip', showlegend=False,
    ))

    # Home plate marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=12, color='white', symbol='pentagon'),
        hoverinfo='skip', showlegend=False,
    ))

    # -- Plot batted balls by result type --
    for result_type in ['Out', 'Single', 'Double', 'Triple', 'Home Run']:
        subset = df[df['result'] == result_type]
        if subset.empty:
            continue
        marker_kwargs = dict(
            size=7, color=COLOR_MAP.get(result_type, '#999'),
            opacity=0.75, line=dict(width=0.5, color='white'),
        )
        if result_type == 'Out':
            marker_kwargs['symbol'] = 'x'
            marker_kwargs['opacity'] = 0.45
            marker_kwargs['size'] = 5
        fig.add_trace(go.Scatter(
            x=subset['field_x'].tolist(), y=subset['field_y'].tolist(),
            mode='markers', marker=marker_kwargs, name=result_type,
            hovertemplate=f'<b>{result_type}</b><br>(%{{x:.0f}}, %{{y:.0f}}) ft<extra></extra>',
        ))

    fig.update_layout(
        title=f'Spray Chart – {chart_type.capitalize()}',
        height=520, width=520,
        hovermode='closest',
        plot_bgcolor='rgba(34, 90, 34, 0.55)',
        paper_bgcolor='white',
        xaxis=dict(
            scaleanchor='y', scaleratio=1, showgrid=False,
            zeroline=False, showticklabels=False, range=[-350, 350],
        ),
        yaxis=dict(
            scaleanchor='x', scaleratio=1, showgrid=False,
            zeroline=False, showticklabels=False, range=[-30, 430],
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )

    st.plotly_chart(fig, config=_get_plotly_config())


def render_runs_per_inning_chart(rpi_df: pd.DataFrame) -> None:
    """Render a grouped bar chart of runs for / against per inning with stoplight."""
    if rpi_df.empty:
        st.info('No runs-per-inning data available.')
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rpi_df['Inning'], y=rpi_df['Runs For'],
        name='Runs For', marker_color='#2ecc71',
    ))
    fig.add_trace(go.Bar(
        x=rpi_df['Inning'], y=rpi_df['Runs Against'],
        name='Runs Against', marker_color='#e74c3c',
    ))
    for _, row in rpi_df.iterrows():
        y_top = max(int(row['Runs For']), int(row['Runs Against']), 1) + 1
        fig.add_annotation(
            x=row['Inning'], y=y_top,
            text=row['Heat'], showarrow=False,
            font=dict(size=16),
        )
    fig.update_layout(
        barmode='group',
        title='Runs Per Inning Tracker',
        xaxis_title='Inning',
        yaxis_title='Total Runs',
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, config=_get_plotly_config())