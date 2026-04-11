from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _get_plotly_config() -> dict:
    return {
        'scrollZoom': False,
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False,
    }


# ─── Field outline helpers ────────────────────────────────────────────────────

def _add_field_outline(fig: go.Figure) -> None:
    """Add a standard baseball field outline to a Plotly figure (centered at home plate)."""
    import numpy as np

    # Foul lines (left field & right field)
    foul_len = 330
    for angle_deg in (45, 135):
        rad = math.radians(angle_deg)
        fig.add_trace(go.Scatter(
            x=[0, foul_len * math.cos(rad)],
            y=[0, foul_len * math.sin(rad)],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.6)', width=2),
            hoverinfo='skip',
            showlegend=False,
        ))

    # Outfield wall arc (~330 ft left/right, ~405 ft center) using smooth sine approximation
    angles = np.linspace(math.radians(20), math.radians(160), 60)

    def _wall_r(a: float) -> float:
        frac = (a - math.radians(20)) / math.radians(140)
        return 330 + (405 - 330) * math.sin(frac * math.pi)

    ox = [_wall_r(a) * math.cos(a) for a in angles]
    oy = [_wall_r(a) * math.sin(a) for a in angles]
    fig.add_trace(go.Scatter(
        x=ox, y=oy,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', width=2),
        hoverinfo='skip',
        showlegend=False,
    ))

    # Infield diamond (90 ft bases)
    base = 63.64  # 90 / sqrt(2)
    diamond_x = [0, base, 0, -base, 0]
    diamond_y = [0, base, base * 2, base, 0]
    fig.add_trace(go.Scatter(
        x=diamond_x, y=diamond_y,
        mode='lines',
        line=dict(color='rgba(210,180,140,0.5)', width=1),
        hoverinfo='skip',
        showlegend=False,
    ))

    # Pitcher's mound
    fig.add_trace(go.Scatter(
        x=[0], y=[60.5],
        mode='markers',
        marker=dict(size=6, color='rgba(210,180,140,0.7)', symbol='circle'),
        hoverinfo='skip',
        showlegend=False,
    ))


def _field_layout(title: str) -> dict:
    return dict(
        title=title,
        height=520,
        plot_bgcolor='rgba(40,100,40,0.85)',
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            range=[-380, 380],
            color='white',
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            range=[-30, 440],
            color='white',
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest',
        legend=dict(bgcolor='rgba(0,0,0,0.4)', font=dict(color='white')),
    )


# ─── Schedule / basic charts ──────────────────────────────────────────────────

def render_schedule_chart(schedule_df: pd.DataFrame, key: str = 'schedule_chart') -> None:
    if schedule_df.empty:
        st.info('No schedule data available for charting.')
        return
    df = schedule_df.copy()
    df['matchup'] = df['away'] + ' at ' + df['home']
    fig = px.bar(df, x='matchup', y=['away_score', 'home_score'], barmode='group', title="Today's Score Snapshot")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_recent_trend_chart(recent_games_df: pd.DataFrame, key: str = 'recent_trend') -> None:
    if recent_games_df.empty:
        st.info('No completed recent games are available yet.')
        return
    df = recent_games_df.copy()
    fig = px.line(df, x='Date', y=['Team Runs', 'Opp Runs'], markers=True, title='Recent Runs Trend')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_run_diff_chart(recent_games_df: pd.DataFrame, key: str = 'run_diff') -> None:
    if recent_games_df.empty:
        return
    fig = px.bar(recent_games_df, x='Date', y='Run Diff', color='Result', title='Recent Game Run Differential')
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_rolling_chart(rolling_df: pd.DataFrame, key: str = 'rolling') -> None:
    if rolling_df.empty:
        st.info('Not enough completed games for rolling trend lines yet.')
        return
    fig = px.line(rolling_df, x='Date', y=['Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'], markers=True, title='Rolling Team Trend Lines')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_pitch_mix_chart(pitch_mix_df: pd.DataFrame, key: str = 'pitch_mix') -> None:
    if pitch_mix_df.empty:
        st.info('Pitch-mix data is not available yet.')
        return
    fig = px.bar(pitch_mix_df, x='Pitch Type', y='Usage %', color='Success', title='Pitch Type Usage')
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_statcast_scatter(batter_df: pd.DataFrame, key: str = 'statcast_scatter') -> None:
    if batter_df.empty:
        st.info('Exit velocity data is not available yet.')
        return
    fig = px.scatter(batter_df, x='Avg EV', y='Hard Hit %', color='Grade', hover_name='Batter', title='Batter Quality of Contact')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


# ─── Runs by inning ───────────────────────────────────────────────────────────

def render_runs_by_inning_bar(inning_df: pd.DataFrame, key: str = 'rbi_bar') -> None:
    """Grouped bar chart: runs scored vs allowed per inning."""
    if inning_df.empty:
        st.info('No inning run data available.')
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=inning_df['Inning'],
        y=inning_df['Runs Scored'],
        name='Runs Scored',
        marker_color='steelblue',
        hovertemplate=(
            'Inning %{x}<br>Runs Scored: %{y}'
            '<extra></extra>'
        ),
    ))
    fig.add_trace(go.Bar(
        x=inning_df['Inning'],
        y=inning_df['Runs Allowed'],
        name='Runs Allowed',
        marker_color='tomato',
        hovertemplate=(
            'Inning %{x}<br>Runs Allowed: %{y}'
            '<extra></extra>'
        ),
    ))
    # Overlay differential line
    fig.add_trace(go.Scatter(
        x=inning_df['Inning'],
        y=inning_df['Differential'],
        name='Differential',
        mode='lines+markers',
        line=dict(color='gold', width=2),
        marker=dict(size=6),
        yaxis='y2',
        hovertemplate='Inning %{x}<br>Diff: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title='Season Runs By Inning',
        barmode='group',
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(title='Inning', tickmode='linear'),
        yaxis=dict(title='Runs'),
        yaxis2=dict(
            title='Differential',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_runs_by_inning_heatmap(inning_df: pd.DataFrame, key: str = 'rbi_heatmap') -> None:
    """Heatmap of runs scored and allowed per inning."""
    if inning_df.empty:
        st.info('No inning run data available for heatmap.')
        return
    df = inning_df.copy()
    z = [df['Runs Scored'].tolist(), df['Runs Allowed'].tolist(), df['Differential'].tolist()]
    y_labels = ['Runs Scored', 'Runs Allowed', 'Differential']
    x_labels = [str(i) for i in df['Inning'].tolist()]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        hoverongaps=False,
        hovertemplate='Inning %{x}<br>%{y}: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title='Runs By Inning Heatmap',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


# ─── Team ranking summary ─────────────────────────────────────────────────────

def render_team_rank_cards(rank_df: pd.DataFrame, team_name: str) -> None:
    """Display selected-team ranking summary grouped by category."""
    if rank_df.empty:
        st.info('Team ranking data not available.')
        return
    for category in rank_df['Category'].unique():
        st.markdown(f'**{category}**')
        cat_df = rank_df[rank_df['Category'] == category].reset_index(drop=True)
        cols = st.columns(min(len(cat_df), 5))
        for i, row in cat_df.iterrows():
            c = cols[i % 5]
            rank_label = row.get('Rank', 'N/A')
            val = row.get('Value', 0)
            signal = row.get('Signal', '🟡')
            stat = row.get('Stat', '')
            # Format value nicely
            if isinstance(val, float):
                val_str = f'{val:.3f}' if val < 10 else f'{val:.1f}'
            else:
                val_str = str(val)
            c.metric(
                label=f'{signal} {stat}',
                value=val_str,
                delta=f'Rank {rank_label} of 30',
            )


# ─── WAR leaderboard ─────────────────────────────────────────────────────────

def render_war_chart(war_df: pd.DataFrame, key: str = 'war_chart') -> None:
    """Horizontal bar chart of WAR leaders."""
    if war_df.empty:
        st.info('WAR data not available.')
        return
    df = war_df.copy()
    # Ensure WAR numeric
    df['WAR'] = pd.to_numeric(df['WAR'], errors='coerce')
    df = df.dropna(subset=['WAR']).sort_values('WAR', ascending=True).tail(30)
    player_col = 'Player' if 'Player' in df.columns else df.columns[0]
    color_col = 'Type' if 'Type' in df.columns else None
    fig = px.bar(
        df,
        x='WAR',
        y=player_col,
        orientation='h',
        color=color_col,
        title='WAR Leaders',
        hover_data=[c for c in ('Team', 'G', 'PA', 'HR', 'AVG', 'ERA', 'IP') if c in df.columns],
    )
    fig.update_layout(
        height=max(400, len(df) * 18 + 100),
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title='',
        xaxis_title='WAR',
    )
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


# ─── Spray chart ──────────────────────────────────────────────────────────────

def render_player_spray_chart(spray_df: pd.DataFrame, player_name: str, key: str = 'spray_chart') -> None:
    """Render an interactive spray chart for a single player."""
    if spray_df.empty:
        st.info(f'No batted-ball data available for {player_name}.')
        return
    if 'spray_x' not in spray_df.columns or 'spray_y' not in spray_df.columns:
        st.info('Hit coordinate data not available.')
        return

    df = spray_df.copy()
    color_map = {
        'Hit': '#00cc44',
        'Out': '#ff4444',
        'home_run': '#ffaa00',
        'double': '#44aaff',
        'triple': '#cc44ff',
        'single': '#00cc44',
    }

    fig = go.Figure()
    _add_field_outline(fig)

    # Hover template
    hover_cols = [c for c in ('result', 'events', 'game_date', 'launch_speed', 'launch_angle', 'hit_distance_sc') if c in df.columns]

    for result_val in df['result'].unique():
        sub = df[df['result'] == result_val]
        color = color_map.get(result_val, '#aaaaaa')
        symbol = 'circle' if result_val == 'Hit' else 'x'
        # Build hover text
        hover_parts = [f'<b>{result_val}</b>']
        for col in hover_cols:
            if col in sub.columns and col != 'result':
                hover_parts.append(f'{col}: %{{customdata[{hover_cols.index(col)}]}}')

        custom = sub[[c for c in hover_cols if c in sub.columns]].values.tolist() if hover_cols else None
        ht = '<br>'.join([
            f'<b>{result_val}</b>',
            *[
                f'{c.replace("_", " ").title()}: %{{customdata[{i}]}}'
                for i, c in enumerate(hover_cols) if c in sub.columns and c != 'result'
            ],
        ]) + '<extra></extra>'

        fig.add_trace(go.Scatter(
            x=sub['spray_x'],
            y=sub['spray_y'],
            mode='markers',
            name=result_val,
            marker=dict(size=9, color=color, symbol=symbol, opacity=0.8,
                        line=dict(width=0.5, color='rgba(0,0,0,0.3)')),
            customdata=sub[[c for c in hover_cols if c in sub.columns]].values,
            hovertemplate=ht,
        ))

    fig.update_layout(**_field_layout(f'Spray Chart — {player_name}'))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


# ─── Home run visualization ───────────────────────────────────────────────────

def _has_3d_fields(hr_df: pd.DataFrame) -> bool:
    """Check whether we have enough data for a meaningful 3D HR chart."""
    required = {'spray_x', 'spray_y', 'launch_speed', 'launch_angle', 'apex_ft'}
    available = required & set(hr_df.columns)
    if len(available) < 4:
        return False
    # Need enough non-null values
    sub = hr_df[list(available)].dropna()
    return len(sub) >= 5


def render_team_hr_2d(hr_df: pd.DataFrame, key: str = 'hr_2d') -> None:
    """2D field spray chart for team home runs."""
    if hr_df.empty:
        st.info('No home run data available.')
        return
    if 'spray_x' not in hr_df.columns or 'spray_y' not in hr_df.columns:
        st.info('Home run coordinate data not available.')
        return

    df = hr_df.copy()

    # Build hover columns
    hover_cols = [c for c in ('player', 'game_date', 'distance', 'launch_speed', 'launch_angle', 'apex_ft') if c in df.columns]

    ht_parts = ['<b>Home Run</b>']
    for i, col in enumerate(hover_cols):
        label = col.replace('_', ' ').title()
        suffix = ' ft' if col in ('distance', 'apex_ft') else (
            ' mph' if col == 'launch_speed' else (
                '°' if col == 'launch_angle' else ''
            )
        )
        ht_parts.append(f'{label}: %{{customdata[{i}]}}{suffix}')
    hovertemplate = '<br>'.join(ht_parts) + '<extra></extra>'

    color_col = 'distance' if 'distance' in df.columns else (
        'launch_speed' if 'launch_speed' in df.columns else None
    )

    fig = go.Figure()
    _add_field_outline(fig)

    if color_col and color_col in df.columns:
        # Continuous color by distance/EV
        fig.add_trace(go.Scatter(
            x=df['spray_x'],
            y=df['spray_y'],
            mode='markers',
            name='Home Runs',
            marker=dict(
                size=12,
                color=df[color_col],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title=color_col.replace('_', ' ').title(), x=1.02),
                opacity=0.9,
                line=dict(width=1, color='white'),
                symbol='star',
            ),
            customdata=df[[c for c in hover_cols if c in df.columns]].values,
            hovertemplate=hovertemplate,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['spray_x'],
            y=df['spray_y'],
            mode='markers',
            name='Home Runs',
            marker=dict(size=12, color='gold', symbol='star', opacity=0.9),
            customdata=df[[c for c in hover_cols if c in df.columns]].values,
            hovertemplate=hovertemplate,
        ))

    layout = _field_layout('Team Home Runs — This Season')
    layout['title'] = f'Team Home Runs — This Season ({len(df)} HRs)'
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_team_hr_3d(hr_df: pd.DataFrame, key: str = 'hr_3d') -> None:
    """3D scatter of HR landing spots with apex height as Z axis."""
    if hr_df.empty:
        st.info('No home run data available for 3D view.')
        return

    df = hr_df.dropna(subset=['spray_x', 'spray_y', 'apex_ft']).copy()
    if df.empty:
        st.info('Insufficient data fields for 3D view — showing 2D chart instead.')
        render_team_hr_2d(hr_df, key=key + '_fallback')
        return

    hover_cols = [c for c in ('player', 'game_date', 'distance', 'launch_speed', 'launch_angle') if c in df.columns]
    ht_parts = ['<b>Home Run</b>', 'Side: %{x:.0f} ft', 'Depth: %{y:.0f} ft', 'Apex: %{z:.0f} ft']
    for i, col in enumerate(hover_cols):
        label = col.replace('_', ' ').title()
        ht_parts.append(f'{label}: %{{customdata[{i}]}}')
    hovertemplate = '<br>'.join(ht_parts) + '<extra></extra>'

    color_col = 'distance' if 'distance' in df.columns else 'launch_speed' if 'launch_speed' in df.columns else None

    fig = go.Figure(data=go.Scatter3d(
        x=df['spray_x'],
        y=df['spray_y'],
        z=df['apex_ft'],
        mode='markers',
        marker=dict(
            size=7,
            color=df[color_col] if color_col and color_col in df.columns else 'gold',
            colorscale='YlOrRd',
            showscale=bool(color_col),
            colorbar=dict(title=color_col.replace('_', ' ').title() if color_col else ''),
            opacity=0.85,
            line=dict(width=0.5, color='white'),
        ),
        customdata=df[[c for c in hover_cols if c in df.columns]].values,
        hovertemplate=hovertemplate,
    ))
    fig.update_layout(
        title=f'Team Home Runs 3D — Apex Height ({len(df)} HRs)',
        height=560,
        scene=dict(
            xaxis=dict(title='Side (ft)', backgroundcolor='rgba(40,100,40,0.5)', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Distance (ft)', backgroundcolor='rgba(40,100,40,0.5)', gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(title='Apex Height (ft)', backgroundcolor='rgba(30,30,60,0.5)', gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(20,20,40,0.95)',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key)


def render_hr_distribution(hr_df: pd.DataFrame, key: str = 'hr_dist') -> None:
    """Side-by-side distributions of HR distance and launch angle."""
    if hr_df.empty:
        return
    c1, c2 = st.columns(2)
    if 'distance' in hr_df.columns:
        dist_data = hr_df['distance'].dropna()
        if not dist_data.empty:
            fig = px.histogram(dist_data, x=dist_data, nbins=20, title='HR Distance Distribution', labels={'x': 'Distance (ft)'})
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            c1.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key + '_dist')
    if 'launch_angle' in hr_df.columns:
        la_data = hr_df['launch_angle'].dropna()
        if not la_data.empty:
            fig = px.histogram(la_data, x=la_data, nbins=15, title='HR Launch Angle Distribution', labels={'x': 'Launch Angle (°)'})
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            c2.plotly_chart(fig, use_container_width=True, config=_get_plotly_config(), key=key + '_la')


# ─── Rankings tables ──────────────────────────────────────────────────────────

def render_mlb_rankings_table(
    rank_df: pd.DataFrame,
    display_cols: list[str],
    title: str,
    key: str,
    sort_col: str | None = None,
) -> None:
    """Render a sortable MLB-wide rankings table."""
    if rank_df.empty:
        st.info(f'{title} data not available.')
        return
    cols = [c for c in display_cols if c in rank_df.columns]
    if not cols:
        st.info(f'No columns to display for {title}.')
        return
    df = rank_df[cols].copy()
    # Convert everything to str for safe display
    for c in df.columns:
        df[c] = df[c].apply(lambda v: str(v) if not isinstance(v, str) else v)
    if sort_col and sort_col in cols:
        df = df.sort_values(sort_col)
    st.markdown(f'##### {title}')
    st.dataframe(df, use_container_width=True, hide_index=True, key=key)


# ─── Standings ────────────────────────────────────────────────────────────────

def render_standings_table(standings_df: pd.DataFrame, division: str, key: str) -> None:
    """Render a division standings table."""
    if standings_df.empty:
        st.info(f'{division} standings not available.')
        return
    div_df = standings_df[standings_df['division'] == division].copy()
    if div_df.empty:
        st.info(f'No data for {division}.')
        return
    display = ['team_name', 'wins', 'losses', 'pct', 'gb', 'streak', 'runs_scored', 'runs_allowed']
    cols = [c for c in display if c in div_df.columns]
    rename = {
        'team_name': 'Team', 'wins': 'W', 'losses': 'L', 'pct': 'PCT',
        'gb': 'GB', 'streak': 'Streak', 'runs_scored': 'RS', 'runs_allowed': 'RA',
    }
    out = div_df[cols].rename(columns=rename)
    for c in out.columns:
        out[c] = out[c].apply(lambda v: str(v) if not isinstance(v, str) else v)
    st.dataframe(out, use_container_width=True, hide_index=True, key=key)
