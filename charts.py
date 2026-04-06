from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_plotly_config() -> dict:
    return {
        'scrollZoom': False,
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False,
    }


# ---------------------------------------------------------------------------
# Existing chart functions
# ---------------------------------------------------------------------------

def render_schedule_chart(schedule_df: pd.DataFrame) -> None:
    if schedule_df.empty:
        st.info('No schedule data available for charting.')
        return
    df = schedule_df.copy()
    df['matchup'] = df['away'] + ' at ' + df['home']
    fig = px.bar(df, x='matchup', y=['away_score', 'home_score'], barmode='group',
                 title="Today's Score Snapshot")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_recent_trend_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        st.info('No completed recent games are available yet.')
        return
    df = recent_games_df.copy()
    fig = px.line(df, x='Date', y=['Team Runs', 'Opp Runs'], markers=True,
                  title='Recent Runs Trend')
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_run_diff_chart(recent_games_df: pd.DataFrame) -> None:
    if recent_games_df.empty:
        return
    fig = px.bar(recent_games_df, x='Date', y='Run Diff', color='Result',
                 title='Recent Game Run Differential')
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_rolling_chart(rolling_df: pd.DataFrame) -> None:
    if rolling_df.empty:
        st.info('Not enough completed games for rolling trend lines yet.')
        return
    fig = px.line(rolling_df, x='Date', y=['Runs 3', 'Runs 5', 'Diff 3', 'Diff 5'],
                  markers=True, title='Rolling Team Trend Lines')
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_pitch_mix_chart(pitch_mix_df: pd.DataFrame) -> None:
    if pitch_mix_df.empty:
        st.info('Pitch-mix data is not available yet.')
        return
    fig = px.bar(pitch_mix_df, x='Pitch Type', y='Usage %', color='Success',
                 title='Pitch Type Usage')
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


def render_statcast_scatter(batter_df: pd.DataFrame) -> None:
    if batter_df.empty:
        st.info('Exit velocity data is not available yet.')
        return
    kwargs: dict = dict(
        x='Avg EV',
        y='Hard Hit %',
        color='Grade',
        hover_name='Batter',
        title='Batter Quality of Contact',
    )
    if 'BIP' in batter_df.columns:
        kwargs['size'] = 'BIP'
    fig = px.scatter(batter_df, **kwargs)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


# ---------------------------------------------------------------------------
# Spray chart (complete rewrite)
# ---------------------------------------------------------------------------

_EVENT_COLORS: dict[str, str] = {
    'home_run': '#FFD700',
    'triple': '#9B59B6',
    'double': '#3498DB',
    'single': '#27AE60',
    'field_out': '#E74C3C',
    'force_out': '#E74C3C',
    'grounded_into_double_play': '#E74C3C',
    'double_play': '#E74C3C',
    'sac_fly': '#E74C3C',
    'fielders_choice_out': '#E74C3C',
    'fielders_choice': '#E74C3C',
    'sac_bunt': '#E74C3C',
}
_DEFAULT_EVENT_COLOR = '#95A5A6'

_EVENT_DISPLAY: dict[str, str] = {
    'home_run': 'Home Run',
    'triple': 'Triple',
    'double': 'Double',
    'single': 'Single',
    'field_out': 'Out',
    'force_out': 'Out',
    'grounded_into_double_play': 'Out',
    'double_play': 'Out',
    'sac_fly': 'Out',
    'fielders_choice_out': 'Out',
    'fielders_choice': 'Out',
    'sac_bunt': 'Out',
}


def _arc_points(distances: list[float], angle_start_deg: float = 45.0,
                angle_end_deg: float = 135.0, n: int = 60
                ) -> list[tuple[list[float], list[float]]]:
    """Return (xs, ys) point lists for circular arcs at the given distances."""
    result = []
    angles = np.linspace(math.radians(angle_start_deg),
                         math.radians(angle_end_deg), n)
    for r in distances:
        xs = (r * np.cos(angles)).tolist()
        ys = (r * np.sin(angles)).tolist()
        result.append((xs, ys))
    return result


def _outfield_fence_points(n: int = 120) -> tuple[list[float], list[float]]:
    """
    Approximate outfield fence: 330ft at foul lines, 400ft at center.
    Uses r(θ) = 330 + 70*sin(2*(θ-π/4)) for θ in [45°, 135°].
    """
    angles = np.linspace(math.radians(45), math.radians(135), n)
    radii = 330 + 70 * np.sin(2 * (angles - math.radians(45)))
    xs = (radii * np.cos(angles)).tolist()
    ys = (radii * np.sin(angles)).tolist()
    # Close back to foul lines
    xs = [xs[0]] + xs + [xs[-1]]
    ys = [0.0] + ys + [0.0]
    return xs, ys


def render_spray_chart(statcast_df: pd.DataFrame,
                       title: str = 'Spray Chart – Last 30 Games') -> None:
    """
    Render a 2-D spray chart using centered field coordinates (field_x, field_y).
    Home plate is at (0, 0); positive y points toward center field.
    """
    required = {'field_x', 'field_y'}
    if statcast_df is None or statcast_df.empty or not required.issubset(statcast_df.columns):
        st.info('Spray chart data is not available yet.')
        return

    df = statcast_df.copy()
    df = df.dropna(subset=['field_x', 'field_y'])
    if df.empty:
        st.info('No hit-location data available for the spray chart.')
        return

    df['field_x'] = pd.to_numeric(df['field_x'], errors='coerce')
    df['field_y'] = pd.to_numeric(df['field_y'], errors='coerce')
    df = df.dropna(subset=['field_x', 'field_y'])
    if df.empty:
        st.info('No valid hit-location data available for the spray chart.')
        return

    # Ensure optional columns exist
    for col in ('events', 'player_name', 'launch_speed', 'hit_distance_sc', 'game_date'):
        if col not in df.columns:
            df[col] = None

    df['_color'] = df['events'].map(_EVENT_COLORS).fillna(_DEFAULT_EVENT_COLOR)
    df['_label'] = df['events'].map(_EVENT_DISPLAY).fillna('Other')
    df['_ev'] = pd.to_numeric(df['launch_speed'], errors='coerce')
    df['_dist'] = pd.to_numeric(df['hit_distance_sc'], errors='coerce')

    fig = go.Figure()

    # ── Field background shape ───────────────────────────────────────────────
    fence_xs, fence_ys = _outfield_fence_points()
    fig.add_shape(
        type='path',
        path='M 0,0 ' + ' '.join(f'L {x:.1f},{y:.1f}' for x, y in zip(fence_xs, fence_ys)) + ' Z',
        fillcolor='rgba(144, 238, 144, 0.3)',
        line=dict(color='rgba(144, 238, 144, 0.0)', width=0),
        layer='below',
    )

    # ── Foul lines ───────────────────────────────────────────────────────────
    for dx in (-320, 320):
        fig.add_trace(go.Scatter(
            x=[0, dx], y=[0, 320],
            mode='lines',
            line=dict(color='white', width=1.5),
            hoverinfo='skip',
            showlegend=False,
        ))

    # ── Outfield fence arc ───────────────────────────────────────────────────
    ox, oy = _outfield_fence_points()
    fig.add_trace(go.Scatter(
        x=ox, y=oy,
        mode='lines',
        line=dict(color='white', width=2),
        hoverinfo='skip',
        showlegend=False,
        name='Fence',
    ))

    # ── Distance markers (partial arcs, 45°–135°) ───────────────────────────
    for dist, (axs, ays) in zip(
        [200, 250, 300, 350, 400, 450],
        _arc_points([200, 250, 300, 350, 400, 450]),
    ):
        fig.add_trace(go.Scatter(
            x=axs, y=ays,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'),
            hoverinfo='skip',
            showlegend=False,
        ))
        # Label at top of each arc (angle 90°)
        fig.add_annotation(
            x=0, y=dist,
            text=f"{dist}'",
            showarrow=False,
            font=dict(color='rgba(255,255,255,0.6)', size=9),
            bgcolor='rgba(0,0,0,0)',
        )

    # ── Baselines (infield diamond) ──────────────────────────────────────────
    # 90-ft bases → ~63.6ft at 45° in centered coords
    base = 63.64
    fig.add_trace(go.Scatter(
        x=[0, base, 0, -base, 0],
        y=[0, base, 2 * base, base, 0],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', width=1),
        hoverinfo='skip',
        showlegend=False,
    ))

    # ── Hit dots grouped by event type ──────────────────────────────────────
    legend_order = ['Home Run', 'Triple', 'Double', 'Single', 'Out', 'Other']
    label_color_map = {
        'Home Run': '#FFD700',
        'Triple': '#9B59B6',
        'Double': '#3498DB',
        'Single': '#27AE60',
        'Out': '#E74C3C',
        'Other': '#95A5A6',
    }
    for label in legend_order:
        subset = df[df['_label'] == label]
        if subset.empty:
            continue
        color = label_color_map.get(label, _DEFAULT_EVENT_COLOR)

        hover_parts = ['<b>%{customdata[0]}</b>',
                       'Result: %{customdata[1]}',
                       'EV: %{customdata[2]} mph',
                       'Dist: %{customdata[3]} ft',
                       'Date: %{customdata[4]}',
                       '<extra></extra>']

        def _fmt(col: pd.Series) -> pd.Series:
            return col.fillna('N/A').astype(str)

        custom = np.column_stack([
            _fmt(subset['player_name']),
            _fmt(subset['events']),
            _fmt(subset['_ev'].round(1)),
            _fmt(subset['_dist'].round(0)),
            _fmt(subset['game_date']),
        ])

        fig.add_trace(go.Scatter(
            x=subset['field_x'],
            y=subset['field_y'],
            mode='markers',
            marker=dict(size=8, color=color, opacity=0.85,
                        line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
            name=label,
            customdata=custom,
            hovertemplate='<br>'.join(hover_parts),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=550,
        plot_bgcolor='rgba(34, 100, 34, 0.15)',
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-420, 420],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-30, 470],
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.01,
            xanchor='center',
            x=0.5,
            font=dict(size=11),
        ),
        hovermode='closest',
    )

    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


# ---------------------------------------------------------------------------
# Win probability bar
# ---------------------------------------------------------------------------

def render_win_probability_bar(
    team_a_pct: float,
    team_b_pct: float,
    team_a_name: str,
    team_b_name: str,
) -> None:
    """Horizontal stacked bar showing win-probability split."""
    a = round(float(team_a_pct), 1)
    b = round(float(team_b_pct), 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[a],
        y=['Win %'],
        orientation='h',
        name=team_a_name,
        marker_color='#2196F3',
        text=f'{a}%',
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=13, family='Arial Black'),
    ))
    fig.add_trace(go.Bar(
        x=[b],
        y=['Win %'],
        orientation='h',
        name=team_b_name,
        marker_color='#FF9800',
        text=f'{b}%',
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=13, family='Arial Black'),
    ))
    fig.update_layout(
        barmode='stack',
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 100]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config=_get_plotly_config())


# ---------------------------------------------------------------------------
# Monte Carlo results
# ---------------------------------------------------------------------------

def render_monte_carlo_results(
    mc_result: dict,
    team_a_meta: dict,
    team_b_meta: dict,
) -> None:
    """
    Render Monte Carlo simulation results for a matchup.

    mc_result keys: team_a_win_pct, team_b_win_pct, avg_score_a, avg_score_b,
        median_score_a, median_score_b, expected_run_diff,
        p10_a, p90_a, p10_b, p90_b, n_sims, lambda_a, lambda_b,
        model_inputs_used

    team_a_meta / team_b_meta keys: full_name, logo_url, abbreviation
    """
    if not mc_result:
        st.info('Monte Carlo simulation results are not available.')
        return

    required_keys = {'team_a_win_pct', 'team_b_win_pct', 'avg_score_a', 'avg_score_b'}
    if not required_keys.issubset(mc_result):
        st.info('Monte Carlo simulation results are incomplete.')
        return

    a_pct = float(mc_result.get('team_a_win_pct', 50.0))
    b_pct = float(mc_result.get('team_b_win_pct', 50.0))
    n_sims = int(mc_result.get('n_sims', 0))
    exp_diff = mc_result.get('expected_run_diff', 0.0)

    a_name = team_a_meta.get('full_name', 'Team A')
    b_name = team_b_meta.get('full_name', 'Team B')
    a_logo = team_a_meta.get('logo_url', '')
    b_logo = team_b_meta.get('logo_url', '')

    # 1. Win probability bar
    render_win_probability_bar(a_pct, b_pct, a_name, b_name)

    # 2. Compact metrics row
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if a_logo:
            st.image(a_logo, width=60)
        st.markdown(f'**{a_name}**')
        st.metric(label='Win Probability', value=f'{a_pct:.1f}%')
    with col2:
        st.markdown(
            '<div style="text-align:center;padding-top:8px">'
            '<span style="font-size:1.4rem;font-weight:bold">VS</span></div>',
            unsafe_allow_html=True,
        )
        if n_sims:
            st.caption(f'{n_sims:,} simulations')
        if exp_diff is not None:
            diff_sign = '+' if float(exp_diff) >= 0 else ''
            st.caption(f'Exp. run diff: {diff_sign}{float(exp_diff):.2f}')
    with col3:
        if b_logo:
            st.image(b_logo, width=60)
        st.markdown(f'**{b_name}**')
        st.metric(label='Win Probability', value=f'{b_pct:.1f}%')

    st.divider()

    # 3. Score summary table
    def _fmt(key: str, decimals: int = 2) -> str:
        v = mc_result.get(key)
        if v is None:
            return 'N/A'
        try:
            return f'{float(v):.{decimals}f}'
        except (TypeError, ValueError):
            return 'N/A'

    summary_data = {
        'Metric': ['Avg Score', 'Median Score', '10th Percentile', '90th Percentile'],
        a_name: [
            _fmt('avg_score_a'),
            _fmt('median_score_a'),
            _fmt('p10_a'),
            _fmt('p90_a'),
        ],
        b_name: [
            _fmt('avg_score_b'),
            _fmt('median_score_b'),
            _fmt('p10_b'),
            _fmt('p90_b'),
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # 4. Caption
    st.caption(
        'Monte Carlo simulation based on recent performance. Not a betting model.'
    )


# ---------------------------------------------------------------------------
# Division standings table
# ---------------------------------------------------------------------------

def render_division_standings_table(
    standings_df: pd.DataFrame,
    selected_team_id: int = 0,
) -> None:
    """
    Display division standings as an HTML table with optional logo inline images.
    standings_df cols: team_id, team_name, wins, losses, gb, win_pct,
                       division_rank, logo_url (optional)
    """
    if standings_df is None or standings_df.empty:
        st.info('Division standings are not available.')
        return

    df = standings_df.copy()

    # Ensure numeric types
    for col in ('wins', 'losses', 'gb', 'win_pct', 'division_rank', 'team_id'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'division_rank' in df.columns:
        df = df.sort_values('division_rank').reset_index(drop=True)

    has_logo = 'logo_url' in df.columns
    has_gb = 'gb' in df.columns
    has_win_pct = 'win_pct' in df.columns

    rows_html = []
    for _, row in df.iterrows():
        tid = int(row['team_id']) if pd.notna(row.get('team_id', None)) else -1
        is_selected = (tid == selected_team_id and selected_team_id != 0)
        style = 'font-weight:bold;background-color:rgba(33,150,243,0.15);' if is_selected else ''

        rank = int(row['division_rank']) if 'division_rank' in row and pd.notna(row['division_rank']) else '-'
        name = str(row.get('team_name', ''))
        wins = int(row['wins']) if 'wins' in row and pd.notna(row['wins']) else '-'
        losses = int(row['losses']) if 'losses' in row and pd.notna(row['losses']) else '-'
        gb_val = row.get('gb', None)
        gb = (f'{float(gb_val):.1f}' if pd.notna(gb_val) and str(gb_val) not in ('0', '0.0', '-') else '-') if has_gb else '-'
        wp = f'{float(row["win_pct"]):.3f}' if has_win_pct and pd.notna(row.get('win_pct')) else '-'

        if has_logo and pd.notna(row.get('logo_url')):
            logo_html = f'<img src="{row["logo_url"]}" width="22" style="vertical-align:middle;margin-right:4px">'
        else:
            logo_html = ''

        rows_html.append(
            f'<tr style="{style}">'
            f'<td style="text-align:center;padding:4px 8px">{rank}</td>'
            f'<td style="padding:4px 8px">{logo_html}{name}</td>'
            f'<td style="text-align:center;padding:4px 8px">{wins}</td>'
            f'<td style="text-align:center;padding:4px 8px">{losses}</td>'
            f'<td style="text-align:center;padding:4px 8px">{gb}</td>'
            f'<td style="text-align:center;padding:4px 8px">{wp}</td>'
            f'</tr>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9rem">'
        '<thead><tr style="border-bottom:2px solid #444">'
        '<th style="padding:6px 8px">Rank</th>'
        '<th style="padding:6px 8px;text-align:left">Team</th>'
        '<th style="padding:6px 8px">W</th>'
        '<th style="padding:6px 8px">L</th>'
        '<th style="padding:6px 8px">GB</th>'
        '<th style="padding:6px 8px">Win%</th>'
        '</tr></thead>'
        '<tbody>' + ''.join(rows_html) + '</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Wild card standings table
# ---------------------------------------------------------------------------

def render_wildcard_standings_table(
    wc_df: pd.DataFrame,
    selected_team_id: int = 0,
) -> None:
    """
    Display wild card standings as an HTML table with optional logo inline images.
    wc_df cols: team_id, team_name, wins, losses, gb, win_pct, wc_rank,
                logo_url (optional)
    """
    if wc_df is None or wc_df.empty:
        st.info('Wild card standings are not available.')
        return

    df = wc_df.copy()

    for col in ('wins', 'losses', 'gb', 'win_pct', 'wc_rank', 'team_id'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'wc_rank' in df.columns:
        df = df.sort_values('wc_rank').reset_index(drop=True)

    has_logo = 'logo_url' in df.columns
    has_gb = 'gb' in df.columns
    has_win_pct = 'win_pct' in df.columns

    rows_html = []
    for _, row in df.iterrows():
        tid = int(row['team_id']) if pd.notna(row.get('team_id', None)) else -1
        is_selected = (tid == selected_team_id and selected_team_id != 0)
        style = 'font-weight:bold;background-color:rgba(33,150,243,0.15);' if is_selected else ''

        rank = int(row['wc_rank']) if 'wc_rank' in row and pd.notna(row['wc_rank']) else '-'
        name = str(row.get('team_name', ''))
        wins = int(row['wins']) if 'wins' in row and pd.notna(row['wins']) else '-'
        losses = int(row['losses']) if 'losses' in row and pd.notna(row['losses']) else '-'
        gb_val = row.get('gb', None)
        gb = (f'{float(gb_val):.1f}' if pd.notna(gb_val) and str(gb_val) not in ('0', '0.0', '-') else '-') if has_gb else '-'
        wp = f'{float(row["win_pct"]):.3f}' if has_win_pct and pd.notna(row.get('win_pct')) else '-'

        if has_logo and pd.notna(row.get('logo_url')):
            logo_html = f'<img src="{row["logo_url"]}" width="22" style="vertical-align:middle;margin-right:4px">'
        else:
            logo_html = ''

        rows_html.append(
            f'<tr style="{style}">'
            f'<td style="text-align:center;padding:4px 8px">{rank}</td>'
            f'<td style="padding:4px 8px">{logo_html}{name}</td>'
            f'<td style="text-align:center;padding:4px 8px">{wins}</td>'
            f'<td style="text-align:center;padding:4px 8px">{losses}</td>'
            f'<td style="text-align:center;padding:4px 8px">{gb}</td>'
            f'<td style="text-align:center;padding:4px 8px">{wp}</td>'
            f'</tr>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9rem">'
        '<thead><tr style="border-bottom:2px solid #444">'
        '<th style="padding:6px 8px">WC Rank</th>'
        '<th style="padding:6px 8px;text-align:left">Team</th>'
        '<th style="padding:6px 8px">W</th>'
        '<th style="padding:6px 8px">L</th>'
        '<th style="padding:6px 8px">GB</th>'
        '<th style="padding:6px 8px">Win%</th>'
        '</tr></thead>'
        '<tbody>' + ''.join(rows_html) + '</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
