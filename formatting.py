from __future__ import annotations

from typing import Any


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        text = str(value).strip().replace(',', '')
        if text == '' or text.lower() in {'nan', 'none'}:
            return default
        return int(float(text))
    except Exception:
        return default


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip().replace(',', '')
        if text == '' or text.lower() in {'nan', 'none'}:
            return default
        return float(text)
    except Exception:
        return default


def clean_text(value: Any, default: str = '-') -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def safe_pct(numerator: float, denominator: float, digits: int = 1) -> float:
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100, digits)


def stoplight(delta: float, neutral_band: float = 0.05) -> str:
    if delta > neutral_band:
        return f'🟢 +{delta:.2f}'
    if delta < -neutral_band:
        return f'🔴 {delta:.2f}'
    return '🟡 0.00'


def signed(value: float, digits: int = 2) -> str:
    number = coerce_float(value, 0.0)
    if number > 0:
        return f'+{number:.{digits}f}'
    return f'{number:.{digits}f}'


def format_record(wins: int, losses: int) -> str:
    return f'{wins}-{losses}'


def trend_arrow(delta: float, neutral_band: float = 2.0) -> str:
    """Return a colored trend arrow based on delta value.

    Green up for positive, yellow flat for near neutral, red down for negative.
    """
    if delta > neutral_band:
        return '🟢 ▲'
    if delta < -neutral_band:
        return '🔴 ▼'
    return '🟡 ▸'


def stoplight_pct(win_pct: float) -> str:
    """Stoplight indicator for opponent last-10 win percentage.

    green if win pct >= 0.600
    yellow if win pct >= 0.400 and < 0.600
    red if win pct < 0.400
    """
    if win_pct >= 0.600:
        return '🟢'
    if win_pct >= 0.400:
        return '🟡'
    return '🔴'
