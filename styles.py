"""Centralised theme CSS for the Streamlit UI.

All custom styling lives here rather than being scattered through ui.py. The
app injects ``theme_css(theme)`` once per run; the light/dark toggle in the
sidebar chooses which palette is used. Mobile rules (Milestone 3 / Issue #12)
are shared by both themes.
"""

from __future__ import annotations

THEMES = {
    "dark": {
        "bg": "#0f172a",          # app background (slate 900)
        "sidebar": "#111827",     # sidebar background
        "bubble": "#1e293b",      # chat bubble / input background (slate 800)
        "text": "#e2e8f0",        # primary text (slate 200)
        "muted": "#94a3b8",       # captions (slate 400)
        "border": "#334155",      # bubble / divider border (slate 700)
        "primary": "#3b82f6",     # accent (blue 500)
    },
    "light": {
        "bg": "#ffffff",
        "sidebar": "#f8fafc",     # slate 50
        "bubble": "#f1f5f9",      # slate 100
        "text": "#0f172a",        # slate 900
        "muted": "#475569",       # slate 600
        "border": "#e2e8f0",      # slate 200
        "primary": "#2563eb",     # blue 600
    },
}

DEFAULT_THEME = "dark"


def theme_css(theme: str = DEFAULT_THEME) -> str:
    """Return a <style> block that applies the given palette and mobile rules."""
    t = THEMES.get(theme, THEMES[DEFAULT_THEME])
    return f"""
<style>
  .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stSidebar"] {{ background-color: {t['sidebar']}; }}
  [data-testid="stSidebar"] * {{ color: {t['text']}; }}

  /* Chat bubbles */
  [data-testid="stChatMessage"] {{
    background-color: {t['bubble']};
    border: 1px solid {t['border']};
    border-radius: 12px;
    padding: 0.4rem 0.9rem;
    margin-bottom: 0.5rem;
  }}

  /* Chat input */
  [data-testid="stChatInput"] {{
    background-color: {t['bubble']};
    border: 1px solid {t['border']};
    border-radius: 12px;
  }}

  /* Captions / secondary text */
  [data-testid="stCaptionContainer"], .stCaption {{ color: {t['muted']} !important; }}

  /* Accent for headings */
  h1, h2, h3 {{ color: {t['text']}; }}

  /* Mobile responsiveness (Issue #12) */
  @media (max-width: 640px) {{
    .block-container {{ padding: 1rem 0.75rem 4rem 0.75rem !important; }}
    h1 {{ font-size: 1.5rem !important; }}
    [data-testid="stChatMessage"] {{ padding: 0.35rem 0.7rem; }}
  }}
</style>
"""


def rtl_css(enabled: bool) -> str:
    """Return a <style> block that flips chat text to right-to-left (Issue #18).

    Applied only when the active language uses an RTL script; empty otherwise so
    left-to-right languages are unaffected.
    """
    if not enabled:
        return ""
    return """
<style>
  [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
  [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
</style>
"""
