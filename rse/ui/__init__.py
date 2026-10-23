"""Local Gradio interface.

A client of `rse.compose`; it adds no musical logic of its own. The server
binds loopback only and never uses Gradio's public share tunnel — see
`app.check_bind`.
"""

from rse.ui.app import DEFAULT_HOST, DEFAULT_PORT, UiConfig, UiError, build_app, launch

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "UiConfig",
    "UiError",
    "build_app",
    "launch",
]
