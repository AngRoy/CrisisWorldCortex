"""pytest root config.

Puts repo root on sys.path so bare-name imports (`import cortex`,
`import server.simulator`, etc.) resolve when running from any cwd.
Pytest usually does this via rootdir detection, but doing it explicitly
matches the bare-name import convention documented in the CLAUDE.md files.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
