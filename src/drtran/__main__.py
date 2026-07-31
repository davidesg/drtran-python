"""`python -m drtran` — the same entry point as the `drtran-py` script.

Deliberately NOT installed as `drtran`: that name belongs to the C binary on a
machine that has both, and shadowing it silently is how a battery starts
comparing a program against itself.
"""

import sys

from .cli import main

sys.exit(main())
