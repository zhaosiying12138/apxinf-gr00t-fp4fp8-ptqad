#!/usr/bin/env python3
"""Build the v12 RTN W4A4 encoding-budget inventory.

The implementation is shared with the existing CPU-only inventory builder;
this v12 entry point keeps the public reproduction commands free of the
legacy experiment label.
"""
from __future__ import annotations

from build_recipe_inventory_v11 import main


if __name__ == "__main__":
    raise SystemExit(main())
