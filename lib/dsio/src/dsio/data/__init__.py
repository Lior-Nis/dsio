"""Canonical stores, lazy windowed views, and the data root.

Store once, index many: a corpus is written once as continuous signal, and every window
configuration is an index of offsets over it. See docs/adr/0005 for why the payload is
flat binary rather than a chunked format.
"""
