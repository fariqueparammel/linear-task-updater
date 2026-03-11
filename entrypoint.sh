#!/bin/sh
# Fix bind-mount volume permissions (runs as root before dropping privileges)
chown -R appuser:appuser /app/state /app/logs 2>/dev/null || true
exec gosu appuser python -u main.py
