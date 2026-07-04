#!/bin/bash
# Start the Volatility Scanner
cd "$(dirname "$0")"
source venv/bin/activate
python3 app.py
