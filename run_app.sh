#!/bin/sh
# Predict the virtual polymers (needs run_train.sh first), build the app and serve it.
set -e
python polyimide_virtual_app/predict_virtual.py
python polyimide_virtual_app/build_app.py
python -m http.server 8766 --bind 127.0.0.1 --directory polyimide_virtual_app/dist
