#!/bin/sh
# Train the three film models and evaluate them. Run from this folder with the environment active.
set -e
cd polyimide_models
for target in Tg_film tensile_modulus_film_clean CTE_film; do
  rm -f "$target"/*.npy
  python train_models.py --target "$target"
  python evaluate_oof10.py --target "$target"
done
