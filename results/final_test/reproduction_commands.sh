# Run from a fresh checkout with the published dependencies installed.
# Set MIDAIR_ROOT to the original extracted MidAir root. These commands create
# a new independent reproduction; the recorded checkout's exposure ledger
# intentionally prevents executing its final test twice.
# If the separate checkout contains the recorded ledger, first archive it in
# THAT COPY ONLY as results/final_test/EXPOSURE.recorded.json. Keep this
# recorded study checkout's ledger intact. No bypass is provided by the runner.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/final_study.py --stage validation-timing \
  --dataset-root "$MIDAIR_ROOT" --output artifacts/final_study/validation_timing
python scripts/final_study.py --stage audit \
  --dataset-root "$MIDAIR_ROOT" --output artifacts/final_study/isolation.json
python scripts/final_study.py --stage test --allow-final-test \
  --dataset-root "$MIDAIR_ROOT" --timing-dir artifacts/final_study/validation_timing \
  --output artifacts/final_study/test
env -u MIDAIR_ROOT OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python scripts/verify_final_study.py --study-dir artifacts/final_study/test \
  --output artifacts/final_study/verification.json
# Verify compact recorded artifacts:
(cd results/final_test && sha256sum -c SHA256SUMS)
