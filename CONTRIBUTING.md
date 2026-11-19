# Contributing

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app with `python scripts/cilia_app.py` and visit <http://127.0.0.1:8000>.

## Before Opening a Change

1. Keep the second-channel-only rule intact: model inputs must select `CHANNEL_INDEX = 1` and preserve the one-channel dimension.
2. Put reusable image/model logic in `cilia_detector/`; keep browser-specific routes and UI behavior in `scripts/cilia_app.py`.
3. Add or update a regression test in `tests/` when changing image layout handling, channel selection, checkpoint compatibility, or an API response.
4. Keep generated data, model checkpoints, virtual environments, and output folders out of commits.

## Tests

```bash
MPLCONFIGDIR=.build/matplotlib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python -m unittest discover -s tests -v
```

## Style

- Use descriptive names and standard Python formatting.
- Prefer existing helpers over duplicate TIFF, normalization, pairing, or checkpoint logic.
- Keep error messages understandable to a researcher using microscopy data.
- Preserve saved-checkpoint and input-data compatibility unless a migration is documented.
