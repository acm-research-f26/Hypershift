# W2: STHGCN official code (engineering log)

Date 2026-10-05. Outcome: **BLOCKED at step 2 (data not obtainable).** No port, smoke or Kaggle run was started (no data to run on; no GPU hours used).

## Repos (cloned into git-ignored `external/`)
- `midas-research/sthgcn-icdm` @ `9f6be4883e9d1abc80014fb6e739c980ff818333` (2021-06-13). Python 3.6 / PyTorch / PyG / networkx per its README.
- `dmis-lab/hats` @ `854b1605813d2f539a4443fd5a9fae5cada4ee0d` (2019-12-02). Its README says S&P 500, 2013-02-08 to 2019-06-17, 1174 trading days, Yahoo prices + Wikidata relations.

## What the code does (read, not run)
- Task: 3-class (down/neutral/up, `label_proportion 1 1 1`) next-day movement per stock, lookback 50 days, feature `return` only (`Makefile`: `--feature_list return --feat_att --rel_att --test_phase N`).
- Phases: `test_size=100`, `dev_size=50`, `train_proportion=3` (train 300-50 days, dev 50, test 100), `tot_ph = int(1652/100) = 16`; rolling test phases (`dataset.py` L49-L56). The code hard-codes 1652 days and 423 stocks (`base_train.py` L19, `HGNN.py`, `trainer.py`), which does **not** match the HATS README (1174 days, S&P 500), so the STHGCN data is not the public HATS download as described.
- Hypergraph incidence H is loaded from `hypergraph.npy` in the working directory (`base_train.py` L16). **That file is not in the repo** and no script in either repo creates it.
- Training: Adam lr 2e-4, wd 5e-4, 300 epochs, batch of one day. **No validation use and no epoch selection**: `BaseTrain.train` evaluates the *test* set every epoch.
- Evaluator (`evaluator.py`): `evaluate` uses `pred_li`/`true_li` before assignment (UnboundLocalError on the first call) and only prints a sklearn `classification_report`; `metric()` (acc, macro/micro F1, expected return) is called but its result is discarded. So the shipped code cannot run to completion without edits, and does not itself output the paper's metrics.

## Blocker (exact)
`bash download.sh` in both repos fetches Google Drive file id `1VXMkPmWNRsRCpQVkYQKfgR15l5_XVrWO` (`graph_classification/download.sh` uses `1uPKONgALMCoNDHchgdu7rk_aATNk7j-l`).
- `gdown https://drive.google.com/uc?id=1VXMkPmWNRsRCpQVkYQKfgR15l5_XVrWO` fails ("Cannot retrieve the public link ... permission").
- `curl` to `drive.usercontent.google.com/download?...&confirm=t`, `docs.google.com/uc?export=download`, and `drive.google.com/file/d/<id>/view` all return **HTTP 404** (file deleted). Second id: HTTP 404 too.
- Web search for a mirror found none. Rebuilding the data is not possible faithfully: the exact 423-stock list, `ordered_ticker.pkl`, `adj_mat.pkl`, `rel_num.pkl`, the 85-relation Wikidata extraction, and `hypergraph.npy` are unpublished.

## STHGCN paper values
IEEE page (doi 10.1109/ICDM50108.2020.00057) is paywalled; no open PDF obtained. A web-search snippet quoted NASDAQ acc 40.11 / F1 39.46 and NYSE acc 47.08 / F1 38.47, but I could not verify the source or table and it may come from another paper's comparison table. **UNKNOWN** (page/table) until the PDF is supplied. The predeclared rule needs the paper's value and std, so no yes/no verdict is possible.

## Port changes
None made (nothing was ported). If data appear later, the minimum edits are: define `pred_li`/`true_li`, keep `metric()` output, supply `hypergraph.npy`, replace `device='cuda'` hard-codes as needed, add dev-based epoch selection as an explicit declared deviation, and note `from_tensor`/PyG API differences.

## Commands
`git clone` of both repos; `pip install gdown` into `.venv`; `gdown`/`curl` probes above (CPU, no GPU use).

## Needed from the user to unblock
Any copy of `data.tar` (price/`processed/*_data.csv`, relation pkl files, `hypergraph.npy`), or the STHGCN authors' contact; and the STHGCN paper PDF.
