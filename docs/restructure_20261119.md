# Restructuring of 19 November 2026

The first public commit (`e42b49d`, 19 November 2026) published the working folder as it was. This
restructuring makes the code match the results, removes stale material and keeps raw and third-party
data out of the repository. No scientific result changed.

## 1. Code restored for the source-velocity revision

The tables and reports use the S velocity at each event's depth (decision D11). The Codex session made
this change between 11:33 and 11:50 KST and ran it; at 11:52 KST the files changed in that session turn
were reverted.
The tables and reports kept the new values, but the code went back to the fixed 3.5 km/s version and
four new scripts were deleted (`source_velocity.py`, `update_source_velocities.py`,
`report_source_velocity.py`, `validate_source_velocity_update.py`). The session log does not record what
reverted the files.

The 19 files were rebuilt by replaying the 26 recorded file changes of that session turn onto the
reverted files; every changed line matched its recorded context. The rebuilt `run_catalog.py`,
`spectra.py` and `source_velocity.py` give exactly the processing signature stored in all 643 per-event
results (`c141e939…`), so they are byte-identical to the code that produced them.

## 2. Configuration files written by their scripts

Several files had been edited by hand after their scripts last ran. The scripts now write them:

| File | Script | Added content |
|---|---|---|
| `inputs/study_config.json` | `prepare_study.py` | sensor choice, response removal, noise window, P-timing check, source-velocity keys; "distinct sites" in the quality rules |
| `inputs/response_sources.json` | `prepare_responses.py` | NECIS response file and checksum; the response-priority rule now applied |
| `tables/release_index.json` (was `validation.json`) | `write_manifests.py` | potency, follow-up and literature checks |

`prepare_study.py` and `prepare_responses.py` reproduce `cohort.csv`, `study_config.json`,
`master_velocity.xml` and `response_sources.json` byte for byte. `study_config.json` and
`response_sources.json` are part of the processing signatures, so this matters: rerunning the old
scripts would have marked every cached per-event result as out of date.

One fix: `sync_sangju_magnitudes.py` inserted the line "These are convention-audit examples, not the
current adopted magnitudes." into `../sangju_20261117/README.txt` on every run, and seven copies had
accumulated. It now inserts the line once and removes the repeats.

## 3. Moved out of the project

Moved to `/home/msseo/works/_quarantine_korean_mw_ml_20261119/` (list in `moves.log`), not deleted:

- Seven one-line scripts that only called another script (`build_report.py`, `make_figures.py`,
  `plot_local_magnitude.py`, `report_local_magnitude.py`, `run_sangju.py`, `validate_study.py`,
  `validate_local_study.py`) and three retired scripts that only stopped with a message
  (`check_Mw_pick_timing.py`, `sangju_ml_controls.py`, `update_ML_responses.py`).
- An early check of the Sangju moment with synthetic seismograms (`prepare_target_physical.py`,
  `calculate_target_greens.py`, `fit_target_physical.py`, `inputs/sangju_physical_requests.json`,
  `synthetics/`, `logs/physical_recovery.log`). No current table or report uses it, and it read the old
  velocity-model files that the source-velocity revision replaced.
- `validate_local_magnitude.py`: last run before the HG update. It fails with the current response code,
  and its outputs were no longer in the project.
- Notebook checkpoint copies of reports, the plain-text `README.txt` (replaced by `README.md`), and the
  root `MANIFEST.csv` and `validation.json` (now written to `tables/`).

## 4. Kept out of the repository

These remain on the server and are listed with checksums in `tables/MANIFEST.csv`: per-event results,
run logs, raw waveforms, instrument responses, the KR_Mw CSV files (`code/fetch_KR_Mw.py` downloads
them at the pinned commit) and the paper PDFs. See the README, section "Not in the repository".

## 5. Verification

The whole pipeline after the waveform download was rerun in a scratch copy with the final code:

- `prepare_study.py` and `prepare_responses.py` in a separate copy;
- `run_catalog.py` and `local_magnitude.py`, which found all cached per-event results current;
- the processing checks, the HH–HG appendix chain, the analysis and report chain, the potency chain and
  `write_manifests.py`.

Every step ran without error. Of the 185 tables, figures and reports compared with the published
versions:

- Report texts and figure pixels are identical.
- Tables are identical or differ only in row or column order, or in floating-point rounding (largest
  relative difference 6e-13). The case numbering of the depth regressions differs too; matched by
  content, the 84 fits, their predictions and bootstrap samples agree.
- Recorded checksums differ only where they cover regenerated PDFs or scratch paths.
- The release checks list the potency report among the checked PDFs, because it now exists when they
  run.
- The HH–HG check records 21 pages for the main report, its current length, instead of 20.

The pipeline (README steps 2–9) was then rerun in the project itself. Every table, figure and report in
the repository is therefore written by the committed code. The run logs are in `logs/rerun_20261119/`,
which is not in git.
