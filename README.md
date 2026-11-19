# Moment and local magnitudes of South Korean earthquakes

Moment magnitude (Mw) from S-wave spectra and an independently measured local magnitude (ML) for the
642 earthquakes of KMA ML ≥ 2 from January 2010 to October 2026 whose depths are well determined, and
for the ML 3.6 Sangju earthquake of 17 November 2026, which is kept out of all fits. The two magnitudes
are compared with each other, with the KMA ML, with the published Mw catalog of Hong and Sheen (2026),
and with potency magnitude.

**Main report:** [`reports/Korean_Mw_ML_study.pdf`](reports/Korean_Mw_ML_study.pdf), 21 pages: methods,
results, figures and the sensor appendix. The numbers in all reports are read from the files in `tables/`.

| Report | Pages | Content |
|---|---|---|
| [`Korean_Mw_ML_study.pdf`](reports/Korean_Mw_ML_study.pdf) | 21 | Main report; ends with the HH–HG appendix |
| [`Independent_ML_and_Mw_comparison.pdf`](reports/Independent_ML_and_Mw_comparison.pdf) | 10 | The ML part of the main report alone |
| [`Velocity_models_and_magnitudes.pdf`](reports/Velocity_models_and_magnitudes.pdf) | 3 | S velocity at the source from two velocity models and its effect on Mw |
| [`Estimator_and_KR_Mw_comparison.pdf`](reports/Estimator_and_KR_Mw_comparison.pdf) | 6 | Multitaper versus direct Fourier spectra; first comparison with KR_Mw |
| [`Hong_KR_Mw_regression_and_stability.pdf`](reports/Hong_KR_Mw_regression_and_stability.pdf) | 9 | Method differences from Hong and Sheen (2026); OLS, orthogonal and Deming fits |
| [`Potency_magnitude_and_quadratic_scaling.pdf`](reports/Potency_magnitude_and_quadratic_scaling.pdf) | 11 | Potency magnitude; linear versus quadratic ML scaling; depth trend |
| [`Spectral_fit_atlas.pdf`](reports/Spectral_fit_atlas.pdf) | 161 | Spectrum and fit of every event, rejected ones included |
| [`hh_hg_comparison/reports/HH_HG_paired_comparison.pdf`](hh_hg_comparison/reports/HH_HG_paired_comparison.pdf) | 8 | Co-located broadband (HH) and accelerometer (HG) records at Haenam and Sangju |

## What was done

1. **Events.** KMA events of ML ≥ 2, 1 January 2010 to 31 October 2026, whose depths have grade G2 or G3
   in the national depth assessment of the Sangju study: 642 events. Depth is used as a continuous
   value; the national D90 is 16.89 km.
2. **Records.** One three-component record per site within 100 km: HH if usable, otherwise EL, otherwise
   HG. The instrument response of the right date is removed to displacement, with no water level and a
   0.1/0.2/25/40 Hz cosine taper in frequency. Response priority: the audited Haenam inventory, then the
   current NECIS response, then the historical station metadata of Ahn et al. (2024). P times that do not
   fit the distance are rejected.
3. **Moment magnitude.** S windows of 10.1 s and 20.1 s starting 0.1 s before S. Noise windows of the
   same length end 1 s before P. Signal-to-noise amplitude ratio ≥ 3.
   - Spectra: multitaper (Prieto 2022; NW = 4, 7 tapers), with the three components summed. Fourier
     amplitude = √(N·dt·P/2).
   - Path correction: 1/r spreading over the hypocentral distance, and Q(f) = 229.2 f^0.73 along the
     observed S travel time.
   - Fit: the station average is fitted with a generalized Boatwright spectrum over 0.5–20 Hz.
   - Moment: M0 = 4πρβ³Ω/(2·0.63), with ρ = 2800 kg/m³ and β the S velocity at the event depth from
     Kim & Kim (1983) with Vp/Vs = 1.73. Kim et al. (2011) gives the comparison.
   - Magnitude: Mw = (log10 M0 − 9.05)/1.5.
   - Quality grade A needs at least 8 sites, an observed low-frequency plateau and a usable band of at
     least one decade. The multitaper and Fourier, Boatwright and Brune, and 10.1 s and 20.1 s estimates
     must also agree within 0.10.
4. **Local magnitude.** The Sheen et al. (2018) scale as used in the Haenam and Buan projects:
   - Wood–Anderson records, filtered 0.5–10 Hz.
   - Geometric mean of the N and E peak amplitudes.
   - 10–100 km epicentral distance, no station terms.
   - Event ML is the median over at least 3 sites. Grade A needs at least 5 sites and scatter ≤ 0.35.
5. **HH and HG.** At 0.7–1.2 Hz, the HG/HH spectral ratio of co-located records is 1.00–1.14 at four
   Haenam stations and 1.01–1.09 at five Sangju stations, at most 0.04 in Mw
   (`hh_hg_comparison/tables/spectral_comparison.csv`). HG records are therefore used. The sixth Sangju
   station, KS.BURB, gives 0.79 and is left out of the Sangju magnitudes.
6. **Mw–ML relation.** Least squares with one total weight per earthquake sequence, with whole-sequence
   bootstrap intervals and leave-one-sequence-out prediction errors. Orthogonal (ODR) and Deming fits are
   checks. No step selects events or stations by Mw–ML agreement.
7. **Hong and Sheen (2026).** The KR_Mw catalog is matched by exact UTC origin time. Its magnitudes are
   used as published; the method differences are read from the paper.
8. **Potency.** Scalar potency P0 = M0/(ρβ²) and the magnitude MP of Liu et al. (2026). A quadratic
   ML–MP relation is tested against a line, and the residuals are tested for a depth trend. This is a
   separate analysis; it does not change the adopted Mw.

The choices behind these steps, who made them and where their evidence is: [`docs/decisions.csv`](docs/decisions.csv).

## Main results

- **Sangju, 17 November 2026 (held out).**
  - Mw 3.42 (M0 1.50 × 10¹⁴ N m) from 104 sites at 19.76 km depth; 3.40 with the Kim et al. (2011) velocity.
  - ML 3.69 from 103 sites (21 HH, 6 EL, 76 HG); KMA ML 3.6.
  - Source: `tables/Sangju2026_summary.json`.
- **Haenam, 3 May 2020 13:07 UTC (KMA ML 3.1).** Mw 3.17 at the national depth of 21.26 km, with either
  velocity model (`tables/Mw_ML_catalog.csv`).
- **Catalog.**
  - Mw grade A for 443 events, B 61, C 115, F 23.
  - ML measured for 624 events, 511 of grade A.
  - 357 events in 194 sequences have both of grade A (`tables/study_summary.json`,
    `tables/independent_ML_summary.json`).
- **Mw and ML.**
  - Measured ML: Mw = 0.767 + 0.727 ML. Slope interval 0.700–0.754; RMS 0.074; RMS for held-out
    sequences 0.075.
  - KMA ML on the same events: Mw = 0.972 + 0.694 ML, RMS 0.112 (held out 0.114).
  - These lines describe the selected events; they are not conversion formulas
    (`tables/independent_ML_regressions.csv`).
- **Hong and Sheen (2026).** 294 events match, 199 of them of Mw grade A.
  - On these 199, our Mw is lower than their spectral Mw by a median of 0.068, and lower than their
    time-domain Mw by 0.103. With the Kim et al. (2011) velocity the gaps are 0.058 and 0.084
    (`tables/KR_Mw_comparison_statistics.csv`).
  - Their source constants are ρ = 2700 kg/m³ and β = 3.61 km/s at a fixed 10 km depth.
  - Event selection alone lowers the slope. The orthogonal slope of their spectral Mw on KMA ML is 0.96
    for all 294 matches, but 0.76 for the 199 that pass our grade A (`tables/Hong_regression_comparison.csv`).
- **Potency.**
  - Across the 357 events, a quadratic ML–MP relation predicts held-out sequences no better than a line:
    RMS 0.0764 for both. The quadratic coefficient is 0.033, interval −0.015 to 0.070
    (`tables/potency_regressions.csv`).
  - For the six events above ML 4 (five sequences) the quadratic is closer: held-out RMS 0.111 against
    0.161. With five sequences, this difference is not significant
    (`tables/magnitude_followup_prediction_metrics.csv`).
  - After the quadratic ML trend is removed, MP decreases with depth by 0.061 per 10 km (interval
    0.037–0.087). The median residual falls from +0.020 at 5–10 km to −0.035 at 10–15 km and stays near
    −0.03 deeper. This is a trend in the selected events, not a correction
    (`tables/magnitude_followup_depth_regressions.csv`, `tables/magnitude_followup_depth_bins.csv`).

## Layout

| Path | Content |
|---|---|
| `code/` | All scripts, in one folder because they import each other. Run order below. |
| `inputs/` | Event list (`cohort.csv`), processing settings (`study_config.json`), quality rules, response sources, the two velocity models (`source_velocity_models.json`), the KR_Mw source record, a map coastline, the earlier Mw and ML from velocity sensors only (before HG was added; used for the before–after comparison), and a copy of the Haenam ML recipe that `local_magnitude.py` imports from the Haenam project. |
| `tables/` | Catalogs, fits, regressions, checks, the release index and the file manifest (groups below). |
| `figures/` | Figures 01–22 of the reports, as PNG and PDF. |
| `reports/` | The PDF reports. |
| `hh_hg_comparison/` | The paired-sensor study: its own `code/`, `tables/`, `figures/` and report; `inputs/` holds the pair list and checksums. |
| `ml_validation/` | Provenance of the ML measurement and the HG processing checks. |
| `references/` | The papers used ([`references/README.md`](references/README.md)). |
| `docs/` | Decision log and the record of the restructuring of 19 November 2026 ([`docs/restructure_20261119.md`](docs/restructure_20261119.md)). |

On the server the project also holds the per-event results that are not in the repository: `events/`
(fits), `spectra/` (spectra), `station_logs/` (station checks), `ml_events/` (ML amplitudes) and `logs/`.

Tables by prefix:

| Prefix | Content | Written by |
|---|---|---|
| `catalog_*` | Raw output of the waveform processing | `run_catalog.py` |
| `Mw_ML_catalog`, `Mw_ML_primary`, `clean_spectral_fits`, `processing_status`, `all_station_checks`, `study_summary`, `regression_*`, `events_for_waveform_review` | Mw catalog, every fit, Mw–KMA ML fits | `analyze_catalog.py` |
| `Deming_regression_sensitivity`, `leave_one_family_out` | Error-in-both-axes and held-out checks | `regression_controls.py` |
| `independent_ML_*`, `Mw_independent_ML_*`, `ML_regression_bootstrap_*` | Measured ML, its stations, Mw–ML fits and their bootstrap samples | `local_magnitude.py`, `finalize_local_magnitudes.py`, `analyze_local_magnitude.py` |
| `estimator_*`, `MT_Fourier_event_comparison`, `spectral_atlas_index` | Spectrum, window and estimator checks; atlas index | `audit_estimator_waveforms.py`, `compare_estimator_stability.py`, `compare_KR_Mw.py`, `make_atlas.py` |
| `HG_*` | Effect of adding HG records | `validate_hg_processing.py`, `report_hg_update.py` |
| `Sangju2026_*` | Sangju magnitudes and their checks | `update_sangju_hg.py` |
| `KR_Mw_*`, `Hong_*`, `comparison_*` | Comparison with Hong and Sheen (2026) | `compare_KR_Mw.py`, `hong_regressions.py`, `report_Hong_regressions.py` |
| `source_velocity_*` | The two velocity models and the conversion of 19 November 2026 | `update_source_velocities.py`, `report_source_velocity.py`, `validate_source_velocity_update.py` |
| `potency_*`, `magnitude_followup_*` | Potency analysis and its follow-up questions | `analyze_potency.py`, `potency_regional_controls.py`, `analyze_magnitude_followup.py`, `audit_potency_literature.py`, `report_potency.py` |
| `*_validation.json`, `release_index.json`, `MANIFEST.csv` | Pass/fail checks; index of checks and reports; size and SHA-256 of every local file | `validate_*.py`, `write_manifests.py` |

## Run order

Run from the project folder with one BLAS thread per process:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    export NUMBA_CACHE_DIR=/tmp/korean_mw_ml_numba MPLCONFIGDIR=/tmp/korean_mw_ml_matplotlib

1. **Inputs.** `python code/prepare_study.py`, `python code/prepare_responses.py`, `python code/fetch_KR_Mw.py`.
2. **Waveform processing.** `python code/run_catalog.py --workers 8`, `python code/local_magnitude.py --workers 8`,
   `python code/finalize_local_magnitudes.py`. A first run reads every waveform and takes hours. Each
   per-event result stores a signature of the processing code and settings, and is reused while they are
   unchanged; with all results current, this step takes about 30 s.
3. **Processing checks.** `python code/validate_hg_processing.py`, `python code/audit_estimator_waveforms.py`.
4. **HH–HG appendix** (scripts in `hh_hg_comparison/code/`):
   - `download_sangju.py` downloads the Sangju event from NECIS with
     [necis-downloader](https://github.com/seismoseo/necis-downloader) (once; needs NECIS credentials).
   - Then `prepare_pairs.py`, `prepare_pairs.py --sangju`, `compare_pairs.py`, `waterlevel_transfer.py`,
     `plot_comparison.py`, `build_report.py`, `validate_comparison.py`.
5. **Catalog and fits.** `analyze_catalog.py`, `update_sangju_hg.py`, `analyze_local_magnitude.py`,
   `regression_controls.py`, `hong_regressions.py`, `compare_estimator_stability.py`.
6. **Reports.** `report_source_velocity.py`, `compare_KR_Mw.py`, `report_Hong_regressions.py`,
   `report_hg_update.py` (main report, with the HH–HG appendix attached), `make_atlas.py`,
   `sync_sangju_magnitudes.py` (copies the Sangju magnitudes to `../sangju_20261117`).
7. **Checks.** `validate_hg_release.py`, `validate_source_velocity_update.py`.
8. **Potency.** `analyze_potency.py`, `potency_regional_controls.py`, `analyze_magnitude_followup.py`,
   `audit_potency_literature.py`, `report_potency.py` (pages 7–11 come from `report_magnitude_followup.py`),
   `validate_potency.py`, `validate_magnitude_followup.py`.
9. **Index.** `write_manifests.py`.

Steps 5–9 take about 5 minutes. `update_source_velocities.py` is not in the list. It converted the
stored results from the earlier fixed 3.5 km/s source velocity to the depth-dependent one once, on 19
November 2026, and refuses to run again; a new run of step 2 applies the depth-dependent velocity
directly.

## Not in the repository

- **Waveforms.** KMA and NECIS waveforms are not redistributed: the Sangju records in
  `inputs/Sangju_counts/`, and the event and continuous records of `hh_hg_comparison/inputs/`.
- **Per-event results and logs** (about 450 MB), regenerated by step 2.
- **Instrument responses.**
  - `inputs/master_velocity.xml` is built by `prepare_responses.py` from the station metadata of Ahn et
    al. (2024), whose repository has no license.
  - The NECIS response files.
- **The KR_Mw catalog files.** The repository has no license; `code/fetch_KR_Mw.py` downloads them at the
  pinned commit and checks their checksums.
- **Paper PDFs.** They are listed in [`references/README.md`](references/README.md).

`tables/MANIFEST.csv` gives the size and SHA-256 of all of these as used. The scripts use absolute paths
on the analysis server and read from these other projects:

| Project | Used for |
|---|---|
| `Claude/sangju_20261117` | National depth table (event selection); Sangju event, picks, stations and NECIS responses; receives the Sangju magnitudes |
| `16.kma_absolute_location`, `38.SKP_lower_crustal_earthquakes` | Picks and records within 100 km of each event |
| `27.Haenam_sequence_renewal` | Audited response inventory, station table, ML recipe, Haenam HH and HG records |
| `5.P_wave_moment_tensor_inversion/Response` | Station metadata of Ahn et al. (2024), v1.0.2 |
| `Claude/data/necis`, `Claude/necis` | NECIS event catalog and continuous data; necis-downloader |

## Software

Python 3.12.12 with numpy 2.3.5, scipy 1.17.0, pandas 2.2.2, ObsPy 1.4.2, multitaper 1.2.0,
matplotlib 3.10.8, PyMuPDF 1.27.2.3 and lxml 6.0.2.

## Author

Min-Seong Seo, School of Earth and Environmental Sciences, Seoul National University.
