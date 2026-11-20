# Moment and local magnitudes of South Korean earthquakes

Moment magnitude (Mw) from S-wave spectra and an independently measured local magnitude (ML) for the
642 earthquakes of KMA ML ≥ 2 from January 2010 to October 2026 whose depths are well determined, and
for the ML 3.6 Sangju earthquake of 17 November 2026, which is kept out of all fits. The two magnitudes
are compared with each other, with the KMA ML, with the published Mw catalog of Hong and Sheen (2026),
and with potency magnitude.

**Original ML≥2 report:** [`reports/Korean_Mw_ML_study.pdf`](reports/Korean_Mw_ML_study.pdf), 21 pages: methods,
results, figures and the sensor appendix. Report numbers are read from their corresponding `tables/` folders.

**Smaller-event extension:** [Extended_Mw_ML_scaling.pdf](small_events/extension/reports/Extended_Mw_ML_scaling.pdf),
27 pages, now covering **ML, Mw and scalar potency magnitude Mp together**. All 2,317 eligible catalog-ML 1.5–1.9 events, a fixed lower-magnitude sample and the 642 reference
events were assessed: **3,532 attempts** in total. The uniform NW2/three-taper method gives **304 smaller
paired-A events**, 293 outside the pilot; **166 new events have measured ML 1.5–2.0**. The family-weighted reference line is
Mw = 2.200 + 0.717(ML − 2), based on 676 paired-A events in 374 families; quadratic and
fixed-knot models under family weights do not improve prediction of the new small events using our measured ML.
For a description across magnitude ranges, the earlier **50-bin median method** gives
Mw = 2.181607 + 0.669254u + 0.037806u², u = measured ML − 2. Its large-event prediction RMS
is 0.108, versus 0.144 for the bin-median line and 0.169 for the family-weighted line.
Equal-bin weights restore the emphasis on sparse large events that family weights lose after expansion.
The corresponding Mp curve is **Mp = 2.161185 + 0.673739u + 0.034604u²**. It lowers large-event
prediction RMS from 0.1365 for the bin line to 0.1006 for the bin quadratic. Mp is available for all
**2,000 finite-Mw cases** and uses the earlier Liu normalization with depth-dependent source rigidity.
The [expanded potency report](small_events/extension/reports/Potency_magnitude_and_quadratic_scaling.pdf)
extracts the eleven-page joint section: definitions, both ML axes, two velocity models, equations,
regression methods, large events, depth trends, rigidity, Hong comparisons and the same-WA-amplitude ML attenuation/gain controls.
**Independent Wood–Anderson ML was measured for 2,222 events, including 1,698 of grade A.**
The [fourteen-page regression comparison](small_events/extension/reports/Linear_quadratic_Mw_ML_comparison.pdf)
starts with the matched Mp/Mw definitions, curves and equations, then gives the weight diagnosis and
family-weighted linear/quadratic equations, earlier larger-event curves, OLS/ODR and prediction tests for
both measured ML and KMA ML. The KMA-ML quadratic has appreciable curvature; it is a separate result.
This is scaling among accepted measurements.
The original 642-event results are preserved. [All-attempt ML/Mw/Mp catalog](small_events/extension/tables/potency_event_catalog.csv)
and [methods/run order](small_events/extension/README.txt).

The earlier [pilot report](small_events/reports/Small_event_Mw_ML_pilot.pdf) records the fixed initial
selection and the window/taper experiments that preceded the expansion.

| Report | Pages | Content |
|---|---|---|
| [`Korean_Mw_ML_study.pdf`](reports/Korean_Mw_ML_study.pdf) | 21 | Main report; ends with the HH–HG appendix |
| [`Independent_ML_and_Mw_comparison.pdf`](reports/Independent_ML_and_Mw_comparison.pdf) | 10 | The ML part of the main report alone |
| [`Velocity_models_and_magnitudes.pdf`](reports/Velocity_models_and_magnitudes.pdf) | 3 | S velocity at the source from two velocity models and its effect on Mw |
| [`Estimator_and_KR_Mw_comparison.pdf`](reports/Estimator_and_KR_Mw_comparison.pdf) | 6 | Multitaper versus direct Fourier spectra; first comparison with KR_Mw |
| [`Hong_KR_Mw_regression_and_stability.pdf`](reports/Hong_KR_Mw_regression_and_stability.pdf) | 9 | Method differences from Hong and Sheen (2026); OLS, orthogonal and Deming fits |
| [`Potency_magnitude_and_quadratic_scaling.pdf`](reports/Potency_magnitude_and_quadratic_scaling.pdf) | 11 | Potency magnitude; linear versus quadratic ML scaling; depth trend |
| [`Snoke_coefficient_and_bandwidth_audit.pdf`](reports/Snoke_coefficient_and_bandwidth_audit.pdf) | 6 | Ahn and Kang (2026): coefficient audit, bandwidth, source-shape and moment-stability tests |
| [`Spectral_fit_atlas.pdf`](reports/Spectral_fit_atlas.pdf) | 161 | Spectrum and fit of every event, rejected ones included |
| [`hh_hg_comparison/reports/HH_HG_paired_comparison.pdf`](hh_hg_comparison/reports/HH_HG_paired_comparison.pdf) | 8 | Co-located broadband (HH) and accelerometer (HG) records at Haenam and Sangju |
| [`Small_event_Mw_ML_pilot.pdf`](small_events/reports/Small_event_Mw_ML_pilot.pdf) | 7 | Fixed small-event pilot, all failures, independent ML coverage, window/taper controls and exploratory fits |
| [`Small_event_spectral_atlas.pdf`](small_events/reports/Small_event_spectral_atlas.pdf) | 45 | Every one of the 180 attempted pilot/control events, including failures |
| [Extended_Mw_ML_scaling.pdf](small_events/extension/reports/Extended_Mw_ML_scaling.pdf) | 27 | Joint ML/Mw/Mp study: independent ML, original potency definition, matched methods and historical controls, depth/velocity/Hong comparisons, spectral controls and spectra |
| [Expanded potency report](small_events/extension/reports/Potency_magnitude_and_quadratic_scaling.pdf) | 11 | The joint Mw/Mp section alone, with coverage matching the earlier ML≥2 potency study |
| [Linear_quadratic_Mw_ML_comparison.pdf](small_events/extension/reports/Linear_quadratic_Mw_ML_comparison.pdf) | 14 | Matched Mp/Mw equations and curves, followed by the fitting-objective audit and detailed family-weighted Mw controls |
| [Accepted_small_event_spectra.pdf](small_events/extension/reports/Accepted_small_event_spectra.pdf) | 76 | Every smaller event passing both A-quality checks; 304 events |

## Original ML≥2 study

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
9. **Snoke coefficient audit.** Ahn and Kang (2026) identify an inconsistency in one alternative
   Brune K–J integral formula. Our final plateau comes from direct spectral fitting, so no Snoke
   coefficient correction is applied. Independent integral calculations, the saved-spectrum replay,
   passband and source-model sensitivities are in the new audit report. These diagnostics preserve
   adopted Mw/ML, regressions and previous reports.

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
- **Ahn and Kang (2026) audit.** The paper's algebra and finite-band benchmark reproduce independently.
  All 620 finite adopted Mw estimates, including Sangju, replay to numerical precision. For 443
  Mw grade-A events, the diagnostic Snoke estimate uses an assumed high-frequency tail supplying a
  median 40% of its velocity-square integral. This qualifies source-corner and stress-drop use, not
  the observed plateau by itself. Source-model review cases are listed in
  `tables/snoke_model_review_events.csv`; no blanket 0.10 magnitude adjustment is justified.

- **Smaller-event extension.** The 3,532-attempt expansion gives 676 paired-A events in 374 families
  at measured ML≥1.5. The family-balanced slope is 0.717 (interval 0.700–0.734); equal-axis
  ODR gives 0.730. Family-held-out RMS on the 166 new measured-ML 1.5–2 events is 0.071. The uniform
  NW2 estimates differ from the original method by median −0.019 Mw and RMS 0.026 across 424 shared
  Mw-A controls. Full-census and pilot-family-exclusion sensitivities support the linear result.
  Independent ML is finite for 2,222 events (1,698 A); all 46,117 selected station ML readings
  reproduce from their archived Wood–Anderson N/E peak amplitudes. Full previous/expanded
  linear and quadratic coefficients are in `regression_comparison_models.csv`. Using measured ML,
  expanded family-weighted curvature is −0.000095; using KMA ML it is +0.062359 (interval +0.0301–+0.1038).
  Both axes use identical event membership; matched NW2 larger-only versus expanded fits separate
  the addition of smaller events from the spectral-method change.
  The restored 50-bin OLS method gives measured-ML curvature +0.037806 (bin ODR +0.037289).
  On the six measured-ML>4 events, its fitted RMS is 0.0878 and whole-family-held-out event RMS
  is 0.1082; the corresponding family-weighted quadratic gives 0.1673 and 0.1773.
  New-small-event prediction RMS is 0.0686, close to the family line's 0.0685. No event is trimmed
  from the measured-ML bin fits. Large events carry only 0.606% of the expanded family loss,
  compared with 1.815% previously. Tables `fit_objective_*` retain all 80 audited specifications.
  Counts, failures and every comparison are in `small_events/extension/tables/`; the earlier national
  catalog, regressions and reports remain unchanged.

## Layout

| Path | Content |
|---|---|
| `code/` | All scripts, in one folder because they import each other. Run order below. |
| `inputs/` | Event list (`cohort.csv`), processing settings (`study_config.json`), quality rules, response sources, the two velocity models (`source_velocity_models.json`), the KR_Mw source record, a map coastline, the earlier Mw and ML from velocity sensors only (before HG was added; used for the before–after comparison), and a copy of the Haenam ML recipe that `local_magnitude.py` imports from the Haenam project. |
| `tables/` | Catalogs, fits, regressions, checks, the release index and the file manifest (groups below). |
| `figures/` | Figures 01–26 of the reports, as PNG and PDF; the Snoke audit also includes SVG. |
| `reports/` | The PDF reports. |
| `hh_hg_comparison/` | The paired-sensor study: its own `code/`, `tables/`, `figures/` and report; `inputs/` holds the pair list and checksums. |
| `small_events/` | Initial pilot and the completed `extension/`: frozen selections, all attempts, uniform-estimator scaling, figures, reports and validation. Earlier adopted values preserved. |
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
| `snoke_*` | Ahn and Kang equation audit; matched-band integral diagnostics; source-shape and Mw–ML sensitivity controls | `snoke_audit.py`, `report_snoke_audit.py` |
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
9. **Snoke audit.** `snoke_audit.py`, `report_snoke_audit.py`. These read existing spectra and preserve
   adopted results; `--falloff-only` on the first script reuses the completed integral calculation
   when only the additional model-bound controls are required.
10. **Smaller events.** `small_events/README.txt` records the initial pilot. The completed expansion
    has its own commands in `small_events/extension/README.txt`: prepare, run, analyze, compare, report and
    validate. Both preserve the earlier national study and reports.
11. **Index.** `python code/write_manifests.py --project-only` for this project, or without that flag to
    also refresh the Sangju manifest.

Steps 5–9 take several minutes with cached spectra; a fresh pilot also requires waveform processing.
`update_source_velocities.py` is not in the list. It converted the
stored results from the earlier fixed 3.5 km/s source velocity to the depth-dependent one once, on 19
November 2026, and refuses to run again; a new run of step 2 applies the depth-dependent velocity
directly.

## Not in the repository

The 120-MiB joint prediction table is included losslessly as
[`magnitude_scaling_predictions.csv.gz`](small_events/extension/tables/magnitude_scaling_predictions.csv.gz).
After a fresh clone, restore its canonical CSV before running the extension validation or report scripts:

```sh
gzip -dk small_events/extension/tables/magnitude_scaling_predictions.csv.gz
```

The uncompressed CSV remains on the analysis server. Its exact size and SHA-256 are recorded in
`tables/MANIFEST.csv` and the extension validation records.

- **Waveforms.** KMA and NECIS waveforms are not redistributed: the Sangju records in
  `inputs/Sangju_counts/`, and the event and continuous records of `hh_hg_comparison/inputs/`.
- **Per-event results and logs** (about 450 MB), regenerated by step 2.
- **Pilot per-event caches and logs**, regenerated by the commands in `small_events/README.txt`.
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
