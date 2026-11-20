Small-event Mw–ML feasibility pilot — 19 November 2026

The completed 3,532-event expansion is in extension/README.txt and
extension/reports/Extended_Mw_ML_scaling.pdf. This document retains the
initial pilot design and results.

The first realistic target is measured ML about 1.5 for selected events. This
pilot does not establish an unbiased national low-magnitude scaling relation.
The adopted magnitudes, regressions and reports of the ML>=2 study are preserved.

SELECTION
The confident-depth catalog has 9,325 eligible South Korean land/offshore events
with catalog ML 0.5–1.9 (2010–October 2026; G2/G3 depths). Six were drawn uniformly
without replacement in each of 27 magnitude/depth/era strata, seed 20261119:
  catalog ML: 0.5–0.9, 1.0–1.4, 1.5–1.9;
  source depth: <10, 10–16.89, >=16.89 km;
  era: 2010–2016, 2017–2021, 2022–October 2026.
There are 162 small events, 54 per magnitude bin. Another two ML 2.0–2.4 events
per depth/era cell give 18 controls. Selection preceded waveform inspection;
no failed or unavailable event was replaced. All 180 have retained waveforms.
The source priority is the same as the parent study: pilot, national, kma16.
Known-event overlap checking includes all domestic catalog magnitudes, not only
ML>=1.5. Unknown/undetected neighbors remain a limitation.

Same dated response inventory, HH/EL/HG, hypocentral spreading and empirical Q,
source-depth Vs from Kim & Kim (1983), rho=2800 kg/m3, and C=9.05. Primary FAS is
sqrt(N*dt*one-sided-multitaper-PSD/2); direct Fourier is a crosscheck. Acceptance
uses the original A/B station, plateau, bandwidth, estimator, source-shape, RMS
and common-station window gates. ML is the same independent Wood–Anderson,
Sheen et al. (2018), 10–100 km estimate in all three spectral experiments.
Neither the draw nor magnitude acceptance uses agreement between Mw and ML.

RESULTS (SMALL EVENTS ONLY; ALL 162 IN EACH SCHEME)
Scheme                 Paired A   Paired A/B   Measured-ML range, paired A
10.1/20.1 s, NW4 K7        7          20                 1.78–2.23
  band 0.5–20 Hz
3.1/5.1 s, NW4 K7          5           8                 1.78–2.23
  band 1.5–20 Hz
5.1/10.1 s, NW2 K3        11          26                 1.48–2.23
  band 0.5–20 Hz

The unchanged pipeline gives 9 Mw A, 11 B, 35 C, 107 F; independent ML gives
45 A, 26 B, 91 F. Catalog ML below 2 does not imply measured ML below 2. The
existing reference has 357 paired-A events, minimum measured ML about 1.89.
Of 54 trials per catalog-ML bin, unchanged paired-A counts are 0,1,6; NW2
candidate counts are 2,3,6. They need not be the same accepted events.

All finite unchanged-control Mw/ML values and grades reproduce the existing
catalog to numerical precision. Against unchanged Mw-A controls, NW4 3/5 s has
11 shared Mw-A controls, median delta=-0.01355, RMS=0.02321. NW2 5/10 s has
12 shared Mw-A controls, median delta=-0.01380, RMS=0.02072, max |delta|=0.04364.
Both pass the fixed transfer gate (>=5 shared controls, |median|<=0.05,
RMS<=0.10). No amplitude offset was fitted. These are combined window/taper/band
experiments, not a controlled estimate of the effect of window length alone.
Transfer agreement is necessary but not proof of absolute small-event accuracy.

The 3/5-s NW4 trial loses low frequencies because of taper bandwidth; it does
not improve coverage. NW2 is a promising candidate for measured ML about 1.5,
but is not adopted. Do not select the most favorable method per event or pool
methods according to Mw–ML agreement. Do not infer reliable scaling to ML 0.5.

An explicitly secondary A/B regression comparison adds 20 original-window
pairs to the old 357, with family weights, 1,500 whole-family bootstrap draws
and whole-family held-out prediction. Held-out RMS on the 20 small pairs is
0.1188 (line), 0.1202 (quadratic), 0.1227 (continuous lines with knot at measured
ML 2). Equal-axis ODR is a sensitivity; none of these corrects SNR-dependent
missingness. Seven strict additions do not justify a new primary curve. The
NW2 candidate is shown for feasibility and is not used to refit the relation.

LIMITS
The parent is conditional on earlier detection, picking and confident depths.
Balanced pilot fractions are not national detection rates. A broadband usable
band and sufficient useful sites are the main limitations. Every failed case
is retained, including nonfinite estimates. Conditional station bootstrap
intervals hold source shape/model fixed and exclude absolute response,
velocity, attenuation, radiation and site uncertainty; they are not total
95% confidence intervals. Unit-area, noise-free pulse controls test operators
only; real path coda, site effects and noise are absent. No amplitude correction
is inferred from those pulse controls.

FILES
reports/Small_event_Mw_ML_pilot.pdf          Seven-page findings and methods
reports/Small_event_spectral_atlas.pdf       All 180 trials, 45 pages
inputs/pilot_config.json                    Frozen design and parent hashes
inputs/pilot_cohort.csv                     All selected cases and input paths
inputs/eligible_small_events.csv            Full 9,325-event eligible parent
inputs/processing_provenance.json           Code/settings/neighbor hashes
inputs/protected_results.json               Adopted tables/report checksums
tables/pilot_Mw_ML_catalog.csv               Original scheme, all 180 cases
tables/short_window_Mw_ML_catalog.csv        NW4 3/5 s, all 180 cases
tables/nw2_window_Mw_ML_catalog.csv          NW2 5/10 s, all 180 cases
tables/nw2_window_candidate_pairs.csv       The 11 NW2 paired-A candidates
tables/short_window_candidate_pairs.csv     The five short-NW4 paired-A cases
tables/yield_by_magnitude.csv               Original successes and denominators
tables/failure_reasons.csv                  Multiple failures can apply
tables/exploratory_regressions_AB.csv       Secondary selected A/B fits only
tables/family_heldout_predictions_AB.csv    Held-out predictions
tables/pulse_operator_controls.csv         Known-input operator diagnostics
tables/report_validation.json              PDF, source hashes and layout checks
tables/validation.json                     Sampling/scientific consistency audit

CSV CONVENTIONS
event_id and origin_utc use UTC; event_id is a string. time_kst, where present,
is UTC+9. ml is KMA catalog ML; ML_Sheen is our independent ML. lat/lon are
relocated coordinates (degrees); z_central is source depth in km; lower/upper
are inherited depth-model/station-deletion bounds, not single-model formal
standard errors. Missing numerical estimates remain blank in CSV.
Mw_quality and ML_quality are A/B/C/F as applicable. paired_A requires both A;
paired_A_or_B requires both A or B. n_stations_SNR counts selected sites with
at least 20 usable frequency samples; common_window_stations uses their
intersection. n_HH/n_EL/n_HG count selected spectral sensor records, not files.
difference_Fourier, difference_Brune and window_difference_common are absolute
Mw differences. Mw_station_bootstrap_lo/hi and ML_station_bootstrap_lower/upper
are conditional 2.5/97.5-percentile station resampling bounds described above.
primary_window_s, secondary_window_s, multitaper_NW and multitaper_K identify
the actual experiment. No candidate is an additional adopted primary result.

REPRODUCTION (FROM THE PROJECT ROOT)
Use the thread/cache environment in the root README. Required waveform inputs
are read from the existing server projects; no external project is edited.
1. python code/prepare_small_event_pilot.py
2. python code/run_small_event_pilot.py --workers 6
3. python code/analyze_small_event_pilot.py
4. python code/small_event_short_windows.py
5. python code/analyze_small_event_pilot.py --short-windows
6. python code/small_event_nw2_windows.py
7. python code/analyze_small_event_pilot.py --nw2-windows
8. python code/report_small_event_pilot.py
9. python code/validate_small_event_pilot.py
10. python code/write_manifests.py --project-only
Assessment plans in inputs/ fix control-transfer gates. Preparation snapshots
the current adopted checksums; retain the supplied protected_results.json when
auditing this run. Per-event caches, spectra and logs are local and ignored by
git; all-attempt tables, design, validation, reports and figures are retained.
