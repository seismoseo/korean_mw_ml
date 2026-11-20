Expanded ML–Mw–Mp study toward measured ML 1.5

This extension uses one spectral method throughout the small and reference
events. The original national study and the 162-event pilot are preserved.
Processing completeness and final counts are in tables/summary.json.

JOINT POTENCY AND MOMENT ANALYSIS (20 NOVEMBER 2026)
The established ML>=2 workflow is now carried forward for both Mw and Mp.
All 2,000 finite Mw estimates have scalar source potency and Mp; the 801 Mp-A
grades inherit their spectral Mw grade. Main regressions retain exactly the
same 676 paired-A events in 374 families. Missing Mw always gives missing Mp.
All measured Mw/ML values, spectral gates and previous reports are preserved.

Use P0=M0/[rho*Vs(z)^2] in m3, with rho=2800 kg/m3 and source-depth Vs from
Kim & Kim (1983); Kim et al. (2011) is the comparison. Mp retains the earlier
Liu definition: Mp=(2/3)[log10(P0[m3])+1.4563]. Its approximately 36-GPa
reference is a magnitude normalization, while source rigidity is depth dependent.
Mw retains C=9.05. Alternative 28/36-GPa and C=9.05/9.10 normalizations are
explicitly tabulated, rather than silently shifting Liu's published curve.

With u=measured ML-2, the primary bin-OLS descriptive curves are
  Mw = 2.181607 + 0.669254 u + 0.037806 u^2
  Mp = 2.161185 + 0.673739 u + 0.034604 u^2.
On the six ML>4 targets, whole-family-held-out event RMS changes from
0.144355 to 0.108156 for Mw and 0.136450 to 0.100551 for Mp. The Mp quadratic
new-small-event RMS is 0.0651. These are selected-event comparisons with five
independent large-event families, not a national conversion calibration.

Every earlier 128 Mw/Mp regression specification and native-target prediction
score reproduces. The joint table contains 480 cases: both local magnitudes,
both source models, previous/expanded NW4 and larger/expanded NW2 cohorts,
family/event OLS and ODR, bin OLS and added bin ODR, plus the earlier bin-count
and minimum-support controls. Whole-family bootstrap uses actual per-case
counts of 1,000-2,000; reused joint draws retain their saved counts.

The original 84 depth controls are repeated for the expansion, with 24
larger-only controls added. After joint quadratic measured-ML adjustment,
Mp (Kim1983) has depth slope -0.0372 per 10 km, interval [-0.0550,-0.0204];
Mw has -0.0079, interval [-0.0260,+0.0093]. Results are conditional on the
selected catalog, source model, path/site assumptions and local-magnitude
calibration; the trend alone does not establish depth as the cause.

Hong comparisons retain published spectral/time-domain Mw. Hong Mp is a
source-constant conversion using Hong's rho=2700 kg/m3 and Vs=3.61 km/s.
There are 182 paired-A exact matches. Full-catalog and selected-match OLS/ODR
regressions are kept separate. The original Liu curve is plotted unshifted. The same-WA-amplitude HB attenuation/distance
and 2800/2080 gain controls are repeated (32 fits); the gain shift preserves
curvature and prediction errors. Hong matched-ML-range selection is retained.
potency_report_coverage.csv lists the topics carried forward from the old study.

COMPLETED RESULTS
All 3,532 cases are accounted for. There are 304 smaller catalog events with
both magnitudes grade A, including 293 outside the original pilot; 166 new
cases have measured ML1.5–2.0. Primary fits use 676 events in 374 families at
measured ML>=1.5. The family-balanced line is
  Mw = 2.200121 + 0.717134 (ML - 2)
with slope interval [0.700447, 0.734321]. Equal-axis ODR slope is 0.730211.
A family-weighted line predicts the new low-magnitude cases with family-held-out RMS 0.070753;
quadratic/fixed-knot models do not improve prediction. Their curvature/slope-
change intervals include zero for our measured ML. KMA ML is assessed separately
in the explicit regression comparison below. The census-only slope is 0.715772; exclusion
of all pilot families gives 0.714600. These are accepted-event relations.

The transfer gate passes with 424 shared Mw-A controls: median delta=-0.019413,
RMS=0.025644, max |delta|=0.088577. The original method gives 239 smaller
paired-A cases versus 304 for NW2. The retained accepted-small-event archive
contains no HH/HG pair within 100 m for the new paired-sensor diagnostic.
No amplitude correction is fitted. Scientific and PDF checks pass; legacy
adopted catalogs and initial pilot outputs are unchanged.

INDEPENDENT ML AND THE REGRESSION REPORT REVISION (20 NOVEMBER 2026)
ML was measured for 2,222 events, including 1,698 grade A. For catalog ML<2,
the counts are 1,614 finite and 1,200 A. The primary regression uses ML_Sheen,
our waveform-based Wood-Anderson ML; ml is the separate KMA catalog value.
All 46,117 selected station ML readings reproduce from their archived N/E
peak amplitudes using the published epicentral-distance attenuation relation.
Event medians and grades also reproduce. This is an amplitude-to-ML audit,
not a new absolute instrument/path calibration. Measured magnitudes, grades,
selections and all earlier adopted national/pilot results remain unchanged.

Extended_Mw_ML_scaling.pdf now has 27 pages. Pages 2-3 document independent ML;
pages 5-15 give the joint Mw/Mp analysis and are extracted as the eleven-page
Potency_magnitude_and_quadratic_scaling.pdf. The later sections retain detailed
Mw family/weight/transfer/spectral controls. Linear_quadratic_Mw_ML_comparison.pdf
has 14 pages, starting with the matched Mw/Mp definitions, curves and equations.
The comparison includes full equations and intervals for a line and quadratic,
with both our ML and KMA ML on identical event sets. The four cohorts are:
  previous NW4: 357 events / 194 families (original saved measurements/fits);
  expanded NW4: 582 / 343;
  larger-only NW2: 375 / 199;
  expanded NW2: 676 / 374.
Family-balanced OLS (2,000 bootstrap draws) and equal-axis ODR (1,000 draws)
are supplied for every case. ODR assumes equal axis error variance; it does
not infer individual measurement uncertainties. Previous coefficients and
joint bootstrap samples are reused exactly, re-centered from ML3 to ML2.

With u=ML-2, the expanded measured-ML quadratic is
  Mw = 2.200117 + 0.717236 u - 0.000095 u^2,
compared with the earlier larger-event curvature +0.030111. The expanded
KMA-ML quadratic is Mw=2.358574+0.581725 u+0.062359 u^2, with curvature interval
[+0.030114,+0.103827]. The latter does not improve prediction on the 166 new
measured-ML1.5-2 events (RMS line=0.09790, quadratic=0.09832), but lowers RMS
for the six measured-ML>=4 events in five families (0.2842 to 0.2277).
The few large-event families and extrapolation limit that comparison.

Prediction comparisons always score the same 676 uniform-NW2 target Mw values,
with each target family omitted from every training cohort. Magnitude ranges
are defined using measured ML for both axis definitions. Earlier NW4 fits
versus NW2 targets include a method change; larger-only NW2 versus expanded
NW2 keeps the method fixed. The report marks extrapolated curves; the CSVs
also flag each held-out prediction outside its training magnitude range.

FITTING LARGE EVENTS: METHOD AUDIT (20 NOVEMBER 2026)
The earlier headline family-weighted method was reproduced correctly, but
the earlier equal-event OLS and 50-bin median OLS alternatives were omitted
from the extension. Their previous coefficients and native-target held-out
scores now reproduce, using the unchanged potency_regression.py engine.
The bin ODR fit is an additional equal-axis sensitivity on the same medians.

Family weighting assigns each event 1/(accepted family members). Adding
small events reduces the weight share of the six measured-ML>4 cases from
1.815% to 0.606%. Within-family dilution compounds the larger number of
families: Gyeongju grows from 93 to 122 members, while the 2015 and 2019
large-event families each grow from one to two members. This explains why
the same numerical objective loses emphasis on the large-event tail.

The restored bin method uses 50 equal bins over observed training ML;
trim |Mw-initial bin median|>0.5, recompute median ML/Mw, fit each occupied
bin with equal weight. For measured ML, 38 bins are occupied (8 singletons)
and no event is trimmed. The descriptive expanded NW2 quadratic is
  Mw = 2.181607 + 0.669254 u + 0.037806 u^2, u = measured ML - 2.
Its family-bootstrap curvature interval is [-0.021183,+0.044037]. Bin ODR
gives curvature +0.037289. Expanded NW4 bin OLS gives +0.035951. The earlier
357-event bin-OLS curvature was +0.050881; its family-OLS value was +0.030111.

For the same six measured-ML>4 NW2 targets in five families:
  Objective/form             Fitted event RMS    Held-out event RMS
  Family OLS line                 0.166973             0.168618
  Family OLS quadratic            0.167308             0.177307
  Bin OLS line                    0.120901             0.144355
  Bin OLS quadratic               0.087786             0.108156
  Bin ODR quadratic               0.087750             0.106088
On the 166 new measured-ML1.5-2 targets, bin-OLS quadratic held-out event RMS
is 0.068639 versus 0.068528 for the family line. Every holdout removes the
target's entire family and recomputes bins/trim/weights. Both large Gyeongju
events require extrapolation in their held-out fold. Curvature coefficient
intervals resample whole families; paired prediction-gain intervals resample
the five existing held-out losses. The latter is not new-population evidence.

Use the bin curve for the requested descriptive fit across magnitude ranges;
retain family-weighted curves as a different objective. Neither changes a
measured magnitude, grade or accepted event. fit_objective_* records 80
specifications for both ML definitions and all four earlier/expanded cohorts.

FIXED SELECTION
3,532 unique events, 2010–October 2026, all with retained waveforms:
  2,317: every eligible confident-depth catalog-ML 1.5–1.9 event;
    519: nested random ML 1.0–1.4 sample, total min(60,N) per depth/era cell;
     54: original pilot cases at catalog ML 0.5–0.9;
    642: all ML>=2 reference events.
The nested draw retains the original six random cases per cell and selects
the remaining cases uniformly without replacement, seed 20261120. One old
deep/early cell has 39 events and is fully included. No failure is replaced.
The original 162 small-event pilot cases are identified; confirmatory_small
means a selected small event outside that pilot. Exclusion of every pilot
family is an additional sensitivity, not a change in primary acceptance.

METHOD
Primary: 5.1/10.1 s S windows, NW=2, three tapers, 0.5–20 Hz. Control: original
10.1/20.1 s, NW=4, seven tapers. The primary windows have nearly equal spectral
half-bandwidths, about 0.39 Hz; shorter duration and taper count change together.
Same HH/EL/HG response processing, phase picks,
SNR>=3, spreading/Q and independent Wood–Anderson ML. Source-depth Vs from
Kim & Kim (1983), rho=2800 kg/m3, C=9.05; Kim et al. (2011) comparison retained.
The primary FAS remains sqrt(N*dt*one-sided-MT-PSD/2), with direct Fourier
crosscheck. Original A/B rules are read from inputs/analysis_quality_rules.json
at the project root and are verified against their frozen checksum.

All known domestic neighbors, including catalog ML<1.5, enter the same overlap
screen. Exact input geometry/paths and identical known-neighbor lists permit
reuse of an audited baseline cache. Of the reference cases, 126 required
reprocessing; 516 could reuse prior audited inputs. Reuse hashes are frozen
in inputs/reuse_registry.csv. Existing input/output files are not edited.

TRANSFER AND SCALING
The fixed transfer gate is >=30 shared reference Mw-A events, |median delta|
<=0.05 and RMS delta<=0.10. Magnitude bins with >=5 controls also require
|median delta|<=0.05. No amplitude offset or residual-based event selection.
The transfer result is in tables/summary.json and tables/transfer_controls.csv.
A passing gate establishes consistency between methods on shared accepted
events, not absolute magnitude accuracy or complete low-SNR coverage.

Primary regressions: both Mw and measured ML grade A; measured ML>=1.5;
uniform NW2 estimator; one total weight per family. Compare a line, quadratic
and continuous lines with a fixed knot at measured ML2. Intervals use 2,000
whole-family bootstrap draws. CV withholds entire families, including their
reference members. Model comparison on fresh small events excludes the
original pilot from scoring, with 5,000 paired family resamples of held-out
squared-error differences. The original pilot families are also excluded in
a separate regression sensitivity. This is selected-event scaling.

Sensitivities: ODR (equal axis error), Deming variance ratios 0.25 and 4, A/B
inclusion, within-family and event inverse sampling-probability weights, the
catalog-ML>=1.5 census alone, measured-ML
floors 1.7/2.0/2.3, Kim2011 source Vs and matched reference event sets. The
within-family weighting retains one total weight per family. Event design
weights use 1/probability per accepted event, with whole-family resampling.
Neither these weights nor ODR/bootstrapping recover unmeasurable events.

SENSOR CHECK
Where available, original 10.1-s NW4 archives compare HH/HG records within
100 m, requiring both sensors to pass SNR over >=half of a diagnostic band
and at least eight frequency samples. This diagnostic uses no Mw–ML residual,
does not alter primary sensor selection and fits no amplitude correction.
Zero pairs is an explicit data-availability/usable-band outcome, not evidence
of sensor agreement. The earlier dedicated HH–HG study remains separate.

OUTPUTS
reports/Extended_Mw_ML_scaling.pdf       Methods, findings and figures
reports/Linear_quadratic_Mw_ML_comparison.pdf Explicit previous/expanded fits
reports/Accepted_small_event_spectra.pdf Every accepted smaller paired-A case
reports/Potency_magnitude_and_quadratic_scaling.pdf Joint Mw/Mp section alone
tables/potency_event_catalog.csv        All 3532 ML/Mw/Mp attempts, primary NW2
tables/potency_NW4_catalog.csv          Corresponding original-method control
tables/magnitude_scaling_members.csv    Every earlier/expanded fitting cohort
tables/magnitude_scaling_regressions.csv All 480 matched Mw/Mp specifications
tables/magnitude_scaling_predictions.csv Native/shared-target family holdouts
tables/magnitude_scaling_bins.csv       Exact fitted bin medians and counts
tables/magnitude_scaling_bootstrap.npz   Joint whole-family coefficients
tables/magnitude_scaling_performance.csv Fit/CV event/family RMS by target range
tables/magnitude_scaling_prediction_gains.csv Paired held-out-loss comparisons
tables/potency_depth_*                  108 depth controls and bootstrap/CV
tables/potency_Hong_*                   Published comparisons and OLS/ODR
tables/potency_normalization.csv        28/36-GPa and magnitude-zero-point offsets
tables/potency_Liu_comparison.csv       Residuals from the unshifted external curve
tables/potency_report_coverage.csv      Explicit correspondence with earlier study
tables/potency_ML_definition_controls.csv Same-peak attenuation/distance/gain controls
tables/potency_regional_*               32 regional-ML control fits and bootstrap/CV
tables/potency_validation.json          Independent physical/statistical checks
tables/fit_objective_models.csv         80 method/cohort/axis/degree cases
tables/fit_objective_weights.csv        Per-event family weight and membership
tables/fit_objective_bins.csv           Bin medians and initial/retained counts
tables/fit_objective_predictions.csv    Native and shared-target family holdouts
tables/fit_objective_performance.csv    Fit/CV event/family RMS by magnitude range
tables/fit_objective_prediction_gains.csv Paired held-out-family gain intervals
tables/fit_objective_bootstrap.npz      Whole-family coefficient draws
tables/fit_objective_validation.json    Earlier-method and native-CV reproduction
tables/ML_measurement_audit.csv          Event ML median/grade amplitude replay
tables/ML_station_measurements.csv      Saved N/E amplitudes, SNR, distance, ML
tables/regression_comparison_models.csv 32 cases with full coefficients/intervals
tables/regression_comparison_curves.csv Curves, bands and extrapolation flags
tables/regression_comparison_predictions.csv Same-target family holdouts
tables/regression_comparison_performance.csv Low/high magnitude prediction RMS
tables/regression_comparison_bootstrap.npz Joint centered coefficient draws
tables/regression_comparison_validation.json Amplitude and historical-fit audit
tables/Mw_ML_catalog.csv                All 3,532 attempts, failures included
tables/NW2_Mw_ML_catalog.csv             Detailed uniform-method output
tables/NW4_Mw_ML_catalog.csv             Original-method control output
tables/paired_A_MLge1p5.csv              Primary regression set
tables/small_paired_A.csv                Every accepted smaller paired-A case
tables/regressions.csv                  All model/weighting comparisons
tables/regression_bands.csv             Family-bootstrap curve bands
tables/family_heldout_predictions.csv    Complete family-held-out predictions
tables/model_prediction_comparison.csv  Paired held-out error comparisons
tables/reference_overlap_reassessment.csv New overlap screen versus old values
tables/coverage.csv                     Depth/era/measured-ML coverage
tables/failure_reasons.csv              Multiple gate failures can apply
tables/small_event_HH_HG_pairs.csv       Jointly usable co-located sensor pairs
tables/validation.json                  Scientific/computational audit
tables/report_validation.json           PDF counts, bounds, hashes and sources

CSV event times are UTC; event_id is a string. ml is KMA ML; ML_Sheen is
independently measured ML. Depths and their inherited bounds are in km; they
are not formal single-model standard errors. Paired-A requires both A grades;
paired_A_or_B requires both A or B. Blank Mw/ML values remain blank. Conditional
station bootstrap bounds exclude absolute response, velocity, attenuation,
radiation and site errors; they are not total 95% uncertainty intervals.

REPRODUCTION FROM THE PROJECT ROOT
Use the single-thread/cache environment in the root README.
The prediction table exceeds GitHub's per-file limit and is committed as a
lossless .csv.gz. On a fresh clone, first run:
  gzip -dk small_events/extension/tables/magnitude_scaling_predictions.csv.gz
The canonical .csv is kept on Moho; decompression restores its original bytes
and the SHA-256 recorded in the manifest and analysis validation records.
1. python code/prepare_small_event_extension.py
2. python code/run_small_event_extension.py --workers 16
3. python code/analyze_small_event_extension.py --workers 8
4. python code/compare_extension_regressions.py
5. python code/audit_extension_fit_objectives.py
6. python code/analyze_extension_potency.py
7. python code/extension_potency_controls.py
8. python code/report_small_event_extension.py
9. python code/validate_extension_potency.py
10. python code/validate_small_event_extension.py
11. python code/write_manifests.py --project-only
For a report-only revision, --reuse-atlas verifies and reuses the existing
spectral atlas. It changes no magnitude or previously selected spectrum.
Preparation preserves an existing frozen selection. Raw waveforms are read
from the same server projects; per-event caches/logs are local and ignored
by git. No other project or adopted catalog/report is modified.
