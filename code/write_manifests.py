"""Release index (tables/release_index.json) and content manifests of the national and Sangju trees.

Run last. The index collects the pass/fail flags of every validation step and the page count and
hash of every report; MANIFEST.csv lists size and SHA-256 of every local file, including the
per-event results and raw inputs that are not in the git repository."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv,hashlib,json
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--project-only',action='store_true',help='Update this project only; leave the Sangju manifest unchanged')
args=parser.parse_args()

# Keep the top-level validation index tied to the current release, rather than
# leaving numerical counts and PDF hashes from an older processing generation.
release=R/'tables/release_validation.json'
if release.exists():
    import fitz
    numerical=json.loads(release.read_text())
    spectral=json.loads((R/'tables/study_summary.json').read_text())
    local=json.loads((R/'tables/independent_ML_summary.json').read_text())
    reports={}
    for p in sorted((R/'reports').glob('*.pdf')):
        with fitz.open(p) as doc:pages=len(doc)
        reports[p.name]=dict(pages=pages,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    index=dict(passed=bool(numerical['passed']),adopted_estimator='Multitaper / Boatwright',
      candidate_events=spectral['candidate_events'],Mw_quality_counts=spectral['Mw_quality_counts'],
      primary_events=spectral['primary_regression_events'],primary_blocks=spectral['primary_sequence_blocks'],
      paired_primary_events=local['n_primary'],paired_primary_blocks=local['n_blocks'],
      numerical_validation='tables/release_validation.json',reports=reports,
      external_adopted_projects_modified=False)
    comparison=R/'tables/comparison_validation.json'
    if comparison.exists():
        check=json.loads(comparison.read_text())
        for filename,digest in check.get('unchanged_input_sha256',{}).items():
            assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest,filename
        index.update(comparison_validation='tables/comparison_validation.json',
          comparison_passed=check['passed'],primary_values_changed_by_comparison=False)
    velocity=R/'tables/source_velocity_update_validation.json'
    if velocity.exists():
        check=json.loads(velocity.read_text());assert check['passed']
        index.update(source_velocity_model='Kim & Kim (1983), Vp/Vs=1.73',
          source_velocity_comparison='Kim et al. (2011), published Table 2',
          source_velocity_validation='tables/source_velocity_update_validation.json',
          source_velocity_config_sha256=check['source_velocity_config_sha256'],
          source_velocity_updated=True)
    hong=R/'tables/Hong_report_validation.json'
    if hong.exists():
        check=json.loads(hong.read_text());assert check['passed']
        index.update(Hong_regression_validation='tables/Hong_report_validation.json',
          Hong_regression_passed=check['ODR_validation']['passed'],
          estimator_stability_common_plateau_events=check['n_stability_common_plateau'])
    potency=R/'tables/potency_validation.json'
    if potency.exists():
        check=json.loads(potency.read_text());assert check['passed']
        index.update(potency_analysis_validation='tables/potency_validation.json',potency_analysis_passed=True,
          potency_analysis_scope='Source-only scalar potency and exploratory matched Mp/Mw regressions, large-event prediction, normalization and depth-residual controls. Adopted Mw/ML unchanged; Mpv not estimated.',
          potency_paper_method_audit='tables/potency_paper_method_audit.json')
    followup=R/'tables/magnitude_followup_validation.json'
    if followup.exists():
        check=json.loads(followup.read_text());assert check['passed']
        index.update(magnitude_followup_validation='tables/magnitude_followup_validation.json',magnitude_followup_passed=True)
    literature=R/'tables/potency_rigidity_literature_audit.json'
    if literature.exists():
        check=json.loads(literature.read_text());assert check['passed']
        index.update(potency_rigidity_literature_audit='tables/potency_rigidity_literature_audit.json',
          potency_rigidity_literature_passed=True)
    snoke=R/'tables/snoke_audit_validation.json'
    if snoke.exists():
        check=json.loads(snoke.read_text());assert check['passed']
        index.update(snoke_audit_validation='tables/snoke_audit_validation.json',snoke_audit_passed=True,
          Snoke_coefficient_correction_applied=False,
          snoke_audit_scope='Equation and numerical audit with diagnostic integral, bandwidth and source-shape controls; adopted magnitudes and regressions preserved.')
    pilot=R/'small_events/tables/validation.json'
    if pilot.exists():
        check=json.loads(pilot.read_text());assert check['passed']
        for filename,digest in check['protected_results_sha256'].items():
            assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest,filename
        pilot_reports={}
        for filename,key,pages in [('Small_event_Mw_ML_pilot.pdf','report_sha256',7),
                                   ('Small_event_spectral_atlas.pdf','atlas_sha256',45)]:
            path=R/'small_events/reports'/filename
            digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==check[key]
            with fitz.open(path) as doc:assert len(doc)==pages
            pilot_reports[str(path.relative_to(R))]=dict(pages=pages,sha256=digest)
        index.update(small_event_validation='small_events/tables/validation.json',small_event_pilot_passed=True,
            small_event_scope=check['scope'],small_event_adopted_calibration_changed=False,
            small_event_pilot_attempted=check['small_attempted'],small_event_controls=check['control_attempted'],
            small_event_reports=pilot_reports)
    extension=R/'small_events/extension/tables/validation.json'
    if extension.exists():
        check=json.loads(extension.read_text());assert check['passed']
        summary=json.loads((R/'small_events/extension/tables/summary.json').read_text())
        assert summary['complete'] and summary['attempted']==3532
        for filename,digest in check['protected_results_sha256'].items():
            assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest,filename
        for filename,record in check['reports'].items():
            path=R/filename;assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            with fitz.open(path) as doc:assert len(doc)==record['pages']
        comparison_path=R/check['comparison_validation']
        assert hashlib.sha256(comparison_path.read_bytes()).hexdigest()==check['comparison_validation_sha256']
        detail=json.loads(comparison_path.read_text());assert detail['passed']
        objective_path=R/check['fitting_objective_validation']
        assert hashlib.sha256(objective_path.read_bytes()).hexdigest()==check['fitting_objective_validation_sha256']
        objective=json.loads(objective_path.read_text());assert objective['passed']
        potency_path=R/check['extension_potency_validation']
        assert hashlib.sha256(potency_path.read_bytes()).hexdigest()==check['extension_potency_validation_sha256']
        potency=json.loads(potency_path.read_text());assert potency['passed']
        index.update(small_event_extension_validation='small_events/extension/tables/validation.json',
            small_event_extension_passed=True,small_event_extension_transfer_passed=summary['transfer']['passed'],
            small_event_extension_attempted=3532,small_event_extension_small_events=2890,
            small_event_extension_primary_events=summary['primary_events'],small_event_extension_primary_blocks=summary['primary_blocks'],
            small_event_extension_estimator=summary['uniform_estimator'],small_event_extension_reports=check['reports'],
            small_event_extension_regression_comparison=check['comparison_validation'],
            small_event_extension_independent_ML=detail['ML_counts'],
            small_event_extension_fitting_objectives=check['fitting_objective_validation'],
            small_event_extension_fitting_objective_models=objective['n_models'],
            small_event_extension_potency_validation=check['extension_potency_validation'],
            small_event_extension_finite_Mp=potency['finite_potency'],
            small_event_extension_joint_magnitude_cases=potency['regression_cases'],
            small_event_extension_previous_joint_cases_reproduced=potency['previous_native_cases_reproduced'],
            legacy_national_results_preserved_by_extension=True)
    (R/'tables/release_index.json').write_text(json.dumps(index,indent=2)+'\n')

def record(item):
    root,p=item;h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return p.relative_to(root).as_posix(),p.stat().st_size,h.hexdigest()

targets=[(R,R/'tables/MANIFEST.csv')]
if not args.project_only:
    targets.append((R.parent/'sangju_20261117',R.parent/'sangju_20261117/results/MANIFEST.csv'))
for root,manifest in targets:
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p!=manifest and
      not any(s in ['.git','.claude','__pycache__','.ipynb_checkpoints'] for s in p.relative_to(root).parts))
    with ThreadPoolExecutor(12) as pool:rows=list(pool.map(record,((root,p) for p in paths)))
    with manifest.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['path','bytes','sha256']);writer.writerows(rows)
    print(root.name,len(rows),'files',round(sum(x[1] for x in rows)/1048576,1),'MiB',flush=True)
