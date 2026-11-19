"""Release index (tables/release_index.json) and content manifests of the national and Sangju trees.

Run last. The index collects the pass/fail flags of every validation step and the page count and
hash of every report; MANIFEST.csv lists size and SHA-256 of every local file, including the
per-event results and raw inputs that are not in the git repository."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json
R=Path(__file__).resolve().parents[1]

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
    (R/'tables/release_index.json').write_text(json.dumps(index,indent=2)+'\n')

def record(item):
    root,p=item;h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return p.relative_to(root).as_posix(),p.stat().st_size,h.hexdigest()

for root,manifest in [(R,R/'tables/MANIFEST.csv'),
  (R.parent/'sangju_20261117',R.parent/'sangju_20261117/results/MANIFEST.csv')]:
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p!=manifest and
      not any(s in ['.git','.claude','__pycache__','.ipynb_checkpoints'] for s in p.relative_to(root).parts))
    with ThreadPoolExecutor(12) as pool:rows=list(pool.map(record,((root,p) for p in paths)))
    with manifest.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['path','bytes','sha256']);writer.writerows(rows)
    print(root.name,len(rows),'files',round(sum(x[1] for x in rows)/1048576,1),'MiB',flush=True)
