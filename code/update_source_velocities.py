"""Migrate cached fit normalization; verify measured spectra and ML are immutable.

No waveform processing, selection, depth relocation, or spectral shape changes.
The old 3500 m/s coefficient remains an explicitly named numerical control.
"""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import pandas as pd
from source_velocity import CONFIG,CONFIG_SHA256,PRIMARY_MODEL,normalize_fit

R=Path(__file__).resolve().parents[1]

def main():
    path=R/'tables/source_velocity_update.json'
    audit=json.loads(path.read_text())
    for name,digest in audit['immutable_sha256'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
    config=R/'inputs/study_config.json';c=json.loads(config.read_text())
    c.pop('Vs_m_s',None)
    c.update(reference_Vs_m_s=3500.,source_velocity_model=PRIMARY_MODEL,
             source_velocity_comparison='Kim2011',source_velocity_config='inputs/source_velocity_models.json',
             source_velocity_config_sha256=CONFIG_SHA256,
             source_depth_policy=CONFIG['depth_policy'],source_velocity_scope=CONFIG['scope'])
    config.write_text(json.dumps(c,indent=2)+'\n')
    import run_catalog
    signature=run_catalog.config_signature()
    cohort=pd.read_csv(R/'inputs/cohort.csv',dtype={'event_id':str})
    depths=dict(zip(cohort.event_id,cohort.z_central));depths['20261117141017']=19.76
    rows=[];status=[];nfits=0
    for eid,depth in depths.items():
        p=R/'events'/(eid+'.json');obj=json.loads(p.read_text())
        previous=obj['config_sha256']
        obj['fits']=[normalize_fit(x,depth) for x in obj.get('fits',[])]
        obj['config_sha256']=signature
        obj['source_velocity_config_sha256']=CONFIG_SHA256
        obj['source_depth_km']=float(depth)
        obj.setdefault('normalization_migration',dict(previous_config_sha256=previous,
            note='Measured spectral arrays, station records, response choices and picks were preserved; cached fits recomputed from their stored physical plateau.'))
        p.write_text(json.dumps(obj,indent=2)+'\n')
        rows+=obj['fits'];nfits+=len(obj['fits'])
        status.append({k:v for k,v in obj.items() if k not in ['fits','source_sha256','neighbors']})
    pd.DataFrame(rows).to_csv(R/'tables/catalog_spectral_fits.csv',index=False)
    pd.DataFrame(status).to_csv(R/'tables/catalog_processing_status.csv',index=False)
    audit.update(status='normalized',primary_model=PRIMARY_MODEL,comparison_model='Kim2011',
                 source_velocity_config_sha256=CONFIG_SHA256,new_processing_signature=signature,
                 n_cache_events=len(depths),n_normalized_fits=nfits)
    path.write_text(json.dumps(audit,indent=2)+'\n')
    print('Source-normalized',len(depths),'events and',nfits,'fits; immutable measurements verified')

if __name__=='__main__':main()
