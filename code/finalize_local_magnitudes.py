"""Summarize current HH/EL/HG amplitude caches; no historical overrides."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import local_magnitude as L
R=L.ROOT
rows=[];stations=[];audit=[];sigs=set()
kma=pd.read_csv(R/'inputs/cohort.csv',dtype={'event_id':str}).set_index('event_id').ml.to_dict()
kma['20261117141017']=3.6
for eid in kma:
    obj=json.loads((R/'ml_events'/(eid+'.json')).read_text())
    sigs.add(obj['config_sha256']);s=dict(obj['summary']);chosen=obj['selected']
    if len(chosen)>=3:
        v=np.array([c['value'] for c in chosen]);rng=np.random.default_rng(int(eid[-8:]))
        samples=np.median(rng.choice(v,(2000,len(v)),replace=True),axis=1)
        s['ML_station_bootstrap_lower'],s['ML_station_bootstrap_upper']=map(float,np.quantile(samples,[.025,.975]))
    rows.append(dict(event_id=eid,ML_KMA=kma[eid],**s))
    stations.extend(dict(event_id=eid,**c) for c in chosen)
    audit.extend(dict(event_id=eid,network=c['network'],station=c['station'],sensor=c['sensor'],
      component=c['component'],status=c['status'],reason=c.get('reason','')) for c in obj['components'])
assert len(sigs)==1,sigs
pd.DataFrame(rows).to_csv(R/'tables/independent_ML_catalog.csv',index=False)
pd.DataFrame(stations).to_csv(R/'tables/independent_ML_stations.csv',index=False)
pd.DataFrame(audit).to_csv(R/'ml_validation/component_status.csv.gz',index=False)
summary=dict(n_events=len(rows),signature=next(iter(sigs)),P_timing='Applied before amplitude measurement',
  primary_sensors=['HH','EL','HG'],one_observation_per_site=True,
  bootstrap='2000 station resamples; conditional on accepted stations, response, scale and windows')
(R/'ml_validation/postprocessing_provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
