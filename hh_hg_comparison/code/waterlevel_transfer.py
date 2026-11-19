"""Exact inverse-response attenuation imposed by a 60 dB water level."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from obspy import read,read_inventory,UTCDateTime
from obspy.signal.util import _npts2nfft
ROOT=Path(__file__).resolve().parents[1]
pairs=json.loads((ROOT/'inputs/pairs.json').read_text());rows=[]
for pair in pairs:
    for key in ['HHZ','HGZ']:
        tr=read(pair['traces'][key])[0]
        resp=read_inventory(pair['responses'][key]).get_response(tr.id,UTCDateTime(pair['origin_utc']))
        response,f=resp.get_evalresp_response(t_samp=tr.stats.delta,nfft=_npts2nfft(tr.stats.npts),output='DISP',hide_sensitivity_mismatch_warning=True)
        magnitude=abs(response);floor=magnitude.max()*1.e-3
        grid=np.geomspace(.03,25.,350);gain=np.minimum(1,np.interp(grid,f,magnitude)/floor)
        rows.extend(dict(event_id=pair['event_id'],name=pair['name'],station=pair['station'],sensor=key[:2],frequency_Hz=float(fr),WL60_to_noWL=float(g)) for fr,g in zip(grid,gain))
pd.DataFrame(rows).to_csv(ROOT/'tables/waterlevel_transfer.csv',index=False)
print('Saved exact response-domain water-level transfer curves')
