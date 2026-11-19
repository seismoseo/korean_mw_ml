"""Read the explicit model definitions retained with the current Mw release."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'inputs/source_velocity_models.json'
CONFIG=json.loads(MODEL_PATH.read_text())
CONFIG_SHA256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def source_vs(depth_km,model='Kim1983'):
    definition=CONFIG['models'][model]
    depth=np.asarray(depth_km,dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth<0):raise ValueError('Invalid source depth')
    layer=np.searchsorted(definition['layer_tops_km'],depth,side='right')-1
    value=np.asarray(definition['Vs_km_s'])[layer]*1000
    return float(value) if value.ndim==0 else value
