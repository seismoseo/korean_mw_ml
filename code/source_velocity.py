"""Explicit source-layer velocities and magnitude normalization.

The measured, distance/Q-corrected spectrum is independent of this source
coefficient. Both models use a controlled common density of 2800 kg/m3.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'inputs/source_velocity_models.json'
CONFIG = json.loads(MODEL_PATH.read_text())
CONFIG_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
PRIMARY_MODEL = CONFIG['primary_model']
REFERENCE_VS = float(CONFIG['reference_Vs_m_s'])
RHO = float(CONFIG['density_kg_m3'])
C = 9.05
RAD = .63
FS = 2.


def source_vs(depth_km, model=PRIMARY_MODEL):
    definition = CONFIG['models'][model]
    depth = np.asarray(depth_km, dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth < 0):
        raise ValueError('Source depth must be finite and nonnegative')
    layer = np.searchsorted(definition['layer_tops_km'], depth, side='right') - 1
    value = np.asarray(definition['Vs_km_s'])[layer] * 1000.
    return float(value) if value.ndim == 0 else value


def magnitude_shift(depth_km, model=PRIMARY_MODEL, reference_vs=REFERENCE_VS):
    return 2 * np.log10(source_vs(depth_km, model) / reference_vs)


def log_moment_coefficient(depth_km, model=PRIMARY_MODEL):
    return np.log10(4 * np.pi * RHO * source_vs(depth_km, model)**3 / (FS * RAD))


def metadata(depth_km):
    row = dict(source_velocity_model=PRIMARY_MODEL,
               source_velocity_implementation=CONFIG['models'][PRIMARY_MODEL]['implementation'],
               source_velocity_config_sha256=CONFIG_SHA256,
               source_depth_km=float(depth_km), source_density_kg_m3=RHO,
               source_Vs_m_s=source_vs(depth_km))
    for model in CONFIG['models']:
        row['source_Vs_' + model] = source_vs(depth_km, model)
    return row


def normalize_fit(result, depth_km):
    """Normalize from the fitted physical plateau, so repeated calls are safe."""
    row = dict(result)
    row.update(metadata(depth_km))
    omega = row.get('omega_r_m2s', np.nan)
    if not np.isfinite(omega) or omega <= 0:
        return row
    log_omega = np.log10(omega)
    ref_log_m0 = log_omega + np.log10(4*np.pi*RHO*REFERENCE_VS**3/(FS*RAD))
    row['Mw_reference_Vs3500'] = float((ref_log_m0-C)/1.5)
    row['M0_reference_Vs3500_Nm'] = float(10**ref_log_m0)
    for model in CONFIG['models']:
        log_m0 = log_omega + log_moment_coefficient(depth_km, model)
        row['Mw_source_Vs_' + model] = float((log_m0-C)/1.5)
        row['M0_' + model + '_Nm'] = float(10**log_m0)
    row['Mw'] = row['Mw_source_Vs_' + PRIMARY_MODEL]
    row['M0_Nm'] = row['M0_' + PRIMARY_MODEL + '_Nm']
    row['Mw_C9p1'] = row['Mw'] - (9.1-C)/1.5
    return row


def validate_catalog(frame):
    q = frame[frame.Mw.notna()]
    assert q.source_velocity_model.eq(PRIMARY_MODEL).all()
    assert np.allclose(q.source_Vs_m_s, source_vs(q.z_central.to_numpy()))
    assert np.allclose(q.Mw, q.Mw_reference_Vs3500 + magnitude_shift(q.z_central.to_numpy()), atol=1e-11)
    for model in CONFIG['models']:
        assert np.allclose(q['Mw_source_Vs_'+model], q.Mw_reference_Vs3500 + magnitude_shift(q.z_central.to_numpy(), model), atol=1e-11)
    assert np.allclose(q.Mw, (np.log10(q.M0_Nm)-C)/1.5, atol=1e-11)
    return True
