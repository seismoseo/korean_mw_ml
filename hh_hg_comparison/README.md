# HH and HG records at the same stations

Checks whether accelerometer (HG) records can stand beside broadband (HH) records in the spectral Mw and
ML of the main study. 14 station–event pairs are compared: the Sangju earthquake of 17 November 2026 at
six stations, and two Haenam earthquakes (22 and 25 August 2026, ML 3.1 and 1.3) at four stations each.
Both sensors share the station, the sample times, the analysis windows and the physical units. No
amplitude factor is fitted between them.

**Report:** [`reports/HH_HG_paired_comparison.pdf`](reports/HH_HG_paired_comparison.pdf), 8 pages. It is
also the appendix of the main report, [`../reports/Korean_Mw_ML_study.pdf`](../reports/Korean_Mw_ML_study.pdf).

## Results

- **Haenam project.** It already uses HG records: 2,845 horizontal HG station measurements over 274 events
  in its ML table. Its accepted spectral fit of 25 August 2026 uses 40 HG, 16 HH and 3 EL stations
  (`tables/Haenam_sensor_use_summary.csv`, `tables/Haenam_spectral_sensor_example.csv`).
- **Sangju, 0.7–1.2 Hz.** At five stations the median HG/HH amplitude ratio is 1.01–1.09, with coherence
  0.94–1.00. KS.BURB gives 0.79 with coherence 0.58 and is left out of the Sangju magnitudes
  (`tables/spectral_comparison.csv`).
- **Small events.** HG can give a usable ML peak for a small event. For the ML 1.3 Haenam event at JAEB,
  the Wood–Anderson peak SNR is 3.37 on HG and 2.55 on HH (`tables/paired_ML.csv`). This is one station.
- **Low frequencies.** Neither sensor resolves the moment plateau of every small event. At 0.05–0.2 Hz the
  Sangju HG signal-to-noise ratio is only 0.81–2.20. This limits waveform inversion at long periods; it
  does not affect ML or spectral Mw at higher frequencies.
- **Water level.** A 60 dB water level in response removal to displacement attenuates HG differently from
  HH (`tables/waterlevel_transfer.csv`). The main study therefore removes responses without a water level.

## Run order

From the project folder, after `download_sangju.py` has fetched the Sangju event package from NECIS
(once; needs NECIS credentials and [necis-downloader](https://github.com/seismoseo/necis-downloader)):

    python hh_hg_comparison/code/prepare_pairs.py            # Haenam pairs and sensor-use tables
    python hh_hg_comparison/code/prepare_pairs.py --sangju   # Sangju pairs
    python hh_hg_comparison/code/compare_pairs.py
    python hh_hg_comparison/code/waterlevel_transfer.py
    python hh_hg_comparison/code/plot_comparison.py
    python hh_hg_comparison/code/build_report.py
    python hh_hg_comparison/code/validate_comparison.py

`inputs/pairs.json` lists the pairs, windows, picks and source files, and `inputs/source_hashes.json`
gives their checksums. The record cut-outs (`inputs/cutouts/`), response snapshots (`inputs/responses/`),
downloaded event packages and spectra arrays (`spectra/`) stay on the server and are not in the repository.
The Haenam records are read from the Haenam project and are not changed.
