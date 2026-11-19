# References

The paper PDFs used in this study are kept on the analysis server in this folder but are not in the
repository; they keep their publishers' copyright. Place them here under the file names below to run
the scripts that read them (`audit_potency_literature.py` reads the two Liu files; the
source-velocity check `validate_source_velocity_update.py` verifies the checksum of the Hong file).

| File name | Paper | Used for |
|---|---|---|
| `Hong_Sheen_2026_BSSA_Moment_magnitude.pdf` | Hong, Y., and D.-H. Sheen (2026). *Bulletin of the Seismological Society of America*. [doi:10.1785/0120250293](https://doi.org/10.1785/0120250293) | Method comparison with the KR_Mw catalog (`reports/Hong_KR_Mw_regression_and_stability.pdf`) |
| `Liu2026_GRL.pdf` | Liu, Z., H. Meng, Y. Hou, D. T. Trugman, and Y. Ben-Zion (2026). A unified quadratic scaling relation between potency and local earthquake magnitudes for −1.8 ≤ ML ≤ 7.5. *Geophysical Research Letters*. [doi:10.1029/2026GL124983](https://doi.org/10.1029/2026GL124983) | Potency magnitude MP and quadratic ML scaling |
| `Liu2026_GRK_SM.pdf` | Supporting information of Liu et al. (2026), same DOI | Binning procedure; depth dependence (Figure S7) |
| `Trugman2026_SRL.pdf` | Trugman, D. T., and Y. Ben-Zion (2026). Correcting biases from velocity structures in moment magnitude estimates. *Seismological Research Letters*. [doi:10.1785/0220250419](https://doi.org/10.1785/0220250419) | Potency normalization (28 GPa reference) |

Method sources cited in the code and reports:

- Kim & Kim (1983) crustal model as implemented in the Haenam project: Vp = 5.98, 6.38 and 7.95 km/s
  for 0–15, 15–32 and > 32 km, Vp/Vs = 1.73 (definition and provenance in `inputs/source_velocity_models.json`).
- Kim, S., J. Rhie, and G. Kim (2011). *Geophysical Journal International* 185, 453,
  [doi:10.1111/j.1365-246X.2011.04949.x](https://doi.org/10.1111/j.1365-246X.2011.04949.x), Table 2 (comparison velocity model).
- Sheen, D.-H., T.-S. Kang, and J. Rhie (2018). A local magnitude scale for South Korea. *Bulletin of the
  Seismological Society of America* 108(5A), 2748–2755, [doi:10.1785/0120180112](https://doi.org/10.1785/0120180112) (ML scale).
- Prieto, G. A. (2022). The multitaper spectrum analysis package in Python. *Seismological Research Letters*
  93(3), 1922 (multitaper spectra, package `multitaper`).
- Ahn, B.-S., T.-S. Kang, and J. O. Jung (2024). Station metadata integration of regional seismic networks in
  the southern Korean Peninsula. *Journal of the Geological Society of Korea* 60(1); data at
  [github.com/ByeongSeokAhn/StationMetadata](https://github.com/ByeongSeokAhn/StationMetadata), v1.0.2 (historical instrument responses).

Page-referenced reading notes on these papers, and the further literature on the depth dependence of
magnitude differences, are in `tables/Hong_paper_method_audit.json`, `tables/potency_paper_method_audit.json`
and `tables/potency_rigidity_literature_audit.json`.
