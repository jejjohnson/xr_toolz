# Ocean Data Challenges → xr_toolz Integration Survey

Survey of three `ocean-data-challenges` repos (all MIT-style) to identify
features worth integrating into `xr_toolz`. Repo 1 & 3 are forks of a common
eval base; repo 2 is the leaner eNATL60/NATL60 OSSE subset.

- Repo 1: <https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE>
- Repo 2: <https://github.com/ocean-data-challenges/2023_SSH_mapping_train_eNATL60_test_NATL60->
- Repo 3: <https://github.com/ocean-data-challenges/2024c_DC_4DMedSea-ESA>

This document is a working catalogue: we walk through it item-by-item to decide
what to port, adapt, or skip. Status column is updated as decisions are made.

---

## 1. `2024_DC_SSH_mapping_SWOT_OSE`

| # | File | What it provides | Status |
|---|------|------------------|--------|
| 1.1 | [`mod_filter.py`](https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE/blob/main/src/mod_filter.py) | `lanczos_filter` (FFT 1D low-pass), `apply_bandpass_filter(ds, λmin, λmax)` for along-track SLA, `compute_median_dx` (haversine spacing). Lanczos + bandpass not in xr_toolz. | Proposal: [odc-1.1-fir-filters.md](odc-1.1-fir-filters.md) |
| 1.2 | [`mod_interp.py`](https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE/blob/main/src/mod_interp.py) | `run_interpolation(ds_maps, ds_alongtrack)` colocates a gridded map onto along-track obs via pyinterp 4D, plus drifter and u/v variants. **Most-reused function across all three repos.** | Proposal: [odc-1.2-grid-to-points.md](odc-1.2-grid-to-points.md) |
| 1.3 | [`mod_spectral.py`](https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE/blob/main/src/mod_spectral.py) | Segment-based 1D PSD: `compute_segment` (gap-tolerant equal-length segmentation w/ overlap), `spectral_computation` (per-segment FFT + cross-spec + coherence), `compute_resolution` (canonical λx-at-PSD-score=0.5), `compute_psd_scores` driver. | Proposal: [odc-1.3-segmented-psd.md](odc-1.3-segmented-psd.md) |
| 1.4 | [`mod_stat.py`](https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE/blob/main/src/mod_stat.py) | `bin_data` (2D lat/lon binning of along-track residuals), `compute_stat_scores_by_regimes` (coastal/equatorial/high-vs-low-eddy), `dm_test` (Diebold–Mariano significance). | Proposal: [odc-1.4-binned-stats-regions.md](odc-1.4-binned-stats-regions.md) |
| 1.5 | [`mod_plot.py`](https://github.com/ocean-data-challenges/2024_DC_SSH_mapping_SWOT_OSE/blob/main/src/mod_plot.py) | Mostly redundant with our validation viz; net-new are regime-bar plots and `plot_polarization`. | Proposal: [odc-1.5-regime-bars-rotary.md](odc-1.5-regime-bars-rotary.md) |

---

## 2. `2023_SSH_mapping_train_eNATL60_test_NATL60-`

| # | File | What it provides | Status |
|---|------|------------------|--------|
| 2.1 | [`src/mod_eval.py`](https://github.com/ocean-data-challenges/2023_SSH_mapping_train_eNATL60_test_NATL60-/blob/main/src/mod_eval.py) | `rmse_based_scores` returns `(rmse_t, rmse_xy, leaderboard, error_stability=std(rmse_t))`; `psd_based_scores` does **2D (kx, kt) PSD-score + 0.5-contour extraction of (λx, λt)**. xr_toolz only has the 1D isotropic version. | Proposal: [odc-2.1-rmse-and-psd-spacetime.md](odc-2.1-rmse-and-psd-spacetime.md) |
| 2.2 | [`src/mod_regrid.py`](https://github.com/ocean-data-challenges/2023_SSH_mapping_train_eNATL60_test_NATL60-/blob/main/src/mod_regrid.py) | pyinterp 3D regrid + Gauss–Seidel NaN extrapolation. Clean alternative to current sklearn-NN regridder. | Proposal: [odc-2.2-laplacian-gap-fill.md](odc-2.2-laplacian-gap-fill.md) |
| 2.3 | `src/mod_plot.py` | Seasonal PSD-score mosaic panel — only viz not already covered. | Proposal: [odc-2.3-facet-panel.md](odc-2.3-facet-panel.md) |

---

## 3. `2024c_DC_4DMedSea-ESA`

(Inherits repo 1's filter/interp/spectral/stat; only net-new modules listed.)

| # | File | What it provides | Status |
|---|------|------------------|--------|
| 3.1 | [`src/mod_powerspec.py`](https://github.com/ocean-data-challenges/2024c_DC_4DMedSea-ESA/blob/main/src/mod_powerspec.py) | `wavenumber_spectra` (radial 2D PSD with Tukey/Hanning windows), `cross_spectra`, **`spectra_flux` (KE spectral flux from u, v in Fourier space)**, `weighted_scale` (integral scale), `fill_nan`. Spectral KE flux is genuinely missing from xr_toolz. | Proposal: [odc-3.1-spectral-flux-and-scales.md](odc-3.1-spectral-flux-and-scales.md) |
| 3.2 | [`src/mod_switchvar.py`](https://github.com/ocean-data-challenges/2024c_DC_4DMedSea-ESA/blob/main/src/mod_switchvar.py) | `sla_to_ssh(ds, mdt)` (adds CNES MDT), `currents_to_potential_vorticity(u, v, h)`. The rest duplicates our `ocn` ops. | TBD |
| 3.3 | [`src/mod_traj.py`](https://github.com/ocean-data-challenges/2024c_DC_4DMedSea-ESA/blob/main/src/mod_traj.py) | Lagrangian drifter advection: `adv_eul`, `adv_rk4`, `compute_traj` (multi-horizon), `compute_deviation` (binned RMSE between modeled vs observed drifters), `dist_drifters`. **Entire Lagrangian sub-domain missing from xr_toolz.** | Reconciles with [Epic V3 (#49)](https://github.com/jejjohnson/xr_toolz/issues/49) — see [odc-3.2-lagrangian-reconciliation.md](odc-3.2-lagrangian-reconciliation.md). Upstream `adv_eul`/`adv_rk4` → [#51](https://github.com/jejjohnson/xr_toolz/issues/51); `compute_deviation` → [#53](https://github.com/jejjohnson/xr_toolz/issues/53) (new `EndpointErrorMap`); `prepare_drifter_data` → [#54](https://github.com/jejjohnson/xr_toolz/issues/54). |
| 3.4 | `src/mod_compare.py` | Paired-diff stat/PSD plots (the wrappers, not the zoom/longitude utils). | TBD |
| 3.5 | `src/mod_xscale.py` / `mod_utils.py` / `mod_read.py` | Duplicate `xrft`, generic helpers, data-fetch. | Skip |
| 3.6 | `mod_plot.py:movie` | `matplotlib.animation` intercomparison panels. | TBD |

---

## Cross-cutting consensus workflow

The recurring pipeline across all three repos:

```
read
  → bandpass-filter reference along-track
  → colocate gridded map onto along-track via pyinterp
  → bin residuals
  → score (RMSE / segmented-PSD λx / 2D PSD λx,λt / drifter deviation)
  → plot maps + PSD curves + leaderboard
  → paired-diff + DM-test
```

Every notebook in all three repos re-implements this by hand. The single
biggest win is lifting this chain into a `xr_toolz.geo.metrics.ssh_mapping`
mini-pipeline + a `Sequential` recipe.

---

## Top integration candidates (value × ease)

| Rank | Item | Value | Ease | Maps to |
|------|------|-------|------|---------|
| 1 | `AlongTrackColocate` / `DrifterColocate` operator wrapping pyinterp 4D | HIGH | MEDIUM | `geo/inference` |
| 2 | `SegmentedPSDScore` + `effective_resolution(k, psd_diff, psd_ref, t=0.5)` | HIGH | MEDIUM | `geo/metrics` |
| 3 | 2D `(kx, kt)` PSD-score with `(λx, λt)` double-contour + `PSDScore2DPanel` | HIGH | EASY | `geo/metrics` + `viz/validation` |
| 4 | Spectral KE flux `spectra_flux(u, v, lon, lat)` | HIGH | MEDIUM | `ocn` |
| 5 | Lagrangian drifter advection + deviation skill (RK4 + binned-RMSE) | HIGH | HIGH (~600 LOC) | new `xr_toolz.ocn.lagrangian` |
| 6 | Lanczos / bandpass filter primitives for 1D along-track | MEDIUM | EASY | `geo` |
| 7 | Diebold–Mariano paired-forecast test | MEDIUM | EASY | `geo/metrics` |
| 8 | `error_stability = std(rmse_t)` | MEDIUM | TRIVIAL | `geo/metrics` |
| 9 | Regime-stratified scoring (coastal/equatorial/eddy) + bar plot | MEDIUM | MEDIUM | `geo/metrics` + viz |
| 10 | pyinterp + Gauss–Seidel NaN-fill regridder alongside sklearn-NN | MEDIUM | EASY | `geo/regrid` |
| 11 | `sla_to_ssh(ds, mdt)` MDT-addition op + `ssh→PV` | MEDIUM | TRIVIAL | `ocn` |
| 12 | Seasonal PSD-score mosaic viz | LOW-MEDIUM | EASY | `viz/validation` |
| 13 | Movie / animation comparison panel | LOW | MEDIUM | `viz/validation` |

---

## Skip list

- `mod_xscale.py` — duplicates `xrft` (already a dependency).
- `mod_utils.py` — generic helpers redundant with xarray.
- `regional_zoom` / `convert_longitude` — already in `geo/subset`.
- `ssh_to_currents` / `currents_to_relative_vorticity` — already in `ocn`.
- `mod_read.py` data-fetch — belongs outside the library.
- Bulk of `mod_plot.py` — already covered by `_ValidationPanel` /
  `SpatialMapPanel` / `PSDIsotropicScorePanel`.
- `median_filter` — `scipy.signal.medfilt` is preferable.

---

## Decision log

Use this section as we walk through items 1.1 → 3.6 to record decisions
(port / adapt / skip), proposed APIs, and follow-up issue links.

| Item | Decision | Notes / Issue |
|------|----------|---------------|
| _e.g._ 1.2 | Adapt as `geo.inference.AlongTrackColocate` | TBD |
