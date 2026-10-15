
# InSitu-HUXt paper reproduction package

This repository contains the code and processed data for the figures and
statistics in Owens et al., *Forecasting near-Earth solar wind conditions with
in situ observations* (Space Weather manuscript, 2008-2022 hindcasts).

## Quick start

Create the pinned Conda environment, fetch the public OMNI observations, and
run the paper analysis from the repository root:

```powershell
conda env create -f environment.yml
conda activate torch
python code/grab_paper_data.py
python code/generate_flow_diagram.py
python code/insitu_huxt_forecasts_analysis.py
```

Figures and generated tables are written to `figures/`. The OMNI download is
cached as `data/omni_1hour_2007_2023.csv`; subsequent runs are offline.

The repository already contains the processed 1- through 27-day InSitu-HUXt
and WSA-HUXt forecast time series used for the paper. Therefore reproducing the
statistical figures does not require the multi-day raw model rebuild or access
to the original CCMC WSA FITS archive.

## Paper figure map

| Paper figure | Output | Generator |
|---|---|---|
| Forecast-method schematic | `flow_diagram.pdf` | `generate_flow_diagram.py` |
| Ecliptic comparison | `ecliptic_comparison_20140609.pdf` | Optional 2-D example block in `insitu_huxt_forecasts_analysis.py` (requires a matching raw WSA FITS file) |
| Example forecasts | `example_forecast_*.pdf` | `insitu_huxt_forecasts_analysis.py` |
| Full 4-day time series | `timeseries_comparison_4day.pdf` | `insitu_huxt_forecasts_analysis.py` |
| MAE versus lead time | `mae_vs_leadtime.pdf` | `insitu_huxt_forecasts_analysis.py` |
| Taylor/target diagram | `combined_diagram_4day.pdf` | `insitu_huxt_forecasts_analysis.py`, `taylor_diagram.py` |
| HSE demonstration/errors | `hse_demonstration_2017_H1.pdf`, `hse_errors_4day.pdf` | `insitu_huxt_forecasts_analysis.py`, `hse.py` |
| DTW demonstration/errors | `dtw_alignment_4day_2017H1.pdf`, `dtw_histograms_4day.pdf` | `insitu_huxt_forecasts_analysis.py` |

The analysis also emits the HSE and DTW summary tables used in the manuscript.

## Data provenance and grabbers

- **OMNI hourly solar wind:** downloaded from NASA CDAWeb through SunPy by
  `code/grab_paper_data.py` (the underlying implementation is
  `huxt_inputs.get_omni`).
- **Richardson-Cane ICME intervals:** the exact processed catalogue used in the
  analysis is bundled as `data/Richardson_Cane_Porcessed_ICME_list.csv`.
- **WSA-HUXt comparison:** processed lead-time time series are bundled as
  `data/wsa_huxt_forecasts_*day_leadtime.csv`. The original WSA solutions were
  supplied by NASA CCMC; `code/run_huxt_wsa_ccmc.py` documents their HUXt
  processing.
- **InSitu-HUXt hindcasts:** processed lead-time time series are bundled as
  `data/insitu_huxt_forecasts_*day_leadtime.csv`. Rebuild them with
  `code/insitu_huxt_forecasts.py` after downloading OMNI.
- **Optional DONKI data:** the reusable grabbers
  `huxt_inputs.get_DONKI_ICMEs` and `huxt_inputs.get_DONKI_coneCMEs` query the
  public NASA DONKI API.

## Rebuilding raw forecasts

Raw daily forecasts are not required for the paper figures. To rebuild them,
set writable directories and run the model scripts:

```powershell
$env:HUXT_INSITU_RAW_DIR = "D:\huxt\insitu"
$env:HUXT_WSA_RAW_DIR = "D:\huxt\wsa"
python code/insitu_huxt_forecasts.py
```

To make `insitu_huxt_forecasts_analysis.py` compile raw daily files back into
the bundled lead-time products, set `compute_forecast_timeseries_now = True` in
that script. Repository paths can be overridden with `HUXT_DATA_DIR`,
`HUXT_FIGURE_DIR`, `HUXT_OMNI_CACHE`, `HUXT_INSITU_RAW_DIR`, and
`HUXT_WSA_RAW_DIR`.

## Scope note

The numerical figure pipeline is self-contained apart from the public OMNI
download. The ecliptic WSA panel needs the specific CCMC FITS solution used for
that example; it is not needed to reproduce the paper's forecast scores or
conclusions.

