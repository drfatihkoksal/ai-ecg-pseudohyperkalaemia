# AI-ECG potassium to triage haemolysed hyperkalaemia

Analysis code for the study *An electrocardiogram-based deep learning potassium estimate to
distinguish haemolytic pseudohyperkalaemia from true hyperkalaemia* (development on HEEDB,
external validation on MIMIC-IV).

The repository holds **code only**. It contains no patient data, no record-level derived files
(labels, cohort tables, waveform stores) and no trained model weights. Both data sources are
credentialed and their use agreements forbid redistributing record-level data or derivatives.
To reproduce the analysis you need your own approved access to each source.

## Data access

| Source | Version | Access | Used for |
|---|---|---|---|
| MIMIC-IV | v3.1 (doi:10.13026/kpb9-mt58) | PhysioNet credentialed | external validation: labs, eMAR, ICU inputs/procedures, admissions, diagnoses |
| MIMIC-IV-ECG | v1.0 (doi:10.13026/4nqg-sb35) | PhysioNet | external validation: waveforms, `record_list.csv`, `machine_measurements.csv` |
| Harvard-Emory ECG Database (HEEDB) | v5.0 (doi:10.60508/rv6h-7d10), site I0001 | Brain Data Science Platform (BDSP), credentialed DUA | development: WFDB waveforms, metadata, 12SL statements, ICD codes |
| HEEDB site I0001 structured EHR | Labs, Medications (parquet) | BDSP, same DUA | development: potassium values, haemolysis flags, potassium-lowering therapy |

Expected layout under `HEEDB_ROOT`: `ECG/I0001/{metadata,12SL_diagnoses,ICD_codes,...}` and
`EHR/I0001/{Labs,Medications}/*.parquet` (copied from the BDSP access point
`EHR/I0001-EHR/data_Structured/I0001_ParquetFiles`).

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # install the torch build that matches your CUDA

export MIMIC_ROOT=/path/to/mimiciv/3.1           # contains hosp/ and icu/
export MIMIC_ECG_ROOT=/path/to/mimic-iv-ecg/1.0  # contains files/, record_list.csv, machine_measurements.csv
export HEEDB_ROOT=/path/to/heedb                 # see layout above
# optional: MIMICABY_ROOT defaults to the repository root; data/ and outputs/ are created under it
```

Developed with Python 3.13, PyTorch 2.11 (CUDA 12.8) on one GPU. Scripts read `MIMICABY_ROOT`
as the repository root and import each other by path from `scripts/`, so keep that layout.

## Pipeline for the paper

Run from the repository root, in order. Stages marked *upstream* belong to an earlier phase of
the project but produce inputs the paper pipeline reads.

| # | Script | Writes | Paper |
|---|---|---|---|
| 1 | `00_extract_labs.py` (*upstream*) | `data/labs_cr_k.parquet` (creatinine, potassium) | Methods |
| 2 | `03_build_manifest.py` (*upstream*) | `data/pairs_manifest.parquet` | — |
| 3 | `06_preprocess_waveforms.py` (*upstream*) | `data/waveforms/ecg_meta.parquet`; also the shared preprocessing module (250 Hz, 0.5–40 Hz band-pass, 60 Hz notch, no per-record normalisation) imported by 21 and 26 | Suppl. Methods |
| 4 | `20_pseudohyperk_labels.py` | `data/pseudohyperk_index.parquet` (haemolysed index specimens; pseudo/true/reference labels) | Methods, Fig. 1 |
| 5 | `21_preprocess_v3.py` | `data/waveforms_v3/`, `data/v3_ecg_k.parquet` | — |
| 6 | `22_train_k_v3.py` | `outputs/v3/ecgk_oof.parquet` (MIMIC cross-fitted comparator) | Suppl. |
| 7 | `23_khodorkovsky.py` | `data/pseudohyperk_features.parquet`, `outputs/khodorkovsky_rule.csv` | comparator rule |
| 8 | `25_heedb_pairs.py` | `data/heedb/heedb_ecg_k.parquet` | — |
| 9 | `26_preprocess_heedb.py --workers 16 --chunk 200000` | `$HEEDB_ROOT/derived/strips16.npy` (~113 GB), `meta.parquet`; resumable | — |
| 10 | `27_heedb_labels.py` | `data/heedb/heedb_ecg_k_labels.parquet` | Methods |
| 11 | `28_train_k_heedb.py` | `data/models/heedb/ecgk_{A,B}.pt`, `FROZEN.json`, `outputs/heedb/internal_test.csv` | Table 2 (internal) |
| 12 | `29_external_mimic.py` | `outputs/heedb/mimic_ecgk_external.parquet`, `external_mimic_accuracy.csv` | Table 2 (external) |
| 13 | `24_eval_v3.py --ecgk outputs/heedb/mimic_ecgk_external.parquet --tag _external` | `outputs/v3/eval_*_external.csv` | Table 3 |
| 14 | `30_subgroups.py` | `outputs/heedb/subgroups*.csv` | Suppl. subgroup tables |
| 15 | `31_paper_extras.py` | `outputs/paper/table1.csv`, `decision_ci*.csv`, `decision_curve.csv`, `flow_counts.csv`, Figures 2–3 | Table 1, decision analysis, Figs 2–3 |
| 16 | `32_figure1_flow.py` | `outputs/paper/figure1_flow.(png\|pdf)` | Fig. 1 |

Note on steps 2–3: `20_pseudohyperk_labels.py` reads `ecg_meta.parquet` only through a LEFT JOIN
for descriptive QC columns; the cohort, labels and the QC filter used in every analysis come from
steps 4–5 (`waveforms_v3/meta.parquet`, `ok` and ≤1 bad lead). Step 3 must still have run so the
file exists.

Design safeguards that the code enforces:

- Splits are by patient. HEEDB is split 80/10/10; model variant selection uses the HEEDB
  validation split only, and `28_train_k_heedb.py` writes `FROZEN.json` before MIMIC is read.
- `29_external_mimic.py` refuses to run without `FROZEN.json` and only scores; nothing is
  trained or tuned on MIMIC.
- The decision threshold (≤2% missed true hyperkalaemia) is set on training folds and refitted
  inside each patient-cluster bootstrap replicate (`31_paper_extras.py`).

Compute: steps 9 and 11 dominate (tens of hours of CPU preprocessing; several GPU hours of
training). DuckDB scans of `labevents` and the HEEDB medication table need ≥64 GB RAM; do not run
them alongside step 9.

## Other scripts

`01`, `02`, `04`, `05`, `07*`–`19`, `run_*.sh` are from earlier phases of the project (paired-ECG
creatinine/potassium change models). They are kept for transparency about the analytic history
and are not needed for the paper. `01` and `02` were first written against the
MIMIC-IV-ECG *matched subset* release and may need adjustment for v1.0.

## Data-use note

Do not commit anything under `data/` or `outputs/` (both are git-ignored): they contain
record-level derivatives. Trained weights are record-level derivatives under the BDSP DUA and
can be shared only with BDSP approval.

## Citation

See `CITATION.cff`; the archived release DOI (Zenodo) and the article citation will be added here.

## License

MIT (code only; see `LICENSE`).
