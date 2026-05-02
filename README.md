# Weather Forecasting with Manual RNN and LSTM

This project uses the Jena Climate dataset to predict future temperature from
historical weather observations. The implementation compares a persistence
baseline against a manually implemented RNN and a manually implemented LSTM.

The dataset is not included in this repository. Use the public dataset URL and
the provided download command so the code can be run without bundling large
data files inside the submission.

## Problem setup

- Dataset: Jena Climate
- Dataset URL: `https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip`
- Target variable: `T (degC)`
- Default input window: previous 24 hourly observations
- Default forecast horizon: next 1 hour
- Default comparison: Persistence Baseline vs Manual RNN vs Manual LSTM

The raw dataset is recorded every 10 minutes. This project resamples it to
hourly data before training.

## Repository structure

```text
Group-Project-ML/
|-- data/
|   |-- raw/
|   `-- processed/
|-- results/
|-- src/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run order

1. Download the dataset:

```bash
python main.py download-data
```

2. Inspect the raw file:

```bash
python main.py describe-data
```

3. Preprocess the dataset into hourly observations:

```bash
python main.py preprocess
```

4. Run the persistence baseline:

```bash
python main.py baseline
```

5. Train and evaluate the manual RNN:

```bash
python main.py train-rnn
```

6. Train and evaluate the manual LSTM:

```bash
python main.py train-lstm
```

## Output files

The main outputs are written to `results/`:

- `experiment_log.csv`
- `metrics_summary.csv`
- prediction plots for baseline, RNN, and LSTM
- residual plots for RNN and LSTM

Processed hourly data is written to `data/processed/jena_climate_hourly.csv`.
The experiment log and metrics summary files are maintained as the required run
log for recorded experiments and parameter settings.

## Main source files

- `src/data_loader.py`: dataset download and CSV loading
- `src/preprocessing.py`: cleaning, hourly resampling, normalization, and splits
- `src/sequences.py`: sliding-window sequence generation
- `src/baseline.py`: persistence baseline
- `src/manual_rnn.py`: manual many-to-one RNN implementation
- `src/manual_lstm.py`: manual many-to-one LSTM implementation
- `src/train_rnn.py`: RNN training and evaluation
- `src/train_lstm.py`: LSTM training and evaluation
