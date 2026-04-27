# Group Project ML

Project scaffold for the ML course final project on `RNN/LSTM for time series prediction` using the Jena Climate dataset.

The repository is structured around the assignment requirements:

- manual implementation of the main learning algorithms
- public dataset link instead of committing raw data
- reproducible run instructions
- results artifacts and experiment logs
- report support for the final IEEE-format submission

## Proposed project scope

- Task: predict future temperature from historical weather data
- Dataset: Jena Climate
- Resampling: hourly averages
- Default target: `T (degC)`
- Default window: previous 24 hours
- Default horizon: next 1 hour
- Comparison: persistence baseline vs manual RNN vs manual LSTM

## Repository layout

```text
Group-Project-ML/
|-- data/
|   |-- README.md
|   |-- raw/
|   `-- processed/
|-- report/
|-- results/
|-- src/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and extract the Jena Climate dataset:

```bash
python main.py download-data
```

4. Inspect the raw dataset summary:

```bash
python main.py describe-data
```

5. Build the hourly dataset:

```bash
python main.py preprocess
```

6. Run the persistence baseline:

```bash
python main.py baseline
```

7. Inspect the scaffolded manual-model pipeline:

```bash
python main.py train-rnn
python main.py train-lstm
```

Those last two commands currently prepare normalized sequence splits and initialize model stubs. They are intentionally not full training runs yet because the manual algorithm implementation still needs to be filled in by the team.

## Default workflow

1. Preprocess raw 10-minute data into hourly data.
2. Run the persistence baseline and log RMSE/MAE.
3. Implement the manual RNN in [src/manual_rnn.py](/c:/Users/17372/Desktop/ML Assignment 3/Group-Project-ML/src/manual_rnn.py).
4. Implement the manual LSTM in [src/manual_lstm.py](/c:/Users/17372/Desktop/ML Assignment 3/Group-Project-ML/src/manual_lstm.py).
5. Save plots and metrics in `results/`.
6. Write the final report in IEEE format and export it to `report/`.

## Assignment alignment

This scaffold matches the project description by including:

- code with clear run instructions
- a place to record experiment parameters and outcomes
- a results folder for plots and summary metrics
- a report folder for the final PDF
- separation between preprocessing/evaluation utilities and the manual model code

## Immediate coding targets

- `src/data_loader.py`: handle dataset download/extract and robust CSV parsing
- `src/preprocessing.py`: clean sentinels, resample hourly, normalize from train-only stats, and build sequence splits
- `src/sequences.py`: expose model-ready sliding-window arrays
- `src/manual_rnn.py`: implement forward pass, loss, BPTT, and updates
- `src/manual_lstm.py`: implement gates, cell state updates, loss, BPTT, and updates
- `src/train_rnn.py`: connect the manual RNN training loop
- `src/train_lstm.py`: connect the manual LSTM training loop
