# Results folder

Use this folder for:

- `experiment_log.csv`: parameters and outcomes for each run
- `metrics_summary.csv`: compact train/validation/test metrics for each run
- prediction plots for baseline, RNN, and LSTM

Recommended plot names:

- `baseline_predictions.png`
- `baseline_residuals.png`
- `rnn_predictions.png`
- `lstm_predictions.png`

The persistence baseline appends MAE/RMSE values to both CSV files and records
the generated prediction plot path in `experiment_log.csv`.

