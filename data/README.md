# Data layout

Do not commit the raw dataset to the repository. The assignment requires a public dataset link rather than bundling the dataset in the submission.

## Dataset

- Name: Jena Climate
- Public download: `https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip`

## Expected local paths

```text
data/raw/jena_climate_2009_2016.csv
data/processed/jena_climate_hourly.csv
```

## Workflow

1. Download and extract the dataset with:

```bash
python main.py download-data
```

2. Inspect the raw file and confirm the columns/date range:

```bash
python main.py describe-data
```

3. Build the processed hourly dataset:

```bash
python main.py preprocess
```

That command creates:

- `data/processed/jena_climate_hourly.csv`
- `data/processed/jena_climate_hourly_metadata.json`

The preprocessing step also replaces the known `-9999.0` bad wind-velocity sentinels before resampling.
