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

1. Download the ZIP file from the public link above.
2. Extract `jena_climate_2009_2016.csv`.
3. Place it into `data/raw/`.
4. Run:

```bash
python main.py preprocess
```

That command creates the hourly dataset used by the baseline and the manual-model training scaffolds.

