# Music Genre Classification

This project aims to classify music tracks into genres based on audio features extracted from each track. The analysis uses the **GTZAN dataset** from TensorFlow Datasets.

## Project Structure

- `notebooks/` – Jupyter notebooks for EDA and visualization
- `src/` – Python scripts for feature extraction and model training
- `data/` – Instructions for downloading dataset (data not included)
- `docker/` – Dockerfile and environment setup
- `README.md` – Project documentation

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music_genre_classification.git
2. Install dependencies:
   ```bash
    pip install -r requirements.txt

3. Download the GTZAN dataset following instructions in data/README.md.

4. Usage

Run the notebooks in notebooks/ to explore the dataset and visualize audio features.

Use scripts in src/ to train classifiers using scikit-learn.

5. Notes

.ipynb_checkpoints are ignored.

Dataset is not included due to size restrictions.
