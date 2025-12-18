# scripts/01_prepare_datasets.py
from pathlib import Path
import pandas as pd
import sys
import importlib.util
import argparse

# Dodaj src do path'u
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Załaduj moduły bezpośrednio
spec_fe = importlib.util.spec_from_file_location("feature_extractor", src_path / "feature_extractor.py")
spec_ds = importlib.util.spec_from_file_location("dataset_splitter", src_path / "dataset_splitter.py")

feature_extractor_module = importlib.util.module_from_spec(spec_fe)
dataset_splitter_module = importlib.util.module_from_spec(spec_ds)

spec_fe.loader.exec_module(feature_extractor_module)
spec_ds.loader.exec_module(dataset_splitter_module)

FeatureExtractor = feature_extractor_module.FeatureExtractor
DatasetSplitter = dataset_splitter_module.DatasetSplitter


def main():
    parser = argparse.ArgumentParser(description='Task 3: Przygotowanie zbiorów GTZAN')
    parser.add_argument('--data-dir', type=str, required=True, help='Ścieżka do folderu z gatunkami')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Ścieżka do CSV')

    args = parser.parse_args()

    data_raw = Path(args.data_dir).resolve()
    data_processed = Path(args.output_dir).resolve()
    data_processed.mkdir(parents=True, exist_ok=True)  # Utwórz folder

    print(f" Dane wejściowe: {data_raw}")
    print(f" Dane wyjściowe: {data_processed}")


    if not data_raw.exists():
        print(f" Folder NIE istnieje: {data_raw}")
        return

    print(f" Znaleziono folder z gatunkami!")

    extractor = FeatureExtractor()
    features_df = extractor.extract_from_directory(data_raw)

    # Walidacja
    if features_df.empty or len(features_df) == 0:
        print(f"\n Nie znaleziono żadnych plików audio!")
        print(f" Zawartość folderów:")
        for genre_dir in sorted([d for d in data_raw.iterdir() if d.is_dir()])[:5]:
            files = list(genre_dir.glob('*.*'))
            print(f"   {genre_dir.name}: {len(files)} plików")
            if files:
                print(f"      Przykład: {files[0].name}")
        return

    features_path = data_processed / 'features.csv'
    features_df.to_csv(features_path, index=False)
    print(f" Zapisano cechy: {features_path}")

    # Podział na zbiory
    print("\n2️ PODZIAŁ NA ZBIORY...")
    if 'genre' not in features_df.columns:
        print(f" Brak kolumny 'genre'! Kolumny: {list(features_df.columns)}")
        return

    splitter = DatasetSplitter()
    train_df, val_df, test_df = splitter.split(features_df)


    train_path = data_processed / 'train_features.csv'
    val_path = data_processed / 'val_features.csv'
    test_path = data_processed / 'test_features.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n Zapisano zbiory:")
    print(f"    Train: {len(train_df)} próbek → {train_path}")
    print(f"    Val:   {len(val_df)} próbek → {val_path}")
    print(f"    Test:  {len(test_df)} próbek → {test_path}")

    print(f"📊 Statystyka: {features_df.shape[0]} próbek, {features_df.shape[1] - 2} cech")


if __name__ == '__main__':
    main()
