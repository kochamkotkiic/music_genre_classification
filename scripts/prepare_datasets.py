# scripts/01_prepare_datasets.py
from pathlib import Path
import pandas as pd
import sys

# Dodaj src do path'u
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_extractor import FeatureExtractor
from dataset_splitter import DatasetSplitter

def main():
    """Zadanie 3: Przygotuj zbiory train/val/test"""
    
    print("🎯 ZADANIE 3: PRZYGOTOWANIE ZBIORÓW")
    print("="*50)
    
    # Ścieżki - używamy bezwzględnych ścieżek
    project_root = Path(__file__).parent.parent
    data_raw = project_root / 'data' / 'raw' / 'genres'
    data_processed = project_root / 'data' / 'processed'
    
    # Utwórz folder processed jeśli nie istnieje
    data_processed.mkdir(parents=True, exist_ok=True)
    
    # 1. Ekstrakcja cech
    print("\n1️⃣ EKSTRAKCJA CECH...")
    extractor = FeatureExtractor()
    
    if not data_raw.exists():
        print(f"❌ Folder {data_raw} nie istnieje!")
        print(f"💡 Upewnij się, że dane są skopiowane do {data_raw}")
        return
    
    features_df = extractor.extract_from_directory(data_raw)
    
    # Zapisz pełne cechy
    features_path = data_processed / 'features.csv'
    features_df.to_csv(features_path, index=False)
    print(f"💾 Zapisano: {features_path}")
    
    # 2. Podział na zbiory
    print("\n2️⃣ PODZIAŁ NA ZBIORY (60% train, 20% val, 20% test)...")
    splitter = DatasetSplitter()
    train_df, val_df, test_df = splitter.split(features_df)
    
    # Zapisz zbiory
    train_path = data_processed / 'train_features.csv'
    val_path = data_processed / 'val_features.csv'
    test_path = data_processed / 'test_features.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Zapisano zbiory:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    print("\n✅ ZADANIE 3 ZAKOŃCZONE!")
    
    # Statystyka
    print(f"\n📊 Końcowa statystyka:")
    print(f"   Cech: {features_df.shape[1]-2}")  # -2 dla 'genre' i 'filename'
    print(f"   Próbki: {features_df.shape[0]}")
    print(f"   Gatunki: {features_df['genre'].nunique()}")
    print(f"   Rozkład gatunków:")
    for genre, count in features_df['genre'].value_counts().sort_index().items():
        print(f"     {genre}: {count}")

if __name__ == '__main__':
    main()
