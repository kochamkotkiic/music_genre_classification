# scripts/01_prepare_datasets.py
from pathlib import Path
import pandas as pd
import sys
import importlib.util

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

def find_genres_folder(base_path: Path) -> Path:
    """
    Znajdź folder zawierający gatunki muzyki.
    Sprawdza różne możliwe struktury:
    - base_path/genres_original/
    - base_path/Data/genres_original/
    - base_path/ (bezpośrednio foldery gatunków)
    """
    expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # 1. Sprawdź bezpośrednio w base_path
    if base_path.exists():
        found_genres = [d for d in base_path.iterdir() 
                       if d.is_dir() and d.name in expected_genres]
        if len(found_genres) >= 3:
            return base_path
    
    # 2. Sprawdź base_path/genres_original/
    genres_original = base_path / 'genres_original'
    if genres_original.exists() and genres_original.is_dir():
        found_genres = [d for d in genres_original.iterdir() 
                       if d.is_dir() and d.name in expected_genres]
        if len(found_genres) >= 3:
            return genres_original
    
    # 3. Sprawdź base_path/Data/genres_original/
    data_folder = base_path / 'Data'
    if data_folder.exists() and data_folder.is_dir():
        genres_original_in_data = data_folder / 'genres_original'
        if genres_original_in_data.exists() and genres_original_in_data.is_dir():
            found_genres = [d for d in genres_original_in_data.iterdir() 
                           if d.is_dir() and d.name in expected_genres]
            if len(found_genres) >= 3:
                return genres_original_in_data
    
    # 4. Sprawdź base_path/Data/ (bezpośrednio foldery gatunków)
    if data_folder.exists() and data_folder.is_dir():
        found_genres = [d for d in data_folder.iterdir() 
                       if d.is_dir() and d.name in expected_genres]
        if len(found_genres) >= 3:
            return data_folder
    
    # 5. Rekurencyjne przeszukanie (max 2 poziomy głębokości)
    if base_path.exists():
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                # Sprawdź bezpośrednio w subdir
                found_genres = [d for d in subdir.iterdir() 
                               if d.is_dir() and d.name in expected_genres]
                if len(found_genres) >= 3:
                    return subdir
                
                # Sprawdź w podfolderach subdir (max 1 poziom głębiej)
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir():
                        found_genres = [d for d in subsubdir.iterdir() 
                                       if d.is_dir() and d.name in expected_genres]
                        if len(found_genres) >= 3:
                            return subsubdir
    
    return None

def main():
    """Zadanie 3: Przygotuj zbiory train/val/test"""
    
    print("🎯 ZADANIE 3: PRZYGOTOWANIE ZBIORÓW")
    print("="*50)
    
    # Ścieżki - używamy bezwzględnych ścieżek
    project_root = Path(__file__).parent.parent
    data_raw_base = project_root / 'data' / 'raw' / 'genres'
    data_processed = project_root / 'data' / 'processed'
    
    # Utwórz folder processed jeśli nie istnieje
    data_processed.mkdir(parents=True, exist_ok=True)
    
    # Znajdź folder z gatunkami
    print("\n🔍 Szukam folderu z gatunkami muzyki...")
    data_raw = find_genres_folder(data_raw_base)
    
    if data_raw is None:
        print(f"\n❌ Nie znaleziono folderu z gatunkami muzyki!")
        print(f"💡 Sprawdzam w: {data_raw_base}")
        print(f"\n📁 Zawartość {data_raw_base}:")
        
        if data_raw_base.exists():
            for item in sorted(data_raw_base.iterdir())[:10]:
                if item.is_dir():
                    print(f"   📁 {item.name}/")
                    # Pokaż co jest w podfolderze
                    try:
                        subitems = list(item.iterdir())[:5]
                        if subitems:
                            subdirs = [s.name for s in subitems if s.is_dir()]
                            if subdirs:
                                print(f"      → {', '.join(subdirs[:5])}" + 
                                      ("..." if len(subdirs) > 5 else ""))
                    except:
                        pass
                else:
                    print(f"   📄 {item.name}")
        else:
            print(f"   (folder nie istnieje)")
        
        print(f"\n💡 Oczekiwana struktura:")
        print(f"   {data_raw_base}/")
        print(f"   └── Data/")
        print(f"       └── genres_original/")
        print(f"           ├── blues/")
        print(f"           ├── classical/")
        print(f"           └── ...")
        return
    
    print(f"✅ Znaleziono folder z gatunkami: {data_raw}")
    
    # Sprawdź ile gatunków znaleziono
    expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
    found_genres = [d.name for d in data_raw.iterdir() 
                   if d.is_dir() and d.name in expected_genres]
    print(f"📊 Znaleziono {len(found_genres)} gatunków: {', '.join(sorted(found_genres))}")
    
    # 1. Ekstrakcja cech
    print("\n1️⃣ EKSTRAKCJA CECH...")
    extractor = FeatureExtractor()
    
    features_df = extractor.extract_from_directory(data_raw)
    
    # Walidacja - sprawdź czy znaleziono jakieś pliki
    if features_df.empty or len(features_df) == 0:
        print(f"\n❌ Nie znaleziono żadnych plików audio!")
        print(f"💡 Sprawdź:")
        print(f"   1. Czy w folderach gatunków są pliki audio?")
        print(f"   2. Czy pliki mają rozszerzenia: .wav, .au, .mp3, .flac, .ogg?")
        
        # Pokaż co jest w folderach
        print(f"\n📁 Zawartość folderów gatunków:")
        for genre_dir in sorted([d for d in data_raw.iterdir() if d.is_dir()])[:5]:
            files = list(genre_dir.glob('*.*'))
            print(f"   {genre_dir.name}: {len(files)} plików")
            if files:
                print(f"      Przykład: {files[0].name}")
        return
    
    # Zapisz pełne cechy
    features_path = data_processed / 'features.csv'
    features_df.to_csv(features_path, index=False)
    print(f"💾 Zapisano: {features_path}")
    
    # 2. Podział na zbiory
    print("\n2️⃣ PODZIAŁ NA ZBIORY (60% train, 20% val, 20% test)...")
    
    # Sprawdź czy jest kolumna 'genre'
    if 'genre' not in features_df.columns:
        print(f"❌ Brak kolumny 'genre' w DataFrame!")
        print(f"   Dostępne kolumny: {list(features_df.columns)}")
        return
    
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
