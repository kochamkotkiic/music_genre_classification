# scripts/train_knn.py
from pathlib import Path
import pandas as pd
import sys
import joblib

# Dodaj src do path'u
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_pipeline import GenericScikitLearner
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    """Trening kNN klasyfikatora"""
    
    print("🎯 TRENING kNN KLASYFIKATORA")
    print("="*50)
    
    # Ścieżki
    project_root = Path(__file__).parent.parent
    data_processed = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Wczytaj zbiory
    print("\n📂 Wczytywanie zbiorów...")
    train_df = pd.read_csv(data_processed / 'train_features.csv')
    val_df = pd.read_csv(data_processed / 'val_features.csv')
    test_df = pd.read_csv(data_processed / 'test_features.csv')
    
    print(f"   Train: {len(train_df)} próbek")
    print(f"   Val:   {len(val_df)} próbek")
    print(f"   Test:  {len(test_df)} próbek")
    
    # Przygotuj dane
    X_train = train_df.drop(['genre', 'filename'], axis=1)
    y_train = train_df['genre']
    
    X_val = val_df.drop(['genre', 'filename'], axis=1)
    y_val = val_df['genre']
    
    X_test = test_df.drop(['genre', 'filename'], axis=1)
    y_test = test_df['genre']
    
    # Utwórz i wytrenuj model kNN
    print("\n🔧 Tworzenie modelu kNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    model = GenericScikitLearner(knn, name="knn")
    
    print("🏋️ Trening modelu...")
    model.fit(X_train, y_train)
    
    # Ocena na zbiorze walidacyjnym
    print("\n📊 Ocena na zbiorze walidacyjnym...")
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"   Dokładność (val): {val_accuracy:.4f}")
    
    # Ocena na zbiorze testowym
    print("\n📊 Ocena na zbiorze testowym...")
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"   Dokładność (test): {test_accuracy:.4f}")
    
    # Raport klasyfikacji
    print("\n📋 Raport klasyfikacji (test):")
    print(classification_report(y_test, test_pred))
    
    # Macierz pomyłek
    print("\n📊 Macierz pomyłek (test):")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    
    # Zapisz model
    model_path = models_dir / 'knn_model.pkl'
    model.save(str(model_path))
    print(f"\n💾 Zapisano model: {model_path}")
    
    print("\n✅ TRENING ZAKOŃCZONY!")

if __name__ == '__main__':
    main()
