# 🎵 Music Genre Classification - GTZAN

Klasyfikacja utworów muzycznych na **10 gatunków** z **GTZAN dataset** (1000 nagrań × 30s).

##  Struktura Projektu
```plaintext
├── data/          ← przetworzone cechy (train/val/test)
├── models/        ← knn_model.pkl 
├── notebooks/     ← EDA + wizualizacje
├── scripts/       ← wczytanie danych + trening
├── src/           ← ML funkcje
└── docker/        ← środowisko
 ```
**Uruchomienie programu:**

cd ścieżka do projektu/docker

**Uruchomienie dockera z montowanym folderem (ścieżka do folderu z danymi)**

docker-compose run --rm -v "ścieżka do projektu:/app/data" ml-project bash

**Uruchomienie skryptu:**

Wewnątrz kontenera:
python scripts/prepare_datasets.py --data-dir "/app/data"

Wyniki (data/processed) zapisują się w folderze projektu.

Po wczytaniu danych należu uruchomić kod treningu:
python scripts/train_knn.py

##  Wyniki kNN (Baseline)
 Dokładność test: 57.5%
 Classical: 88% F1 (NAJLEPSZY)
 Rock: 22% F1 (NAJGORSZY)

##  Analiza GTZAN Dataset
- **1000 nagrań**, 10 gatunków × 100 utworów
- **Średnia długość:** 30s (29.9-30.6s) 
- **RMS Energy:** Classical↓ | Pop/Metal↑
- **MFCC1:** Energia (Classical niska)
- **t-SNE:** Classical/Jazz separują się najlepiej
- **Trudne pary:** Rock↔Country↔Disco




