# 🐱🎧 Klasyfikacja Gatunków Muzycznych (GTZAN)  

Projekt uczenia maszynowego mający na celu automatyczną klasyfikację gatunków muzycznych na podstawie analizy sygnału audio. Wykorzystano zbiór danych **GTZAN** oraz różne algorytmy klasyfikacji (od prostych modeli po sieci neuronowe).

##  Zbiór Danych i Cechy
* **Dataset:** GTZAN (1000 utworów, 10 gatunków, po 100 próbek 30-sekundowych).
* **Gatunki:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock.
* **Ekstrakcja Cech (59 wymiarów):**
  * **MFCC (13):** Barwa dźwięku.
  * **Chroma (12):** Cechy harmoniczne.
  * **Spectral Contrast:** Rozkład energii w pasmach (kluczowe dla Drzew Decyzyjnych).
  * **RMS Energy:** Głośność/dynamika.
  * **Zero Crossing Rate:** Hałaśliwość sygnału.

##  Wyniki Modeli

Przetestowano 5 głównych podejść. Najlepszym klasycznym modelem okazał się **SVM z jądrem RBF**.

| Model | Dokładność (Accuracy) | Kluczowe wnioski |
| :--- | :---: | :--- |
| **SVM (RBF)** | **70.5%** | **Zwycięzca.** Świetnie radzi sobie z nieliniowością danych. |
| **MLP (Neural Net)** | ~69.0% | Wysoki potencjał, ale wymagał precyzyjnego strojenia (architektura piramidalna). |
| **KNN (k=10)** | 60.0% | Solidny baseline. Dobry dla *Classical*, słaby dla *Rock/Country*. |
| **Naive Bayes** | 58.0% | Zbyt proste założenia (niezależność cech), problem *High Bias*. |
| **Decision Tree** | 53.5% | Najsłabszy wynik, ale wysoka interpretowalność. |

###  Najważniejsze wnioski z analizy (EDA & Modele):
1.  **Najłatwiejsze do rozróżnienia:** *Classical* i *Jazz* (unikalna dynamika i spektrum).
2.  **Najtrudniejsze pary:** *Rock* vs *Country* vs *Disco* (podobne instrumentarium i rytmika).
3.  **Kluczowa cecha:** *Spectral Contrast* okazał się ważniejszy niż MFCC w modelach drzewiastych.
4.  **Skalowanie:** Standaryzacja (`StandardScaler`) była krytyczna dla wyników PCA i treningu sieci neuronowych.

---

##  Jak uruchomić projekt

### Opcja 1: Docker (Rekomendowane)
Środowisko jest w pełni skonteryzowane. Wymaga zainstalowanego Dockera.

1. Przejdź do folderu `docker`:
   ```bash
   cd music_genre_classification/docker
2. Uruchom kontener z mapowaniem danych (podmień ścieżkę do danych GTZAN):
   ```bash
   docker-compose run --rm -p 8888:8888 -v "C:\Sciezka\Do\Danych\GTZAN:/app/data" ml-project
3. Jupyter Lab uruchomi się na porcie 8888. Token znajdziesz w konsoli.
### Opcja 2: Lokalnie (Python 3.10+)
1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r docker/requirements.txt
2. Uruchom Jupyter Lab w folderze projektu:
   ```bash
    jupyter lab
   
###  Struktura Projektu
data/ - Miejsce na przetworzone dane.

docker/ - Pliki konfiguracyjne Dockerfile i docker-compose.

notebooks/ - Notatniki Jupyter z kodem (EDA, Trening modeli).

scripts/ - Skrypty pomocnicze (feature_extractor.py, prepare_datasets.py).

models/ - Zapisane wytrenowane modele .pkl. 
