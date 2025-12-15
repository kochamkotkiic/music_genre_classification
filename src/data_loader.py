################################################################################
# src/data_loader.py
# GTZAN Dataset Loader - Music Genre Classification
################################################################################

"""
GTZAN Genre Dataset Loader using mirdata 0.3.9

Dataset Specification:
- Name in mirdata: 'gtzan_genre' (podkreślenie, nie myślnik!)
- Total samples: 1000 audio tracks
- Genres: 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- Samples per genre: 100 tracks
- Duration: 30 seconds each
- Sample rate: 22050 Hz (mono, 16-bit)
- Format: WAV
- Size: ~1.2 GB
- Location: ~/.mirdata/GTZAN-Genre/ (automatycznie przez mirdata)
"""

import mirdata
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


class GTZANLoader:
    """
    Loader dla GTZAN-Genre dataset'u z mirdata
    
    Prawidłowa nazwa: 'gtzan_genre' (podkreślenie!)
    API: https://mirdata.readthedocs.io/en/0.3.9/source/mirdata.datasets.gtzan_genre.html
    """
    
    DATASET_NAME = 'gtzan_genre'
    GENRES = [
        'blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    
    def __init__(self, data_home: str = None):
        """
        Args:
            data_home (str): Ścieżka do folderu na dane
                            Jeśli None, mirdata używa: ~/.mirdata/GTZAN-Genre/
        """
        self.data_home = data_home
        self.dataset = None
        
        # Inicjalizuj dataset
        self._init_dataset()
    
    def _init_dataset(self):
        """
        Inicjalizuj mirdata dataset
        """
        try:
            self.dataset = mirdata.initialize(
                self.DATASET_NAME,
                data_home=self.data_home
            )
            print(f"✅ GTZAN-Genre dataset initialized (mirdata 0.3.9)")
        except Exception as e:
            print(f"❌ Błąd inicjalizacji dataset'u: {e}")
            raise
    
    # ========== POBIERANIE ==========
    
    def download(self) -> bool:
        """
        Pobierz GTZAN-Genre via mirdata
        
        ℹ️  Dane pobierają się automatycznie do: ~/.mirdata/GTZAN-Genre/
        
        Returns:
            bool: True jeśli pobranie się powiodło
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            print(f"📥 Pobieranie GTZAN-Genre via mirdata...")
            print("ℹ️  Dane zostaną umieszczone w ~/.mirdata/GTZAN-Genre/")
            print("⏳ Pobieranie (może trwać 5-20 minut)...\n")
            
            # mirdata automatycznie sprawdza czy dataset jest pobrany
            # i pobiera go jeśli trzeba
            self.dataset.download()
            
            print("\n✅ GTZAN-Genre pobrane/zweryfikowane pomyślnie!")
            return True
            
        except Exception as e:
            print(f"❌ Błąd przy pobraniu: {e}")
            raise
    
    # ========== ŁADOWANIE DANYCH ==========
    
    def load_tracks(self) -> Dict:
        """
        Załaduj wszystkie utwory z GTZAN
        
        Returns:
            dict: Słownik {track_id: Track object}
        
        Track attributes:
            - track_id (str): ID utworu (np. 'blues.00000')
            - genre (str): Gatunek muzyki
            - audio: tuple (audio_array, sample_rate)
            - audio_path: str - ścieżka do pliku audio
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            print("📂 Ładowanie wszystkich track'ów...")
            tracks = self.dataset.load_tracks()
            print(f"✅ Załadowano {len(tracks)} utworów z GTZAN-Genre")
            return tracks
            
        except Exception as e:
            print(f"❌ Błąd przy ładowaniu: {e}")
            print("\n💡 Wskazówka: Dataset może nie być pobrany.")
            print("   Uruchom: python scripts/download_data.py")
            raise
    
    def get_track_ids(self) -> List[str]:
        """
        Pobierz listę wszystkich track ID
        
        Returns:
            list: Lista track ID (np. ['blues.00000', 'blues.00001', ...])
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            return self.dataset.track_ids
        except Exception as e:
            print(f"❌ Błąd przy pobieraniu track ID: {e}")
            raise
    
    def get_track(self, track_id: str):
        """
        Pobierz konkretny track
        
        Args:
            track_id (str): ID utworu (np. 'blues.00000')
        
        Returns:
            Track: Obiekt track'u
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            return self.dataset.track(track_id)
        except Exception as e:
            print(f"❌ Błąd przy ładowaniu track'u {track_id}: {e}")
            raise
    
    # ========== INFORMACJE O DATASET'CIE ==========
    
    def get_dataset_info(self) -> Dict:
        """
        Informacje o dataset'cie
        
        Returns:
            dict: Metadane dataset'u
        """
        info = {
            'dataset_name': 'GTZAN-Genre',
            'dataset_id_mirdata': 'gtzan_genre',
            'total_samples': 1000,
            'num_genres': len(self.GENRES),
            'genre_list': self.GENRES,
            'samples_per_genre': 100,
            'duration_per_sample_seconds': 30,
            'sample_rate': 22050,
            'format': 'WAV (16-bit mono)',
            'total_size_gb': 1.2,
            'data_location': '~/.mirdata/GTZAN-Genre/',
            'source': 'Tzanetakis & Cook (2002)',
            'mirdata_version': '0.3.9',
        }
        return info
    
    def print_dataset_info(self):
        """
        Wypisz informacje o dataset'cie
        """
        print("\n" + "="*70)
        print("GTZAN-GENRE DATASET INFORMATION")
        print("="*70)
        
        info = self.get_dataset_info()
        
        for key, value in info.items():
            if isinstance(value, list) and key == 'genre_list':
                print(f"\n{key}:")
                for i, genre in enumerate(value, 1):
                    print(f"  {i:2}. {genre}")
            else:
                print(f"{key}: {value}")
        
        print("\n" + "="*70 + "\n")
    
    # ========== STATYSTYKA ==========
    
    def get_genre_distribution(self) -> Dict[str, int]:
        """
        Rozkład gatunków w dataset'cie
        
        Returns:
            dict: {genre: count}
        """
        tracks = self.load_tracks()
        
        genres = {}
        for track in tracks.values():
            genre = track.genre
            genres[genre] = genres.get(genre, 0) + 1
        
        return genres
    
    def print_genre_statistics(self):
        """
        Wypisz statystykę gatunków
        """
        genres = self.get_genre_distribution()
        
        print("\n🎼 Rozkład gatunków:")
        print("-" * 50)
        
        for genre in sorted(genres.keys()):
            count = genres[genre]
            bar_length = count // 5
            bar = "█" * bar_length
            print(f"  {genre:12} │ {bar:20} {count:3} utworów")
        
        print("-" * 50)
        print(f"  RAZEM:         {sum(genres.values())} utworów\n")
    
    # ========== WALIDACJA I INFO O TRACK'ach ==========
    
    def get_track_info(self, track_id: str) -> Dict:
        """
        Informacje o konkretnym utworze
        
        Args:
            track_id (str): ID utworu
        
        Returns:
            dict: Informacje o utworze
        """
        try:
            track = self.get_track(track_id)
            
            # Załaduj audio do uzyskania metadanych
            audio, sr = track.audio
            duration = len(audio) / sr
            
            info = {
                'track_id': track.track_id,
                'genre': track.genre,
                'audio_path': track.audio_path,
                'duration_seconds': duration,
                'sample_rate': sr,
                'num_samples': len(audio),
            }
            
            return info
            
        except Exception as e:
            print(f"❌ Błąd przy pobieraniu info o track'u: {e}")
            raise
    
    def validate_dataset(self) -> Tuple[List, List]:
        """
        Waliduj dataset (sprawdź czy wszystkie pliki istnieją)
        
        Returns:
            tuple: (missing_files, invalid_checksums)
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            print("🔍 Walidacja dataset'u...")
            missing, invalid = self.dataset.validate()
            
            if not missing and not invalid:
                print("✅ Dataset jest prawidłowy (wszystkie pliki OK)")
            else:
                if missing:
                    print(f"⚠️  Brakujące pliki: {len(missing)}")
                    for file in missing[:5]:  # Pokaż pierwsze 5
                        print(f"    - {file}")
                if invalid:
                    print(f"⚠️  Pliki z niewłaściwą sumą kontrolną: {len(invalid)}")
                    for file in invalid[:5]:  # Pokaż pierwsze 5
                        print(f"    - {file}")
            
            return missing, invalid
            
        except Exception as e:
            print(f"❌ Błąd przy walidacji: {e}")
            raise
    
    # ========== SAMPLE OPERATIONS ==========
    
    def get_random_track(self):
        """
        Pobierz losowy track z dataset'u
        
        Returns:
            Track: Losowy track
        """
        if self.dataset is None:
            self._init_dataset()
        
        try:
            return self.dataset.choice_track()
        except Exception as e:
            print(f"❌ Błąd przy wyborze losowego track'u: {e}")
            raise
    
    def get_genre_samples(self, genre: str, limit: int = None) -> Dict:
        """
        Pobierz wszystkie track'i danego gatunku
        
        Args:
            genre (str): Nazwa gatunku (np. 'blues')
            limit (int): Maksymalna liczba track'ów (None = wszystkie)
        
        Returns:
            dict: {track_id: Track}
        """
        if genre not in self.GENRES:
            raise ValueError(f"Nieznany gatunek: {genre}. Dostępne: {self.GENRES}")
        
        all_tracks = self.load_tracks()
        genre_tracks = {
            track_id: track 
            for track_id, track in all_tracks.items() 
            if track.genre == genre
        }
        
        if limit is not None:
            genre_tracks = dict(list(genre_tracks.items())[:limit])
        
        return genre_tracks


# ============================================================
# GŁÓWNA FUNKCJA - PRZYKŁAD UŻYCIA
# ============================================================

if __name__ == '__main__':
    print("\n🎵 GTZAN-Genre Dataset Loader (mirdata 0.3.9)\n")
    
    # Inicjalizacja loadera
    loader = GTZANLoader()
    
    # Wypisz informacje
    loader.print_dataset_info()
    
    # Pobierz i załaduj dataset
    print("📥 Pobieranie i ładowanie dataset'u...\n")
    
    try:
        # Pobranie
        loader.download()
        
        # Załadowanie
        tracks = loader.load_tracks()
        
        # Statystyka
        print("\n" + "="*70)
        print("✅ DATASET ZAŁADOWANY POMYŚLNIE!")
        print("="*70)
        
        # Pierwszy track
        if tracks:
            track_ids = list(tracks.keys())
            first_track_id = track_ids[0]
            first_track = tracks[first_track_id]
            
            print(f"\n📝 Przykład pierwszego utworu:")
            print(f"  • Track ID: {first_track.track_id}")
            print(f"  • Gatunek: {first_track.genre}")
            print(f"  • Audio path: {first_track.audio_path}")
            
            # Audio info
            try:
                audio, sr = first_track.audio
                duration = len(audio) / sr
                print(f"  • Sample rate: {sr} Hz")
                print(f"  • Czas trwania: {duration:.1f}s")
                print(f"  • Liczba sampli: {len(audio):,}")
            except Exception as e:
                print(f"  ⚠️  Audio: {e}")
        
        # Statystyka gatunków
        loader.print_genre_statistics()
        
        print("="*70)
        print("🎉 Gotowe! Dataset jest dostępny w ~/.mirdata/GTZAN-Genre/")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
