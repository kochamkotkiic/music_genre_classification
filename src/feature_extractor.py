# src/feature_extractor.py
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Wyodrębnia 59 cech z librosa"""
    
    def extract_single_file(self, file_path: str) -> Dict:
        """Wyodrębnij cechy z jednego pliku"""
        try:
            # Wczytaj audio (pierwsze 30s)
            y, sr = librosa.load(file_path, duration=30, sr=22050)
            
            # 1. MFCC (13 coef)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # 2. Spectral Centroid & Rolloff
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # 3. Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # 4. Spectral Bandwidth
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            # 5. Spectral Contrast (7 bands)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            
            # 6. Chroma (12 features)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            
            # 7. Tonnetz (6 features)
            tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1)
            
            # 8. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # 9. RMS Energy
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Połącz wszystkie cechy
            features = {
                # MFCC (13)
                **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(13)},
                # Spectral (4)
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'zcr': zcr,
                # Spectral Contrast (7)
                **{f'spectral_contrast_{i+1}': spectral_contrast[i] for i in range(7)},
                # Chroma (12)
                **{f'chroma_{i+1}': chroma[i] for i in range(12)},
                # Tonnetz (6)
                **{f'tonnetz_{i+1}': tonnetz[i] for i in range(6)},
                # Other (3)
                'tempo': tempo,
                'rms': rms
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Błąd przy {file_path}: {e}")
            return None
    
    def get_file_list(self, data_dir: Path) -> List[Dict]:
        """Pobierz listę plików audio z folderów gatunków"""
        files = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Folder {data_path} nie istnieje!")
        
        # Przeszukaj wszystkie podfoldery (gatunki)
        for genre_dir in sorted(data_path.iterdir()):
            if genre_dir.is_dir():
                genre = genre_dir.name
                # Znajdź wszystkie pliki audio
                for audio_file in sorted(genre_dir.glob('*.*')):
                    # Obsługuj różne formaty audio
                    if audio_file.suffix.lower() in ['.wav', '.au', '.mp3', '.flac', '.ogg']:
                        files.append({
                            'file_path': str(audio_file),
                            'filename': audio_file.name,
                            'genre': genre
                        })
        
        print(f"📁 Znaleziono {len(files)} plików audio w {len(set(f['genre'] for f in files))} gatunkach")
        return files
    
    def extract_from_directory(self, data_dir: Path) -> pd.DataFrame:
        """Wyodrębnij cechy ze wszystkich plików"""
        files = self.get_file_list(data_dir)
        
        print(f"🔄 Wyodrębniam cechy z {len(files)} plików...")
        results = []
        
        for i, file_info in enumerate(files):
            features = self.extract_single_file(file_info['file_path'])
            if features:
                features['genre'] = file_info['genre']
                features['filename'] = file_info['filename']
                results.append(features)
            
            if (i + 1) % 50 == 0:
                print(f"  ✓ {i+1}/{len(files)} plików przetworzonych")
        
        df = pd.DataFrame(results)
        print(f"✅ Wyodrębniono {len(df)} próbek x {len(df.columns)-2} cech")
        return df
