#!/usr/bin/env python3
################################################################################
# scripts/download_data.py
# Skrypt do pobrania GTZAN-Genre Dataset'u
################################################################################

import sys
from pathlib import Path

# Dodaj src do path'u
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import GTZANLoader
import argparse


def main():
    """
    Główna funkcja pobierania dataset'u
    """
    parser = argparse.ArgumentParser(
        description='Pobierz GTZAN-Genre dataset (via mirdata 0.3.9)'
    )
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Tylko wyświetl informacje o dataset\'cie (nie pobieraj)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Waliduj dataset po pobraniu'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎵 GTZAN-GENRE DATASET DOWNLOADER (mirdata 0.3.9)")
    print("="*70 + "\n")
    
    try:
        # Inicjalizacja loadera
        print("🔧 Inicjalizacja loadera...\n")
        loader = GTZANLoader()
        
        # Wyświetl informacje
        print("\n📊 INFORMACJE O DATASET'CIE:\n")
        loader.print_dataset_info()
        
        # Jeśli --info-only, zakończ
        if args.info_only:
            print("✅ Informacje wyświetlone.\n")
            print("Aby pobrać dane, uruchom:")
            print("  python scripts/download_data.py\n")
            return
        
        # Pobierz dataset
        print("\n📥 POBIERANIE DATASET'U...\n")
        print("⏳ To może trwać 5-20 minut (dataset ~1.2 GB)\n")
        print("-" * 70 + "\n")
        
        # Pobranie (może to trwać długo)
        loader.download()
        
        # Załadowanie
        print("\n📂 Ładowanie utworów...")
        tracks = loader.load_tracks()
        
        # Statystyka
        print("\n" + "="*70)
        print("✅ DATASET POBRANY I ZAŁADOWANY POMYŚLNIE!")
        print("="*70)
        print(f"📊 Razem utworów: {len(tracks)}")
        
        # Informacje o pierwszym utworze
        if tracks:
            track_ids = list(tracks.keys())
            first_track_id = track_ids[0]
            first_track = tracks[first_track_id]
            
            print(f"\n📝 Przykład pierwszego utworu:")
            print(f"  • Track ID: {first_track.track_id}")
            print(f"  • Gatunek: {first_track.genre}")
            print(f"  • Audio path: {first_track.audio_path}")
            
            # Spróbuj załadować audio
            try:
                audio, sr = first_track.audio
                duration = len(audio) / sr
                print(f"  • Sample rate: {sr} Hz")
                print(f"  • Czas trwania: {duration:.1f}s")
                print(f"  • Liczba sampli: {len(audio):,}")
            except Exception as e:
                print(f"  ⚠️  Nie można załadować audio: {e}")
        
        # Rozkład gatunków
        print("\n🎼 Rozkład gatunków:")
        loader.print_genre_statistics()
        
        # Opcjonalna walidacja
        if args.validate:
            print("\n🔍 Walidacja dataset'u...")
            missing, invalid = loader.validate_dataset()
        
        print("\n" + "="*70)
        print("🎉 Gotowe! Dataset jest dostępny w: ~/.mirdata/GTZAN-Genre/")
        print("="*70)
        print("\n💡 Następnie możesz:")
        print("  1. Wyodrębniać cechy audio (MFCC, spectral, etc.)")
        print("  2. Trenować model klasyfikacji")
        print("  3. Oceniać wydajność modelu")
        print("\nPatrz: Zadanie 1 w project_specification.yml\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pobieranie przerwane przez użytkownika\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ BŁĄD: {e}")
        print("\n💡 Wskazówki:")
        print("  1. Upewnij się że masz dostęp do internetu")
        print("  2. Sprawdź czy wystarczająco miejsca (~1.5 GB)")
        print("  3. Jeśli pobranie się zawiesza, przerwij (Ctrl+C) i spróbuj ponownie")
        print("  4. Sprawdzenie dostępnych dataset'ów:")
        print("     python -c \"import mirdata; print(mirdata.list_datasets())\"")
        
        import traceback
        print("\n📋 Szczegóły błędu:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
