# src/dataset_splitter.py
from sklearn.model_selection import train_test_split
import pandas as pd

class DatasetSplitter:
    """Podział na train(60%)/val(20%)/test(20%)"""
    
    def split(self, df: pd.DataFrame, random_state=42) -> tuple:
        """
        Args:
            df: DataFrame z cechami i kolumną 'genre'
        
        Returns:
            (train_df, val_df, test_df)
        """
        # 1. Train + Temp (60/40)
        train_temp, test = train_test_split(
            df, test_size=0.2, stratify=df['genre'], 
            random_state=random_state
        )
        
        # 2. Train + Val (75/25 z 60% = 60/20)
        train, val = train_test_split(
            train_temp, test_size=0.25, stratify=train_temp['genre'], 
            random_state=random_state
        )
        
        print(f"📊 Podział zbiorów:")
        print(f"  Train: {len(train):4} próbek ({len(train)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val):4} próbek ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test):4} próbek ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
