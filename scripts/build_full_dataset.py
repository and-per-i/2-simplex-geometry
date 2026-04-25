import pandas as pd
from pathlib import Path

def main():
    print("Costruzione del MEGA-dataset per Fine-Tuning Causale...")
    
    data_dir = Path("data/curriculum")
    stages = [
        "stage_1_very_easy.parquet",
        "stage_2_massive_easy.parquet",
        "stage_3_medium.parquet",
        "stage_4_mixed.parquet",
        "stage_5_difficult.parquet"
    ]
    
    dfs = []
    total_samples = 0
    
    for stage in stages:
        file_path = data_dir / stage
        if not file_path.exists():
            print(f"⚠️  Saltato {stage} (non trovato)")
            continue
            
        df = pd.read_parquet(file_path)
        avail = len(df)
        
        # PRENDIAMO TUTTI I CAMPIONI, non più solo 5000
        dfs.append(df)
        total_samples += avail
        
        print(f"✅ {stage}: prelevati TUTTI i {avail:,} campioni")

    # Unisci tutti i dataframe
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle finale del dataset per non avere blocchi ordinati per difficoltà
    print("\nMescolando mezzo milione di esempi, attendere...")
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = Path("data/finetune_raw_full.parquet")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    final_df.to_parquet(output_path, index=False)
    
    print("="*50)
    print(f"🎉 MEGA-Dataset finale creato: {output_path}")
    print(f"   Totale campioni: {len(final_df):,}")
    print("="*50)

if __name__ == "__main__":
    main()
