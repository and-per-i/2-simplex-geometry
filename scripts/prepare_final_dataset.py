import pandas as pd
import os
import shutil

# Configurazione
SOURCE_DATA = {
    "stage_1_very_easy": "data/very_easy/very_easy_500k.parquet",
    "stage_2_massive_easy": "data/easy/very_easy_3000k.parquet",
    "stage_3_medium": "data/easy/easy_medium_500k.parquet",
    "stage_4_mixed": "data/mixed/curriculum_2M.parquet",
    "stage_5_difficult": ["data/difficult/balanced_dataset_part_1.parquet", "data/difficult/balanced_dataset_part_2.parquet"]
}

OUTPUT_DIR = "data/curriculum"

def filter_and_save():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created {OUTPUT_DIR}")

    for stage_name, paths in SOURCE_DATA.items():
        if isinstance(paths, str):
            paths = [paths]
        
        dfs = []
        for p in paths:
            if os.path.exists(p):
                print(f"📖 Loading {p}...")
                df = pd.read_parquet(p)
                # Filtra: tieni solo dove c'è <aux>
                mask = df["solution"].str.contains("<aux>", na=False)
                filtered_df = df[mask].copy()
                dfs.append(filtered_df)
                print(f"   - Found {len(filtered_df)} / {len(df)} samples with aux constructions.")
            else:
                print(f"⚠️ Warning: {p} not found.")

        if dfs:
            final_df = pd.concat(dfs)
            output_path = os.path.join(OUTPUT_DIR, f"{stage_name}.parquet")
            print(f"💾 Saving {len(final_df)} samples to {output_path}...")
            final_df.to_parquet(output_path, index=False)
            print(f"✅ {stage_name} complete.")
        else:
            print(f"❌ No data found for {stage_name}")

if __name__ == "__main__":
    filter_and_save()
