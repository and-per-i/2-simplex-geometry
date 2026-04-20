import os
import sys
import torch
import shutil
from transformers import Trainer, TrainingArguments, TrainerCallback

# Aggiunge src al path per caricare i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.data.finetune_dataset import FinetuneDataset
from tokenizer.hf_tokenizer import load_tokenizer

from collections import deque

# --- 1. Plateau Detection Callback ---
class PlateauHandlerCallback(TrainerCallback):
    """
    Rileva il plateau basandosi sulla MEDIA MOBILE della loss.
    Più robusto contro il rumore dei singoli batch.
    """
    def __init__(self, window_size=20, threshold=0.001, patience=15):
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience
        
        self.loss_window = deque(maxlen=window_size)
        self.best_avg_loss = float('inf')
        self.patience_counter = 0
        self.should_stop_for_plateau = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            current_loss = logs["loss"]
            self.loss_window.append(current_loss)

            # Aspetta di avere abbastanza dati nella finestra prima di decidere
            if len(self.loss_window) < self.window_size:
                return

            current_avg_loss = sum(self.loss_window) / len(self.loss_window)

            # Se la media attuale è significativamente migliore della migliore precedente
            if current_avg_loss < self.best_avg_loss - self.threshold:
                self.best_avg_loss = current_avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                # Log di debug ogni 50 step per vedere lo stato del monitor
                if state.global_step % 50 == 0:
                    print(f"   [Monitor] Media Finestra: {current_avg_loss:.4f} | Migliore: {self.best_avg_loss:.4f} | Patience: {self.patience_counter}/{self.patience}")
            
            if self.patience_counter >= self.patience:
                print(f"\n🛑 TREND PLATEAU DETECTED at step {state.global_step}!")
                print(f"   La media degli ultimi {self.window_size} log ({current_avg_loss:.4f}) non migliora rispetto a {self.best_avg_loss:.4f}")
                self.should_stop_for_plateau = True
                control.should_training_stop = True

# --- 2. Curriculum Configuration ---
# Definisci qui l'ordine dei dataset da affrontare
CURRICULUM = [
    {"level": "very_easy", "path": "data/very_easy/very_easy_500k.parquet"},
    {"level": "easy",      "path": "data/easy/easy_medium_500k.parquet"},
    {"level": "mixed",     "path": "data/mixed/curriculum_2M.parquet"},
    {"level": "difficult", "path": "data/difficult/balanced_dataset_part_1.parquet"},
    {"level": "difficult_p2", "path": "data/difficult/balanced_dataset_part_2.parquet"}
]

def run_curriculum():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Percorsi base
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/weights/geometry.757.model"))
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # --- 3. Custom Data Collator per il Padding Dinamico ---
    def custom_collator(features):
        batch = {}
        max_len = max(len(f["input_ids"]) for f in features)
        
        for key in ["input_ids", "attention_mask", "labels"]:
            padded_seqs = []
            for f in features:
                seq = f[key]
                # 0 per padding, -100 per labels (così la CrossEntropy li ignora)
                pad_val = 0 if key != "labels" else -100
                padded = seq + [pad_val] * (max_len - len(seq))
                padded_seqs.append(padded)
            batch[key] = torch.tensor(padded_seqs, dtype=torch.long)
        return batch

    for stage in CURRICULUM:
        level = stage["level"]
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", stage["path"]))
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset non trovato per {level}: {dataset_path}. Salto...")
            continue

        attempts = 0
        max_attempts_per_level = 2
        base_lr = 5e-5

        while attempts < max_attempts_per_level:
            # Calcolo LR: Dimezza ad ogni tentativo sullo stesso livello
            lr = base_lr / (2 ** attempts) 
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../checkpoints_curriculum/{level}_try_{attempts}"))
            
            print(f"\n" + "="*70)
            print(f"🚀 STAGE: {level.upper()} | TENTATIVO: {attempts+1}/{max_attempts_per_level} | LR: {lr:.2e}")
            print(f"📂 Dataset: {os.path.basename(dataset_path)}")
            print("="*70)

            # Carica Modello e Config
            config = StudentConfig.from_pretrained(current_model_path)
            model = StudentForCausalLM.from_pretrained(
                current_model_path,
                config=config,
                torch_dtype=torch.bfloat16
            ).to(device)

            # Carica Dataset
            train_dataset = FinetuneDataset(
                path=dataset_path,
                tokenizer=tokenizer,
                max_length=config.max_position_embeddings
            )

            plateau_callback = PlateauHandlerCallback(window_size=20, threshold=0.001, patience=25)

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=64,
                gradient_accumulation_steps=2,
                learning_rate=lr,
                num_train_epochs=1,
                bf16=True,
                save_strategy="steps",
                save_steps=1000,
                logging_steps=10,
                report_to="none",
                optim="adamw_torch_fused",
                save_total_limit=2,
                remove_unused_columns=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=custom_collator,
                callbacks=[plateau_callback]
            )

            # Avvia Training
            trainer.train()

            # Dopo il training, decidiamo cosa fare
            if plateau_callback.should_stop_for_plateau:
                print(f"♻️ Plateau rilevato. Tentativo {attempts+1} completato.")
                
                # Cerchiamo l'ultimo checkpoint salvato in questa cartella per ripartire
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                    current_model_path = os.path.join(output_dir, latest)
                    print(f"💾 Riparerò dal prossimo tentativo usando: {latest}")
                
                attempts += 1
            else:
                # Training finito senza plateau -> Passiamo al livello successivo
                print(f"✅ Livello {level} completato senza plateau bloccante!")
                current_model_path = output_dir
                break 

    print("\n" + "🏆"*10)
    print("CURRICULUM TRAINING COMPLETATO!")
    print("🏆"*10)

if __name__ == "__main__":
    run_curriculum()
