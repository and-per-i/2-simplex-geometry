import os
import sys
import torch
import shutil
from transformers import Trainer, TrainingArguments, TrainerCallback

# Aggiunge src e symbolic al path per caricare i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/symbolic")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from src.data.system1_dataset import System1Dataset
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
# Rilevamento automatico dei file nella cartella data/curriculum
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/curriculum"))

def get_curriculum():
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Directory dati non trovata: {DATA_DIR}")
        return []
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
    curriculum = []
    for f in files:
        level_name = f.replace(".parquet", "")
        curriculum.append({"level": level_name, "path": os.path.join("data/curriculum", f)})
    
    print(f"📚 Curriculum rilevato ({len(curriculum)} stage):")
    for stage in curriculum:
        print(f"   - {stage['level']}")
    return curriculum

CURRICULUM = get_curriculum()

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

    # --- Punto di Partenza: Auto-rilevamento dell'ULTIMO checkpoint assoluto ---
    checkpoint_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints"))
    current_model_path = None

    if os.path.exists(checkpoint_root):
        all_checkpoints = []
        for root, dirs, files in os.walk(checkpoint_root):
            for d in dirs:
                if d.startswith("checkpoint-"):
                    full_path = os.path.join(root, d)
                    all_checkpoints.append(full_path)
        
        if all_checkpoints:
            # Prende il checkpoint più recente basato sulla data di modifica
            current_model_path = max(all_checkpoints, key=os.path.getmtime)
            print(f"🔄 Auto-detected LATEST checkpoint: {current_model_path}")

    # Fallback: Se non trova nulla, usa un percorso base o fallisce con grazia
    if not current_model_path or not os.path.exists(current_model_path):
        print("⚠️ No checkpoints found in 'checkpoints/'. Ensure you have a base model to start from.")
        # Se hai un modello base specifico, mettilo qui come fallback estremo
        return 

    print(f"🎯 Starting training from: {current_model_path}")

    for stage in CURRICULUM:
        level = stage["level"]
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", stage["path"]))
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset non trovato per {level}: {dataset_path}. Salto...")
            continue

        attempts = 0
        max_attempts_per_level = 2
        base_lr = 2.0e-5

        while attempts < max_attempts_per_level:
            # Calcolo LR: Dimezza ad ogni tentativo sullo stesso livello
            lr = base_lr / (2 ** attempts) 
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../checkpoints/{level}_try_{attempts}"))
            
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
            train_dataset = System1Dataset(
                path=dataset_path,
                tokenizer=tokenizer,
                max_length=config.max_position_embeddings
            )

            plateau_callback = PlateauHandlerCallback(window_size=20, threshold=0.001, patience=100)

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=lr,
                lr_scheduler_type="cosine",
                warmup_steps=100,
                num_train_epochs=1,
                bf16=True,
                save_strategy="steps",
                save_steps=200,
                logging_steps=10,
                report_to="none",
                optim="adamw_torch_fused",
                save_total_limit=5,
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
                print(f"♻️ Plateau rilevato. Tentativo {attempts+1}/{max_attempts_per_level} completato.")
                
                # Cerchiamo l'ultimo checkpoint salvato in questa cartella per ripartire
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                    current_model_path = os.path.join(output_dir, latest)
                    print(f"💾 Checkpoint rilevato: {latest}")
                
                attempts += 1
                
                # Se abbiamo esaurito i tentativi, forziamo il passaggio al livello successivo
                # usando comunque l'ultimo checkpoint trovato
                if attempts >= max_attempts_per_level:
                    print(f"⏩ Massimi tentativi raggiunti per {level}. Passo al prossimo stage con i pesi attuali.")
            else:
                # Training finito senza plateau (epoca completata) -> Passiamo al livello successivo
                print(f"✅ Livello {level} completato (Epoca conclusa)!")
                current_model_path = output_dir
                break 

    print("\n" + "🏆"*10)
    print("CURRICULUM TRAINING COMPLETATO!")
    print("🏆"*10)

if __name__ == "__main__":
    run_curriculum()
