"""
CurriculumDataset — dataset stratificato per curriculum learning IMO.

Strategia anti-catastrophic forgetting
---------------------------------------
Il fine-tuning su problemi difficili (stage 5) tende a far dimenticare al modello
le costruzioni base. Per evitarlo, ogni fase mantiene una quota di esempi facili/medi
mescolati con gli esempi dello stage corrente.

Fasi di curriculum (definite in CURRICULUM_PHASES):
  Fase 0 — Fondamenta  : stage 1+2+3 ugualmente pesati (rinforzo base)
  Fase 1 — Ponte       : stage 4 (70%) + stage 3 (20%) + stage 1 (10%)
  Fase 2 — IMO push    : stage 5 (75%) + stage 2 (15%) + stage 1 (10%)
  Fase 3 — Consolidamento: stage 5 (85%) + stage 3 (10%) + stage 1 (5%)

Uso:
    ds = CurriculumDataset.for_phase(data_dir, phase=2, tokenizer=tok)
    loader = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Definizione delle fasi
# ---------------------------------------------------------------------------

# Ogni fase è un dict {nome_file: peso_relativo}
CURRICULUM_PHASES: List[Dict[str, float]] = [
    # Fase 0 — fondamenta (1 epoch consigliata)
    {
        "stage_1_very_easy":    1.0,
        "stage_2_massive_easy": 1.0,
        "stage_3_medium":       1.0,
    },
    # Fase 1 — ponte (1-2 epoch)
    {
        "stage_4_mixed":        0.70,
        "stage_3_medium":       0.20,
        "stage_1_very_easy":    0.10,
    },
    # Fase 2 — IMO push con retention (2-3 epoch)
    {
        "stage_5_difficult":    0.75,
        "stage_2_massive_easy": 0.15,
        "stage_1_very_easy":    0.10,
    },
    # Fase 3 — consolidamento finale (1-2 epoch)
    {
        "stage_5_difficult":    0.85,
        "stage_3_medium":       0.10,
        "stage_1_very_easy":    0.05,
    },
]


class CurriculumDataset(Dataset):
    """Dataset con mixing ponderato di stage multipli.

    Args:
        samples:    Lista di stringhe già combinate (question + ' ' + solution).
        tokenizer:  Tokenizer HF-compatibile.
        max_length: Lunghezza massima sequenza in token.
    """

    def __init__(
        self,
        samples: List[str],
        tokenizer,
        max_length: int = 1024,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_length - 2,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]
        ids = ids[: self.max_length]

        t = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids":      t,
            "attention_mask": torch.ones_like(t),
            "labels":         t.clone(),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Dynamic padding al massimo della batch."""
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L]      = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels[i, :L]         = b["labels"]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_phase(
        cls,
        data_dir: str,
        phase: int,
        tokenizer,
        max_length: int = 1024,
        max_samples_per_stage: Optional[int] = None,
        seed: int = 42,
    ) -> "CurriculumDataset":
        """Costruisce il dataset per la fase specificata.

        Args:
            data_dir:             Cartella contenente ``curriculum/stage_*.parquet``.
            phase:                Indice fase (0-3).
            tokenizer:            Tokenizer HF-compatibile.
            max_length:           Token massimi per sequenza.
            max_samples_per_stage: Cap per stage (utile in debug/test rapidi).
            seed:                 Seme per il sampling reproducibile.
        """
        import pandas as pd

        if phase < 0 or phase >= len(CURRICULUM_PHASES):
            raise ValueError(f"phase deve essere 0-{len(CURRICULUM_PHASES)-1}, ricevuto {phase}")

        phase_weights = CURRICULUM_PHASES[phase]
        data_dir = Path(data_dir)
        curriculum_dir = data_dir / "curriculum"

        rng = random.Random(seed)
        all_samples: List[str] = []

        # Calcola il numero totale di campioni dello stage più grande
        # per normalizzare i pesi in campioni assoluti.
        stage_sizes: Dict[str, int] = {}
        stage_data: Dict[str, List[str]] = {}

        for stage_name in phase_weights:
            path = curriculum_dir / f"{stage_name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Stage non trovato: {path}")

            df = pd.read_parquet(path)

            if "question" in df.columns and "solution" in df.columns:
                texts = (df["question"].astype(str) + " " + df["solution"].astype(str)).tolist()
            elif "text" in df.columns:
                texts = df["text"].astype(str).tolist()
            else:
                raise ValueError(f"Colonne non riconosciute in {path}: {list(df.columns)}")

            rng.shuffle(texts)
            if max_samples_per_stage is not None:
                texts = texts[:max_samples_per_stage]

            stage_sizes[stage_name] = len(texts)
            stage_data[stage_name] = texts

        # Stage di riferimento = il più grande (peso 1.0 o il peso più alto)
        max_stage = max(stage_sizes, key=lambda s: stage_sizes[s])
        ref_size = stage_sizes[max_stage]

        # Campiona ogni stage proporzionalmente al suo peso
        for stage_name, weight in phase_weights.items():
            texts = stage_data[stage_name]
            n_take = int(ref_size * weight)

            if n_take >= len(texts):
                # Ripeti il dataset se serve più di quanto disponibile
                repeated = (texts * (n_take // len(texts) + 1))[:n_take]
                all_samples.extend(repeated)
            else:
                all_samples.extend(texts[:n_take])

        rng.shuffle(all_samples)

        total = len(all_samples)
        print(f"📚 Fase {phase} — {total:,} campioni totali")
        for stage_name, weight in phase_weights.items():
            n = int(ref_size * weight)
            actual = min(n, stage_sizes[stage_name] * max(1, n // max(stage_sizes[stage_name], 1)))
            print(f"   {stage_name:30s}  peso={weight:.0%}  campioni≈{min(n, stage_sizes[stage_name] * (n // stage_sizes[stage_name] + 1)):,}")

        return cls(all_samples, tokenizer, max_length=max_length)

    @classmethod
    def for_custom_mix(
        cls,
        data_dir: str,
        weights: Dict[str, float],
        tokenizer,
        max_length: int = 1024,
        max_samples_per_stage: Optional[int] = None,
        seed: int = 42,
    ) -> "CurriculumDataset":
        """Costruisce un dataset con mixing personalizzato.

        Args:
            weights: dict {nome_stage: peso} es. {"stage_5_difficult": 0.8, "stage_1_very_easy": 0.2}
        """
        # Crea una fase temporanea con i pesi dati
        original = CURRICULUM_PHASES[:]
        CURRICULUM_PHASES.append(weights)
        try:
            ds = cls.for_phase(
                data_dir, len(CURRICULUM_PHASES) - 1, tokenizer,
                max_length=max_length,
                max_samples_per_stage=max_samples_per_stage,
                seed=seed,
            )
        finally:
            CURRICULUM_PHASES.pop()
        return ds


# ---------------------------------------------------------------------------
# DynamicCurriculumDataset — single-Trainer curriculum with epoch-based mixing
# ---------------------------------------------------------------------------

# epoch → ({stage: weight}, max_length)
# Epoch 0: foundation (equal mix, short context)
# Epoch 1: bridge to harder problems
# Epoch 2-3: IMO push with retention
# Epoch 4-5: consolidation at full context
EPOCH_SCHEDULE = [
    ({"stage_1_very_easy": 1.0, "stage_2_massive_easy": 1.0, "stage_3_medium": 1.0}, 512),
    ({"stage_4_mixed": 0.70,    "stage_3_medium": 0.20,       "stage_1_very_easy": 0.10}, 640),
    ({"stage_5_difficult": 0.75, "stage_2_massive_easy": 0.15, "stage_1_very_easy": 0.10}, 896),
    ({"stage_5_difficult": 0.75, "stage_2_massive_easy": 0.15, "stage_1_very_easy": 0.10}, 896),
    ({"stage_5_difficult": 0.85, "stage_3_medium": 0.10,       "stage_1_very_easy": 0.05}, 1024),
    ({"stage_5_difficult": 0.85, "stage_3_medium": 0.10,       "stage_1_very_easy": 0.05}, 1024),
]


class DynamicCurriculumDataset(Dataset):
    """Dataset curriculum ricostruito dinamicamente ad ogni epoca.

    Progettato per un singolo Trainer.train() multi-epoch.  Al cambio di epoca
    il Callback chiama set_epoch() che rimescola il dataset con le proporzioni
    appropriate senza ricreare il DataLoader.

    La dimensione del dataset è fissata a quella dell'epoca 0 così che il
    LR scheduler (calcolato una sola volta all'inizio) rimanga coerente.

    Args:
        stage_data:   Dict {nome_stage: List[str]} pre-caricato da parquet.
        tokenizer:    Tokenizer HF-compatibile.
        max_pos:      max_position_embeddings del modello (upper bound hard).
        seed:         Seme per la riproducibilità.
    """

    def __init__(
        self,
        stage_data: Dict[str, List[str]],
        tokenizer,
        max_pos: int = 1024,
        seed: int = 42,
    ):
        self.stage_data = stage_data
        self.tokenizer = tokenizer
        self.max_pos = max_pos
        self.seed = seed

        self.samples: List[str] = []
        self.max_length: int = 512

        # Build epoch 0 to determine stable target size
        self._build(0)
        self._target_size = len(self.samples)

    # ------------------------------------------------------------------

    def _build(self, epoch: int) -> None:
        weights, raw_max_length = EPOCH_SCHEDULE[min(epoch, len(EPOCH_SCHEDULE) - 1)]
        self.max_length = min(raw_max_length, self.max_pos)

        rng = random.Random(self.seed + epoch * 31337)

        available = {name: data for name, data in self.stage_data.items() if name in weights}
        if not available:
            raise ValueError(f"Nessuno stage disponibile per epoch {epoch}. "
                             f"Richiesti: {list(weights)}, presenti: {list(self.stage_data)}")

        ref_size = max(len(v) for v in available.values())
        all_samples: List[str] = []

        for stage_name, weight in weights.items():
            if stage_name not in available:
                continue
            texts = list(available[stage_name])
            rng.shuffle(texts)
            n_take = int(ref_size * weight)
            if n_take >= len(texts):
                repeated = (texts * (n_take // len(texts) + 1))[:n_take]
                all_samples.extend(repeated)
            else:
                all_samples.extend(texts[:n_take])

        rng.shuffle(all_samples)

        # Trim or pad to _target_size (only after the first build sets it)
        if hasattr(self, "_target_size"):
            target = self._target_size
            while len(all_samples) < target:
                all_samples.extend(rng.choices(all_samples, k=target - len(all_samples)))
            all_samples = all_samples[:target]

        self.samples = all_samples

    def set_epoch(self, epoch: int) -> None:
        """Ricostruisce il dataset per l'epoca specificata.  Chiamato dal callback."""
        self._build(epoch)
        weights, _ = EPOCH_SCHEDULE[min(epoch, len(EPOCH_SCHEDULE) - 1)]
        mix = ", ".join(f"{s.split('_')[1]}:{w:.0%}" for s, w in weights.items())
        print(f"\n🔄 Epoca {epoch} — max_length={self.max_length}  mix={{{mix}}}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._target_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length - 2,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]
        ids = ids[: self.max_length]
        t = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids":      t,
            "attention_mask": torch.ones_like(t),
            "labels":         t.clone(),
        }

    # ------------------------------------------------------------------

    @classmethod
    def load_from_dir(
        cls,
        data_dir: str,
        tokenizer,
        max_pos: int = 1024,
        max_samples_per_stage: Optional[int] = None,
        seed: int = 42,
    ) -> "DynamicCurriculumDataset":
        """Carica tutti gli stage da data_dir/curriculum/ e restituisce il dataset.

        Args:
            data_dir:               Cartella contenente ``curriculum/stage_*.parquet``.
            tokenizer:              Tokenizer HF-compatibile.
            max_pos:                max_position_embeddings del modello.
            max_samples_per_stage:  Cap per stage (debug).
            seed:                   Seme.
        """
        import pandas as pd

        curriculum_dir = Path(data_dir) / "curriculum"
        all_stage_names: set = set()
        for sched_weights, _ in EPOCH_SCHEDULE:
            all_stage_names.update(sched_weights.keys())

        stage_data: Dict[str, List[str]] = {}
        rng = random.Random(seed)

        for stage_name in sorted(all_stage_names):
            path = curriculum_dir / f"{stage_name}.parquet"
            if not path.exists():
                print(f"⚠️  Stage non trovato, verrà ignorato: {path}")
                continue

            df = pd.read_parquet(path)
            if "question" in df.columns and "solution" in df.columns:
                texts = (df["question"].astype(str) + " " + df["solution"].astype(str)).tolist()
            elif "text" in df.columns:
                texts = df["text"].astype(str).tolist()
            else:
                raise ValueError(f"Colonne non riconosciute in {path}: {list(df.columns)}")

            rng.shuffle(texts)
            if max_samples_per_stage is not None:
                texts = texts[:max_samples_per_stage]

            stage_data[stage_name] = texts
            print(f"   📂 {stage_name:35s} {len(texts):,} campioni")

        return cls(stage_data, tokenizer, max_pos=max_pos, seed=seed)

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Dynamic padding al massimo della batch."""
        max_len = max(b["input_ids"].shape[0] for b in batch)
        input_ids      = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels         = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, :L]      = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            labels[i, :L]         = b["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
