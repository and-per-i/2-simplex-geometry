"""
HuggingFace-compatible tokenizer wrapper per il tokenizer SentencePiece di AlphaGeometry.

Problema: il Trainer HF richiede un tokenizer che erediti da PreTrainedTokenizer.
L'implementazione originale (tokenizer_client.py) usa un subprocess e non è compatibile.

Soluzione: questo wrapper carica direttamente la libreria sentencepiece e implementa
l'interfaccia PreTrainedTokenizer in modo nativo.
"""

import os
from typing import Dict, List, Optional

import sentencepiece as spm
from transformers import PreTrainedTokenizer


# Token speciali standard — IDs da verificare con il modello SP effettivo
_SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
}


class AlphaGeometryHFTokenizer(PreTrainedTokenizer):
    """
    Wrapper HuggingFace per il tokenizer SentencePiece di AlphaGeometry.

    Compatibile con:
    - Trainer.tokenizer
    - DataCollatorForLanguageModeling
    - tokenizer(text, return_tensors="pt")

    Uso:
        tokenizer = AlphaGeometryHFTokenizer.from_pretrained("tokenizer/")
        # oppure direttamente:
        tokenizer = AlphaGeometryHFTokenizer("tokenizer/weights/geometry.757.model")
    """

    vocab_files_names = {"vocab_file": "geometry.757.model"}

    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sp_model_kwargs: Optional[Dict] = None,
        add_prefix_space: bool = False,
        **kwargs,
    ):
        self.sp_model_kwargs = sp_model_kwargs or {}
        self.vocab_file = vocab_file
        self.add_prefix_space = add_prefix_space

        # Carica il modello SentencePiece direttamente (no subprocess)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            sp_model_kwargs=sp_model_kwargs,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Proprietà obbligatorie di PreTrainedTokenizer
    # -------------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Ritorna 757 per geometry.757.model."""
        return self.sp_model.GetPieceSize()

    def get_vocab(self) -> Dict[str, int]:
        vocab = {
            self.sp_model.IdToPiece(i): i for i in range(self.vocab_size)
        }
        # Aggiungi i token speciali aggiunti (quelli fuori dall'SP)
        vocab.update(self.added_tokens_encoder)
        return vocab

    # -------------------------------------------------------------------------
    # Metodi di tokenizzazione
    # -------------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Converte una stringa in una lista di piece-string."""
        if self.add_prefix_space:
            text = " " + text
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converte un piece-string nel suo ID intero."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converte un ID intero nel suo piece-string."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Riconverte una lista di piece-strings in testo (decode)."""
        return self.sp_model.DecodePieces(tokens)

    # -------------------------------------------------------------------------
    # Salvataggio del vocab (necessario per save_pretrained)
    # -------------------------------------------------------------------------

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        """Copia il file .model nella directory di destinazione."""
        import shutil

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        out_name = (
            f"{filename_prefix}-geometry.757.model"
            if filename_prefix
            else "geometry.757.model"
        )
        out_path = os.path.join(save_directory, out_name)

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_path):
            shutil.copyfile(self.vocab_file, out_path)

        return (out_path,)

    # -------------------------------------------------------------------------
    # Metodi di stato (necessari per pickle/deepcopy nel Trainer)
    # -------------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None  # non serializzabile
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)


# -------------------------------------------------------------------------
# Factory helper
# -------------------------------------------------------------------------

def load_tokenizer(
    model_path: str = "tokenizer/weights/geometry.757.model",
) -> AlphaGeometryHFTokenizer:
    """
    Carica il tokenizer AlphaGeometry in modalità HF-compatible.

    Args:
        model_path: percorso assoluto o relativo al file .model SP.

    Returns:
        AlphaGeometryHFTokenizer pronto per essere usato con Trainer.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Tokenizer model non trovato: {model_path}\n"
            "Verifica che il file geometry.757.model esista."
        )
    return AlphaGeometryHFTokenizer(vocab_file=model_path)
