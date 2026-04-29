"""
StudentConfig — PretrainedConfig per il modello student custom (2-Simplex LM).

Questo config è compatibile con HuggingFace Trainer e permette di:
- Salvare/caricare il modello con .save_pretrained() / .from_pretrained()
- Specificare l'architettura in modo riproducibile
"""

from transformers import PretrainedConfig


class StudentConfig(PretrainedConfig):
    """
    Configuration class per il modello Student (architettura 2-Simplex Transformer).

    Attributi:
        vocab_size (int): Dimensione del vocabolario. DEVE corrispondere al tokenizer
                          (geometry.757.model → 757).
        hidden_size (int): Dimensione degli hidden states del transformer.
        num_hidden_layers (int): Numero di layer Transformer.
        num_attention_heads (int): Numero di teste di attenzione.
        intermediate_size (int): Dimensione del feed-forward layer.
        max_position_embeddings (int): Lunghezza massima della sequenza.
        dropout_prob (float): Dropout applicato ad attention e FFN.
        use_simplex_attention (bool): Se True, usa 2-Simplex attention.
        w1 (int): Dimensione della finestra scorrevole per K1/V1.
        w2 (int): Dimensione della finestra scorrevole per K2/V2.
        initializer_range (float): Std per l'inizializzazione dei pesi.
        pad_token_id (int): ID del token di padding.
        bos_token_id (int): ID del token di inizio sequenza.
        eos_token_id (int): ID del token di fine sequenza.
    """

    # Identificatore univoco dell'architettura — necessario per from_pretrained()
    model_type = "student_2simplex"

    def __init__(
        self,
        vocab_size: int = 757,          # Allineato al tokenizer AlphaGeometry
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 1024,
        dropout_prob: float = 0.1,
        use_simplex_attention: bool = False,
        use_triton: bool = True,
        w1: int = 8,
        w2: int = 8,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Validazione critica: il vocab_size determina la shape dei logits.
        # Teacher e Student DEVONO condividere lo stesso vocab_size.
        assert vocab_size > 0, "vocab_size deve essere positivo"
        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) deve essere divisibile per "
            f"num_attention_heads ({num_attention_heads})"
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.use_simplex_attention = use_simplex_attention
        self.use_triton = use_triton
        self.w1 = w1
        self.w2 = w2
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


    @property
    def head_dim(self) -> int:
        """Dimensione di ogni testa di attenzione."""
        return self.hidden_size // self.num_attention_heads
