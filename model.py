import torch
import torch.nn as nn
from transformers import EsmModel


class ESMClassifier(nn.Module):
    """
    ESM-2 backbone + classification head for protein subcellular localization.

    Architecture:
    - Loads a pre-trained ESM-2 encoder (Meta, trained on 250M protein sequences).
    - Freezes the embedding layer and the first `freeze_layers` encoder blocks to
      preserve general protein representations.
    - Fine-tunes the remaining encoder layers and a lightweight classification head.
    - Uses the <cls> token (position 0) as the sequence-level representation.

    Args:
        model_name    : HuggingFace model ID, e.g. "facebook/esm2_t12_35M_UR50D"
        num_classes   : Number of subcellular localization classes.
        freeze_layers : Number of ESM encoder layers to freeze (counted from layer 0).
    """

    def __init__(self, model_name: str, num_classes: int, freeze_layers: int = 6):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)

        # Freeze embedding layer
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False

        # Freeze first `freeze_layers` transformer blocks
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = self.esm.config.hidden_size  # 480 for esm2_t12_35M_UR50D

        # Dropout 0.3 to reduce the train/val overfitting gap
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = outputs.last_hidden_state[:, 0]  # <cls> token → [batch, hidden]
        return self.classifier(cls_out)
