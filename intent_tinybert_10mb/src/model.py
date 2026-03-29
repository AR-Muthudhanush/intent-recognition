import torch
import torch.nn as nn
from transformers import AutoModel

CLS_TASKS = ["intent", "target_type", "attribute", "spatial_relation", "position"]

class TinyBertMultiTaskSpan(nn.Module):
    """
    TinyBERT encoder
    - 5 classification heads: intent, target_type, attribute, spatial_relation, position
    - 1 span head for spatial_reference: start/end over sequence tokens
    """
    def __init__(self, base_model_name: str, num_labels: dict):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.drop = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            t: nn.Linear(hidden, num_labels[t]) for t in CLS_TASKS
        })

        # token-level span head (start/end)
        self.qa_outputs = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        seq = out.last_hidden_state           # [B, L, H]
        pooled = self.drop(seq[:, 0])         # [B, H]

        cls_logits = {t: self.heads[t](pooled) for t in CLS_TASKS}

        qa = self.qa_outputs(seq)             # [B, L, 2]
        start_logits = qa[:, :, 0]            # [B, L]
        end_logits   = qa[:, :, 1]            # [B, L]

        return cls_logits, start_logits, end_logits
