# custom_peft_classification_model.py
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomPeftClassificationModel(nn.Module):
    def __init__(self, peft_model, num_labels=2, signal_weight=1.0):
        super().__init__()
        self.peft_model = peft_model

        # Try to read hidden_size from config; fallback to common names
        hidden_size = getattr(peft_model.config, "hidden_size", None)
        if hidden_size is None:
            # Some models use d_model
            hidden_size = getattr(peft_model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden size from peft_model.config")

        # classifier head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # loss will be created per-device in forward (so weights on correct device)
        self._base_signal_weight = float(signal_weight)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """
        Forward any gradient_checkpointing calls to the underlying model.
        Accept arbitrary kwargs to be compatible with Trainer behavior.
        """
        if hasattr(self.peft_model, "gradient_checkpointing_enable"):
            # Some models accept kwargs, some don't. Call safely.
            try:
                self.peft_model.gradient_checkpointing_enable(*args, **kwargs)
            except TypeError:
                # older/newer signature - call without kwargs
                self.peft_model.gradient_checkpointing_enable()

    def to(self, *args, **kwargs):
        # Ensure classifier moves with model.to(...)
        self = super().to(*args, **kwargs)
        self.classifier.to(next(self.peft_model.parameters()).device)
        return self

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward the PEFT model, extract last hidden state, apply classifier head.
        Return a SequenceClassifierOutput with 'loss' (if labels given) and 'logits'.
        """
        # Ensure PEFT model returns hidden states
        kwargs.setdefault("output_hidden_states", True)

        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # outputs.hidden_states should exist
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("PEFT model did not return hidden_states. Ensure output_hidden_states=True.")

        # last token hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_size]
        logits = self.classifier(last_hidden)              # [batch, num_labels]

        loss = None
        if labels is not None:
            # labels expected shaped [batch, seq_len] with -100 except last token
            if labels.dim() == 2:
                labels_final = labels[:, -1].to(logits.device)
            else:
                # maybe user supplied [batch] already
                labels_final = labels.to(logits.device)

            # build class weights on the correct device
            class_weights = torch.tensor([1.0, self._base_signal_weight], device=logits.device, dtype=torch.float32)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits, labels_final.long())

        # Return a ModelOutput-like object; Trainer expects this shape
        return SequenceClassifierOutput(loss=loss, logits=logits)
