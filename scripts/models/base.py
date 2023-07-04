from abc import ABC, abstractmethod
from pathlib import Path
import os

import torch
from avalanche.models import MultiHeadClassifier, MultiTaskModule


class BaseModel(ABC, MultiTaskModule):
    def __init__(
        self, seed, output_hidden, is_multihead, in_features, out_features, device
    ):
        super().__init__()
        self.seed = seed
        self.output_hidden = output_hidden
        self.is_multihead = is_multihead
        self.device = device

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # If using multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.is_multihead:
            self.multihead_classifier = MultiHeadClassifier(
                in_features=in_features, initial_out_features=out_features
            ).to(device)

    def toggle_hidden(self, output_hidden):
        self.output_hidden = output_hidden

    def adapt_model(self, experiences):
        if self.is_multihead:
            for exp in experiences:
                self.multihead_classifier.adaptation(exp)

    def get_dir_name(self, parent_dir):
        # Build a consistent directory name
        return f"{parent_dir}/{self.__class__.__name__}_seed{self.seed}"

    def save_weights(self, parent_dir):
        # Get dir name
        dir_name = self.get_dir_name(parent_dir)

        # Call model specific implementation
        self._save_weights_impl(dir_name)

        # Handle multihead
        if self.is_multihead:
            path = Path(dir_name, "classifier.pt")
            torch.save(self.multihead_classifier.state_dict(), path)
            print(f"Classifier saved at: {path}")

    def load_weights(self, parent_dir):
        # Get dir name
        dir_name = self.get_dir_name(parent_dir)

        # Call model specific implementation
        self._load_weights_impl(dir_name)

        # Handle multihead
        if self.is_multihead:
            path = Path(dir_name, "classifier.pt")
            self.multihead_classifier.load_state_dict(torch.load(path), strict=False)
            self.multihead_classifier.to(self.device)

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def hidden_layers(self):
        pass

    @property
    @abstractmethod
    def num_hidden(self):
        pass

    @abstractmethod
    def _save_weights_impl(self, dir_name):
        pass

    @abstractmethod
    def _load_weights_impl(self, dir_name):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def get_hidden_layer(self):
        pass
