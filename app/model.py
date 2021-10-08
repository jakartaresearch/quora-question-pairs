from abc import ABC, abstractmethod
import torch
import warnings
warnings.filterwarnings('ignore')

from transformers import BertForSequenceClassification, BertTokenizer


class ParaphraseIdentifier_BaseModel(ABC):
    """Abstract base class for Paraphrase Identifier model."""
    
    def __init__(self, model_pth):
        self._pth = model_pth
        self._model, self._tokenizer = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Return BERT model."""
        
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer


class ParaphraseIdentifier_Model(ParaphraseIdentifier_BaseModel):
    def _load_model(self):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        model = BertForSequenceClassification.from_pretrained(self._pth)
        tokenizer = BertTokenizer.from_pretrained(self._pth)
        model.to(device)
        model.eval()
        return model, tokenizer
