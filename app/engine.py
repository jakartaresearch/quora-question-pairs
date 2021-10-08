from abc import ABC, abstractmethod
import time
import numpy as np
import torch

from .model import ParaphraseIdentifier_Model
from .utils import decode_label


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


class ParaphraseIdentifier_BaseEngine(ABC):
    def __init__(self, paraphrase_model: ParaphraseIdentifier_Model):
        if not isinstance(paraphrase_model, ParaphraseIdentifier_Model):
            raise TypeError
        self._model: ParaphraseIdentifier_Model = paraphrase_model

    @abstractmethod
    def data_encoding(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass


class ParaphraseIdentifier_Engine(ParaphraseIdentifier_BaseEngine):
    @timeit
    def data_encoding(self, tokenizer, question1, question2):
        encoded_dict = tokenizer.encode_plus(text=question1,
                                             text_pair=question2,
                                             add_special_tokens=True,
                                             max_length=32,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt',
                                             truncation=True
                                             )
        input_id = encoded_dict['input_ids']
        token_type_id = encoded_dict['token_type_ids']
        attention_mask = encoded_dict['attention_mask']
        return input_id, token_type_id, attention_mask

    @timeit
    def predict(self, question1, question2):
        input_id, token_type_id, attention_mask = self.data_encoding(self._model.tokenizer, question1, question2)
        
        with torch.no_grad():
            logits = self._model.model(input_id, 
                                       token_type_ids=token_type_id, 
                                       attention_mask=attention_mask)

        logits = logits[0].detach().cpu().numpy()
        pred = np.argmax(logits, axis=1)
        pred_label = decode_label(pred)
        self.output = pred_label
