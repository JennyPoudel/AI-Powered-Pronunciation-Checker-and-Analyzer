import torch
import torch.nn as nn
import pickle
import pickle
import sys
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def getASRModel(language: str) -> nn.Module:
     processor= AutoProcessor.from_pretrained("jenrish/whisper-small-en")
     model= AutoModelForSpeechSeq2Seq.from_pretrained("jenrish/whisper-small-en")
     decoder= model.model.decoder
     return model, decoder

    


def getTTSModel(language: str) -> nn.Module:
    speaker = 'lj_16khz'  # 16 kHz
    model = torch.hub.load(repo_or_dir='snakers4/silero-models',
                               model='silero_tts',
                               language=language,
                               speaker=speaker,trust_repo=True)
    return model
   
# Load model directly

    
# model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
#                                                model='silero_stt',
#                                                language='en',
#                                                device=torch.device('cpu'))

#     print(model.eval())
# return (model, decoder)
    
# def getTranslationModel(language: str) -> nn.Module:
#     from transformers import AutoTokenizer
#     from transformers import AutoModelForSeq2SeqLM
#     if language == 'de':
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             "Helsinki-NLP/opus-mt-de-en")
#         tokenizer = AutoTokenizer.from_pretrained(
#             "Helsinki-NLP/opus-mt-de-en")
#         # Cache models to avoid Hugging face processing
#         with open('translation_model_de.pickle', 'wb') as handle:
#             pickle.dump(model, handle)
#         with open('translation_tokenizer_de.pickle', 'wb') as handle:
#             pickle.dump(tokenizer, handle)
#     else:
#         raise ValueError('Language not implemented')

#     return model, tokenizer
# getASRModel('en')