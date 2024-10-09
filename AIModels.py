import ModelInterfaces
import torch
import numpy as np
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch.nn.functional as F
from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline

class NeuralASR(ModelInterfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder
        
  # Decoder from CTC-outputs to transcripts

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert(self.audio_transcript != None,
               'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""
        assert(self.word_locations_in_samples != None,
               'Can get word locations without having processed the audio')

        return self.word_locations_in_samples

        
    
# Example usage
# Assuming `audio_tensor` is your input tensor
# process_audio(audio_tensor)

    
  

# Example function to process and transcribe audio
    # def processAudio(self, audio: torch.Tensor):
    #     # Convert tensor to NumPy array if needed
    #     audio = audio.numpy() 
    #     # Convert to mono
    #     if audio.ndim > 1:
    #         # Average the channels to create mono audio
    #         audio = np.mean(audio, axis=1)

        
    #     # Load the ASR pipeline
    #     pipe = pipeline("automatic-speech-recognition", model="jenrish/whisper-small-en", return_timestamps="word")

    #     # Transcribe audio
    #     result = pipe("xyz.wav")
    #     #random_file_name_wav
    #     print(result)

    #     return (result['text'],result['chunks'])
    
    def processAudio(self, audio: torch.Tensor):
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperTokenizerFast, WhisperFeatureExtractor, pipeline
        # model_name = '/content/onnx_model' # folder name
        # model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=False)
        # tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
        # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        # from transformers import pipeline
        # pipe = pipeline('automatic-speech-recognition', 
        #                 model=model, 
        #                 tokenizer=tokenizer, 
        #                 feature_extractor=feature_extractor)
                # Convert tensor to NumPy array if needed
        audio = audio.numpy() 
        # Convert to mono
        if audio.ndim > 1:
            # Average the channels to create mono audio
            audio = np.mean(audio, axis=1)

        # Load the ASR pipeline
        # model_name = 'quantized_model-20240915T113703Z-001' # folder name
        # model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=False,library="transformers")
        # tokenizer = WhisperTokenizerFast.from_pretrained(model_name)
        # feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name,library="transformers")

        # from transformers import pipeline
        # pipe = pipeline('automatic-speech-recognition', 
        #                 model=model, 
        #                 tokenizer=tokenizer, 
        #                 feature_extractor=feature_extractor)
        pipe = pipeline("automatic-speech-recognition", model="jenrish/whisper-small-en", return_timestamps="word")

        # Transcribe audio
        result = pipe("xyz.wav")
        
        # Extract the text and word chunks
        self.audio_transcript = result['text']
        word_chunks = result['chunks']

        # Convert to desired format
        self.word_locations_in_samples = [
            {
                'word': chunk['text'],
                'start_ts': chunk['timestamp'][0],
                'end_ts': chunk['timestamp'][1]
            }
            for chunk in word_chunks
        ]

        # Print or return the formatted result
        print(self.audio_transcript, self.word_locations_in_samples)
        return self.audio_transcript, self.word_locations_in_samples








class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate

    def getAudioFromSentence(self, sentence: str) -> np.array:
        with torch.inference_mode():
            audio_transcript = self.model.apply_tts(texts=[sentence],
                                                    sample_rate=self.sampling_rate)[0]

        return audio_transcript


class NeuralTranslator(ModelInterfaces.ITranslationModel):
    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def translateSentence(self, sentence: str) -> str:
        """Get the transcripts of the process audio"""
        tokenized_text = self.tokenizer(sentence, return_tensors='pt')
        translation = self.model.generate(**tokenized_text)
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens=True)[0]

        return translated_text


