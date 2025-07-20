import requests
from typing import Union, Optional, Dict

import torch
import numpy as np

from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline, chunk_iter
from transformers.utils import is_torchaudio_available
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from punctuators.models import PunctCapSegModelONNX
from diarizers import SegmentationModel


class Punctuator:

    ja_punctuations = ["!", "?", "、", "。"]

    def __init__(self, model: str = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"):
        self.punctuation_model = PunctCapSegModelONNX.from_pretrained(model)

    def punctuate(self, text: str) -> str:
        if any(p in text for p in self.ja_punctuations):
            return text
        punctuated = "".join(self.punctuation_model.infer([text])[0])
        if 'unk' in punctuated.lower():
            return text
        return punctuated


class SpeakerDiarization:

    def __init__(self,
                 device: torch.device,
                 model_id: str = "pyannote/speaker-diarization-3.1",
                 model_id_diarizers: Optional[str] = None):
        self.device = device
        self.pipeline = Pipeline.from_pretrained(model_id)
        self.pipeline = self.pipeline.to(self.device)
        if model_id_diarizers:
            self.pipeline._segmentation.model = SegmentationModel().from_pretrained(
                model_id_diarizers
            ).to_pyannote_model().to(self.device)

    def __call__(self,
                 audio: Union[torch.Tensor, np.ndarray],
                 sampling_rate: int,
                 num_speakers: Optional[int] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None) -> Annotation:
        if sampling_rate is None:
            raise ValueError("sampling_rate must be provided")
        if type(audio) is np.ndarray:
            audio = torch.as_tensor(audio)
        audio = torch.as_tensor(audio, dtype=torch.float32)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.shape) > 3:
            raise ValueError("audio shape must be (channel, time)")
        audio = {"waveform": audio.to(self.device), "sample_rate": sampling_rate}
        output = self.pipeline(audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        return output


class KotobaWhisperPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 model: "PreTrainedModel",
                 model_pyannote: str = "pyannote/speaker-diarization-3.1",
                 model_diarizers: Optional[str] = "diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn",
                 feature_extractor: Union["SequenceFeatureExtractor", str] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 device: Union[int, "torch.device"] = None,
                 device_pyannote: Union[int, "torch.device"] = None,
                 torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
                 **kwargs):
        self.type = "seq2seq_whisper"
        if device is None:
            device = "cpu"
        if device_pyannote is None:
            device_pyannote = device
        if type(device_pyannote) is str:
            device_pyannote = torch.device(device_pyannote)
        self.model_speaker_diarization = SpeakerDiarization(
            device=device_pyannote,
            model_id=model_pyannote,
            model_id_diarizers=model_diarizers
        )
        self.punctuator = None
        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs
        )

    def _sanitize_parameters(self,
                             chunk_length_s: Optional[int] = None,
                             stride_length_s: Optional[int] = None,
                             generate_kwargs: Optional[Dict] = None,
                             max_new_tokens: Optional[int] = None,
                             add_punctuation: bool = False,
                             return_unique_speaker: bool = True,
                             add_silence_end: Optional[float] = None,
                             add_silence_start: Optional[float] = None,
                             num_speakers: Optional[int] = None,
                             min_speakers: Optional[int] = None,
                             max_speakers: Optional[int] = None):
        preprocess_params = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "add_silence_end": add_silence_end,
            "add_silence_start": add_silence_start,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        postprocess_params = {"add_punctuation": add_punctuation, "return_timestamps": True, "return_language": False}
        forward_params = {} if generate_kwargs is None else generate_kwargs
        forward_params.update({"max_new_tokens": max_new_tokens, "return_timestamps": True, "language": "ja", "task": "transcribe"})
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self,
                   inputs,
                   chunk_length_s: Optional[int] = None,
                   stride_length_s: Optional[int] = None,
                   add_silence_end: Optional[float] = None,
                   add_silence_start: Optional[float] = None,
                   num_speakers: Optional[int] = None,
                   min_speakers: Optional[int] = None,
                   max_speakers: Optional[int] = None):

        def _pad_audio_array(_audio):
            if add_silence_start:
                _audio = np.concatenate([np.zeros(int(self.feature_extractor.sampling_rate * add_silence_start)), _audio])
            if add_silence_end:
                _audio = np.concatenate([_audio, np.zeros(int(self.feature_extractor.sampling_rate * add_silence_end))])
            return _audio

        # load file
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()
        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)
        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not ("sampling_rate" in inputs and "array" in inputs):
                raise ValueError(
                    "When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "
                    '"array" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = inputs.pop("array", None)
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AutomaticSpeechRecognitionPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )
                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()

        # validate audio array
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        # diarization
        sd = self.model_speaker_diarization(
            inputs,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            sampling_rate=self.feature_extractor.sampling_rate
        )

        # loop over audio chunks and speakers
        labels = list(sd.labels())
        for n, s in enumerate(labels):
            timelines = list(sd.label_timeline(s))
            for m, i in enumerate(timelines):
                start = int(i.start * self.feature_extractor.sampling_rate)
                end = int(i.end * self.feature_extractor.sampling_rate)
                audio_array = _pad_audio_array(inputs[start: end])

                if chunk_length_s is not None:
                    stride_length_s = chunk_length_s / 6 if stride_length_s is None else stride_length_s
                    stride_length_s = [stride_length_s, stride_length_s] if isinstance(stride_length_s, (int, float)) else stride_length_s
                    align_to = getattr(self.model.config, "inputs_to_logits_ratio", 1)
                    chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate / align_to) * align_to)
                    stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate / align_to) * align_to)
                    stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate / align_to) * align_to)
                    if chunk_len < stride_left + stride_right:
                        raise ValueError("Chunk length must be superior to stride length")
                    for item in chunk_iter(
                            audio_array, self.feature_extractor, chunk_len, stride_left, stride_right, self.torch_dtype
                    ):
                        item["speaker_id"] = s
                        item["speaker_span"] = [i.start, i.end]
                        item["is_last"] = m == len(timelines) - 1 and n == len(labels) - 1 and item["is_last"]
                        yield item
                else:
                    if audio_array.shape[0] > self.feature_extractor.n_samples:
                        processed = self.feature_extractor(
                            audio_array,
                            sampling_rate=self.feature_extractor.sampling_rate,
                            truncation=False,
                            padding="longest",
                            return_tensors="pt",
                        )
                    else:
                        processed = self.feature_extractor(
                            audio_array,
                            sampling_rate=self.feature_extractor.sampling_rate,
                            return_tensors="pt"
                        )
                    if self.torch_dtype is not None:
                        processed = processed.to(dtype=self.torch_dtype)
                    processed["speaker_id"] = s
                    processed["speaker_span"] = [i.start, i.end]
                    processed["is_last"] = m == len(timelines) - 1 and n == len(labels) - 1
                    yield processed

    def _forward(self, model_inputs, **generate_kwargs):
        generate_kwargs["attention_mask"] = model_inputs.pop("attention_mask", None)
        generate_kwargs["input_features"] = model_inputs.pop("input_features")
        tokens = self.model.generate(**generate_kwargs)
        return {"tokens": tokens, **model_inputs}

    def postprocess(self, model_outputs, **postprocess_parameters):
        if postprocess_parameters["add_punctuation"] and self.punctuator is None:
            self.punctuator = Punctuator()
        outputs = {"chunks": []}
        for o in model_outputs:
            text, chunks = self.tokenizer._decode_asr(
                [o],
                return_language=postprocess_parameters["return_language"],
                return_timestamps=postprocess_parameters["return_timestamps"],
                time_precision=self.feature_extractor.chunk_length / self.model.config.max_source_positions,
            )
            start, end = o["speaker_span"]
            new_chunk = []
            for c in chunks["chunks"]:
                c["timestamp"] = [round(c["timestamp"][0] + start, 2), round(c["timestamp"][0] + end, 2)]
                c["speaker_id"] = o["speaker_id"]
                new_chunk.append(c)
            outputs["chunks"] += new_chunk
        outputs["speaker_ids"] = sorted(set([o["speaker_id"] for o in outputs["chunks"]]))
        for s in outputs["speaker_ids"]:
            outputs[f"chunks/{s}"] = sorted([o for o in outputs["chunks"] if o["speaker_id"] == s], key=lambda x: x["timestamp"][0])
            outputs[f"text/{s}"] = "".join([i["text"] for i in outputs[f"chunks/{s}"]])
            if postprocess_parameters["add_punctuation"]:
                outputs[f"text/{s}"] = self.punctuator.punctuate(outputs[f"text/{s}"])
        return outputs
