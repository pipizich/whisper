import gradio as gr
import torch
import numpy as np
import json
import os
from typing import List, Tuple, Dict
from pathlib import Path
import tempfile
from datetime import timedelta
import re
import zipfile
import logging
import sys
from datetime import datetime

# Import required libraries
from transformers import pipeline
import requests
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline, chunk_iter
from transformers.utils import is_torchaudio_available
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from punctuators.models import PunctCapSegModelONNX
from diarizers import SegmentationModel

# Set HuggingFace token
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'

# Suppress unnecessary warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*non-writable tensors.*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for log storage
log_messages = []
MAX_LOG_MESSAGES = 100

def add_log(message: str, level: str = "INFO"):
    """Add a log message to the display buffer"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Handle custom levels
    if level == "SUCCESS":
        display_level = "SUCCESS"
        log_level = logging.INFO
    else:
        display_level = level
        log_level = getattr(logging, level, logging.INFO)
    
    log_entry = f"[{timestamp}] [{display_level}] {message}"
    log_messages.append(log_entry)
    if len(log_messages) > MAX_LOG_MESSAGES:
        log_messages.pop(0)
    
    logger.log(log_level, message)
    return "\n".join(log_messages)

# Get device information
def get_device_info():
    """Get current device information"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        return f"🎮 GPU: {device_name} ({device_memory})"
    else:
        import platform
        return f"💻 CPU: {platform.processor() or 'Unknown'}"

# Copy the classes from kotoba_whisper.py
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
                 model_id_diarizers: str = None):
        self.device = device
        self.pipeline = Pipeline.from_pretrained(model_id, use_auth_token=os.environ['HUGGING_FACE_HUB_TOKEN'])
        self.pipeline = self.pipeline.to(self.device)
        if model_id_diarizers:
            self.pipeline._segmentation.model = SegmentationModel().from_pretrained(
                model_id_diarizers
            ).to_pyannote_model().to(self.device)

    def __call__(self,
                 audio: torch.Tensor | np.ndarray,
                 sampling_rate: int,
                 num_speakers: int = None,
                 min_speakers: int = None,
                 max_speakers: int = None) -> Annotation:
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
                 model,
                 model_pyannote: str = "pyannote/speaker-diarization-3.1",
                 model_diarizers: str = "diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn",
                 feature_extractor=None,
                 tokenizer=None,
                 device=None,
                 device_pyannote=None,
                 torch_dtype=None,
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
                             chunk_length_s=None,
                             stride_length_s=None,
                             generate_kwargs=None,
                             max_new_tokens=None,
                             add_punctuation: bool = False,
                             return_unique_speaker: bool = True,
                             add_silence_end=None,
                             add_silence_start=None,
                             num_speakers=None,
                             min_speakers=None,
                             max_speakers=None,
                             **kwargs):  # Accept additional kwargs
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
                   chunk_length_s=None,
                   stride_length_s=None,
                   add_silence_end=None,
                   add_silence_start=None,
                   num_speakers=None,
                   min_speakers=None,
                   max_speakers=None):

        def _pad_audio_array(_audio):
            if add_silence_start:
                _audio = np.concatenate([np.zeros(int(self.feature_extractor.sampling_rate * add_silence_start)), _audio])
            if add_silence_end:
                _audio = np.concatenate([_audio, np.zeros(int(self.feature_extractor.sampling_rate * add_silence_end))])
            return _audio

        # load file
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()
        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)
        if isinstance(inputs, dict):
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


# Initialize global pipeline variable
pipeline_instance = None
processed_files = []  # Store processed files for download

def load_pipeline():
    """Load the Kotoba Whisper pipeline"""
    global pipeline_instance
    if pipeline_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        add_log(f"Loading Kotoba Whisper pipeline on {device}...")
        
        # For standard pipeline without diarization
        from transformers import pipeline as hf_pipeline
        pipeline_instance = hf_pipeline(
            "automatic-speech-recognition",
            model="kotoba-tech/kotoba-whisper-v2.2",
            device=device,
            torch_dtype=torch_dtype,
            token=os.environ['HUGGING_FACE_HUB_TOKEN']
        )
        
        add_log("Pipeline loaded successfully!")
    return pipeline_instance

def load_custom_pipeline():
    """Load the custom Kotoba Whisper pipeline with diarization support"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    add_log(f"Loading custom pipeline with diarization support on {device}...")
    
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "kotoba-tech/kotoba-whisper-v2.2",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        token=os.environ['HUGGING_FACE_HUB_TOKEN']
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "kotoba-tech/kotoba-whisper-v2.2",
        token=os.environ['HUGGING_FACE_HUB_TOKEN']
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "kotoba-tech/kotoba-whisper-v2.2",
        token=os.environ['HUGGING_FACE_HUB_TOKEN']
    )
    
    add_log("Custom pipeline loaded successfully!")
    
    return KotobaWhisperPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        torch_dtype=torch_dtype
    )

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_srt(chunks: List[Dict], speaker_wise: bool = False) -> str:
    """Generate SRT format from chunks"""
    srt_content = []
    index = 1
    
    for chunk in chunks:
        start_time = format_timestamp(chunk["timestamp"][0])
        end_time = format_timestamp(chunk["timestamp"][1])
        text = chunk["text"].strip()
        
        if speaker_wise and "speaker_id" in chunk:
            text = f"[{chunk['speaker_id']}] {text}"
        
        srt_content.append(f"{index}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # Empty line between subtitles
        index += 1
    
    return "\n".join(srt_content)

def generate_txt(result: Dict, diarization: bool = False) -> str:
    """Generate plain text format"""
    if diarization and "speaker_ids" in result:
        txt_content = []
        for speaker in result["speaker_ids"]:
            txt_content.append(f"[{speaker}]:")
            txt_content.append(result[f"text/{speaker}"])
            txt_content.append("")  # Empty line between speakers
        return "\n".join(txt_content)
    else:
        return result.get("text", "")

def process_audio_file(
    file_path: str,
    add_punctuation: bool,
    enable_diarization: bool,
    num_speakers: int
) -> Tuple[str, str, str]:
    """Process a single audio file and return SRT, TXT, and JSON content"""
    
    try:
        if enable_diarization:
            # Use custom pipeline with diarization
            pipe = load_custom_pipeline()
            
            # Set number of speakers
            kwargs = {
                "add_punctuation": add_punctuation,
                "chunk_length_s": 30,
                "ignore_warning": True  # Suppress experimental warning
            }
            
            if num_speakers > 0:
                kwargs["num_speakers"] = num_speakers
            
            add_log(f"Running diarization with {num_speakers if num_speakers > 0 else 'auto'} speakers...")
            result = pipe(file_path, **kwargs)
        else:
            # Use standard pipeline
            pipe = load_pipeline()
            result = pipe(
                file_path,
                chunk_length_s=30,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe", "language": "ja"},
                ignore_warning=True  # Suppress experimental warning
            )
            
            # Add punctuation if requested
            if add_punctuation:
                add_log("Adding punctuation...")
                punctuator = Punctuator()
                result["text"] = punctuator.punctuate(result["text"])
                for chunk in result.get("chunks", []):
                    chunk["text"] = punctuator.punctuate(chunk["text"])
        
        # Generate outputs
        if enable_diarization and "chunks" in result:
            srt_content = generate_srt(result["chunks"], speaker_wise=True)
        else:
            srt_content = generate_srt(result.get("chunks", []))
        
        txt_content = generate_txt(result, enable_diarization)
        json_content = json.dumps(result, ensure_ascii=False, indent=2)
        
        return srt_content, txt_content, json_content
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        add_log(error_msg, "ERROR")
        # Return empty but valid files instead of error messages
        error_json = {"error": str(e), "file": os.path.basename(file_path)}
        return "", f"Error: {str(e)}", json.dumps(error_json, ensure_ascii=False, indent=2)

def process_batch(
    files,
    add_punctuation: bool,
    enable_diarization: bool,
    num_speakers: int,
    progress=gr.Progress()
) -> Tuple[List[Tuple[str, str, str, str]], str, str]:
    """Process multiple audio files"""
    global processed_files
    processed_files = []  # Reset for new batch
    
    if not files:
        return [], "\n".join(log_messages), None
    
    results = []
    output_dir = tempfile.mkdtemp()
    
    for i, file in enumerate(files):
        progress((i + 1) / len(files), f"Processing {os.path.basename(file.name)}...")
        add_log(f"Processing file {i+1}/{len(files)}: {os.path.basename(file.name)}")
        
        # Get base filename without extension
        base_name = Path(file.name).stem
        
        # Process the file
        srt_content, txt_content, json_content = process_audio_file(
            file.name,
            add_punctuation,
            enable_diarization,
            num_speakers
        )
        
        # Create output files
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        
        # Write content to files
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_content)
        
        # Store paths for download
        processed_files.extend([srt_path, txt_path, json_path])
        
        results.append((
            os.path.basename(file.name),
            srt_path,
            txt_path,
            json_path
        ))
        
        add_log(f"✅ Completed: {os.path.basename(file.name)}")
    
    # Create zip file for all processed files
    zip_path = None
    if processed_files:
        zip_path = create_zip_archive(processed_files)
        add_log(f"Created zip archive with {len(processed_files)} files")
    
    add_log(f"Batch processing completed! Processed {len(files)} files.", "SUCCESS")
    
    return results, "\n".join(log_messages), zip_path

def create_zip_archive(file_paths: List[str]) -> str:
    """Create a zip archive of all processed files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(tempfile.gettempdir(), f"kotoba_whisper_output_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)
    
    return zip_path

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Kotoba Whisper Subtitle Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown(f"""
        # 🎯 Kotoba Whisper Subtitle Generator
        
        **Device:** {get_device_info()}
        
        Upload audio files to generate subtitles in SRT, TXT, and JSON formats using Kotoba Whisper v2.2.
        
        **Features:**
        - Batch processing support
        - Optional punctuation enhancement
        - Optional speaker diarization
        - Multiple output formats
        - Download all files as ZIP
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                audio_files = gr.File(
                    label="Upload Audio Files",
                    file_count="multiple",
                    file_types=["audio"]
                )
                
                add_punctuation = gr.Checkbox(
                    label="Add Punctuation",
                    value=False,
                    info="Uses xlm-roberta_punctuation_fullstop_truecase model"
                )
                
                enable_diarization = gr.Checkbox(
                    label="Enable Speaker Diarization",
                    value=False,
                    info="Uses pyannote models for speaker separation"
                )
                
                num_speakers = gr.Number(
                    label="Number of Speakers",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=10,
                    info="Set to 0 for automatic detection",
                    visible=False
                )
                
                process_btn = gr.Button("🚀 Process Audio Files", variant="primary")
                
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📄 Processing Results")
                output_files = gr.Dataframe(
                    headers=["Filename", "SRT", "TXT", "JSON"],
                    datatype=["str", "file", "file", "file"],
                    label="Generated Files",
                    interactive=False
                )
                
                download_all_btn = gr.File(
                    label="📦 Download All Files (ZIP)",
                    visible=True,
                    interactive=False
                )
        
        # Log display
        with gr.Row():
            with gr.Column():
                log_display = gr.Textbox(
                    label="📋 Processing Logs",
                    lines=10,
                    max_lines=20,
                    value="System ready...",
                    interactive=False
                )
            with gr.Column(scale=0.2):
                clear_log_btn = gr.Button("🗑️ Clear Logs", variant="secondary")
        
        # Clear log button
        def clear_logs():
            log_messages.clear()
            log_messages.append("Logs cleared.")
            return "\n".join(log_messages)
        
        clear_log_btn.click(
            fn=clear_logs,
            outputs=[log_display]
        )
        
        # Show/hide speaker number input based on diarization toggle
        enable_diarization.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_diarization],
            outputs=[num_speakers]
        )
        
        # Process button click
        process_btn.click(
            fn=process_batch,
            inputs=[
                audio_files,
                add_punctuation,
                enable_diarization,
                num_speakers
            ],
            outputs=[output_files, log_display, download_all_btn]
        )
        
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - Supported audio formats: MP3, WAV, M4A, FLAC, etc.
        - Processing time depends on audio length and selected features
        - Diarization requires additional processing time
        - All models are cached locally after first use
        - Logs show real-time processing status
        
        ### 💡 Tips:
        - For best results with Japanese audio, ensure clear audio quality
        - Speaker diarization works best with distinct voices
        - Punctuation model may need time to load on first use
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    # Pre-load the standard pipeline for faster first inference
    print("Initializing Kotoba Whisper Subtitle Generator...")
    print(f"Device: {get_device_info()}")
    
    try:
        load_pipeline()
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not pre-load pipeline: {e}")
        add_log(f"Pipeline pre-loading failed: {e}", "WARNING")
    
    # Create and launch the interface
    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )