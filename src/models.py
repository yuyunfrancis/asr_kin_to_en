import os
import logging
import subprocess
import sys
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logger = logging.getLogger("transcription_app")

# Get Hugging Face token from environment
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

def load_pipeline_model(model_name="openai/whisper-small"):
    """Load model using the pipeline approach"""
    logger.info(f"Loading pipeline model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return pipeline(
        "automatic-speech-recognition", 
        model_name, 
        chunk_length_s=30, 
        stride_length_s=5, 
        return_timestamps=True, 
        device=device,
        framework="pt"
    )

def load_whisper_model(model_name="mbazaNLP/Whisper-Small-Kinyarwanda"):
    """Load Whisper model and processor"""
    logger.info(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name, token=hugging_face_token)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=hugging_face_token)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Using GPU for transcription")
    
    return processor, model

def load_nemo_model(model_name="mbazaNLP/Kinyarwanda_nemo_stt_conformer_model"):
    """Load NVIDIA NeMo ASR model"""
    logger.info(f"Loading NeMo model: {model_name}")
    
    try:
        # Check if nemo_toolkit is installed
        import importlib
        nemo_spec = importlib.util.find_spec("nemo_toolkit")
        nemo_asr_spec = importlib.util.find_spec("nemo.collections.asr")
        
        if nemo_spec is None or nemo_asr_spec is None:
            logger.warning("NeMo toolkit not found. Attempting to install...")
            # Try to install NeMo toolkit
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[all]"])
            
        # Import NeMo after ensuring it's installed
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Using GPU for NeMo model")
        
        return model, True
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "access" in error_msg.lower() or "restricted" in error_msg.lower():
            logger.error(f"Authentication error: No access to model {model_name}. Please check your Hugging Face token.")
            error_message = f"Authentication error: You don't have access to the NeMo model. Please check your Hugging Face token or request access to the model."
            return None, False, error_message
        else:
            logger.error(f"Failed to load NeMo model: {e}")
            return None, False, str(e)

def load_translation_model(direction="kin-to-en"):
    """
    Load the translation model and tokenizer
    
    Args:
        direction: Translation direction, either "kin-to-en" or "en-to-kin"
        
    Returns:
        Tokenizer and model for translation
    """
    if direction == "kin-to-en":
        model_name = "RogerB/marian-finetuned-multidataset-kin-to-en"
    else:  # en-to-kin
        model_name = "mbazaNLP/Nllb_finetuned_education_en_kin"
    
    logger.info(f"Loading translation model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Using GPU for translation")
        
    return tokenizer, model

def initialize_transcription_models(model_type, language_code):
    """Initialize the necessary models based on selection"""
    if "nemo" in model_type:
        # Load NeMo model
        nemo_model, nemo_loaded = load_nemo_model("mbazaNLP/Kinyarwanda_nemo_stt_conformer_model")
        
        # Always load Whisper as well for hybrid approach
        whisper_processor, whisper_model = load_whisper_model("mbazaNLP/Whisper-Small-Kinyarwanda")
        
        return {
            "nemo_model": nemo_model if nemo_loaded else None,
            "whisper_processor": whisper_processor,
            "whisper_model": whisper_model,
            "nemo_loaded": nemo_loaded
        }
    else:
        # Load Whisper only
        if "kinyarwanda" in model_type:
            processor, model = load_whisper_model("mbazaNLP/Whisper-Small-Kinyarwanda")
            return {
                "nemo_model": None,
                "whisper_processor": processor,
                "whisper_model": model,
                "nemo_loaded": False
            }
        else:
            model = load_pipeline_model("openai/whisper-small")
            return {
                "nemo_model": None,
                "whisper_model": model,
                "whisper_processor": None,
                "nemo_loaded": False
            }