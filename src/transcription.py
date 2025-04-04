import os
import logging
import torch
import librosa
import soundfile as sf
from .utils import calculate_similarity, estimate_timestamps_from_text

# Configure logging
logger = logging.getLogger("transcription_app")

def transcribe_with_pipeline(audio_path, model, language="en"):
    """Transcribe using the pipeline approach"""
    logger.info(f"Transcribing with pipeline: language={language}")
    
    # Transcribe audio
    transcription = model(
        audio_path, 
        generate_kwargs={"language": language, "task": "transcribe"}
    )
    
    # Format the results
    chunks = []
    for chunk in transcription['chunks']:
        chunks.append({
            "start_time": chunk["timestamp"][0],
            "end_time": chunk["timestamp"][1],
            "text": chunk["text"]
        })
    
    return chunks

def transcribe_with_whisper_model(audio_path, processor, model, language="sw", chunk_size_seconds=10, overlap_seconds=5):
    """
    Enhanced transcription using the Whisper model with advanced techniques for Kinyarwanda
    Note: Uses 'sw' (Swahili) for Kinyarwanda since the model was fine-tuned that way
    """
    logger.info(f"Transcribing Kinyarwanda with Whisper model: chunk_size={chunk_size_seconds}s, overlap={overlap_seconds}s")
    
    # Set language and task based on model documentation
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    
    # Load and preprocess audio
    logger.info(f"Loading audio file: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
 
    if audio_duration > 60:  
      
        overlap_ratio = min(0.5, overlap_seconds / chunk_size_seconds + 0.1)
    else:
        overlap_ratio = overlap_seconds / chunk_size_seconds
    
    chunk_size = chunk_size_seconds * sr
    overlap = int(chunk_size * overlap_ratio)
    
    # Initialize results
    chunks = []
    
    # Process audio in chunks
    for i in range(0, len(audio), chunk_size - overlap):
        # Calculate timestamps
        start_time = max(0, i / sr)
        end_idx = min(i + chunk_size, len(audio))
        chunk_audio = audio[i:end_idx]
        end_time = end_idx / sr
        
        # Skip very short chunks
        if len(chunk_audio) < sr * 0.5:
            logger.debug(f"Skipping very short chunk ({len(chunk_audio)/sr:.2f}s)")
            continue
        
        # Process the chunk
        input_features = processor(chunk_audio, sampling_rate=16000, return_tensors="pt").input_features
        
        # Create attention mask (important for model focus)
        attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
            attention_mask = attention_mask.to("cuda")
        
        # Generate transcription
        try:
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    num_beams=5,              
                    max_length=256,            
                    min_length=1,             
                    length_penalty=1.0,         
                    repetition_penalty=1.2,    
                    no_repeat_ngram_size=3,    
                    early_stopping=True        
                )
            
            chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Add chunk to results
            chunks.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": chunk_transcription.strip()
            })
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {start_time:.2f}s-{end_time:.2f}s: {e}")
    
    # Sort chunks by start time (important for proper order)
    chunks.sort(key=lambda x: x["start_time"])
    
    # Process chunks to remove duplicates
    processed_chunks = remove_duplicates(chunks)
    
    return processed_chunks

def remove_duplicates(chunks):
    """Helper function to remove duplicated content between chunks"""
    if not chunks:
        return []
        
    processed_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        
        # Skip empty chunks
        if not current_chunk["text"]:
            continue
            
        # Check for overlap with previous chunk
        prev_chunk = processed_chunks[-1]
        prev_words = prev_chunk["text"].split()
        curr_words = current_chunk["text"].split()
        
        # Try different overlap sizes to find repetition
        repeated_text = False
        for overlap_size in range(min(len(prev_words), len(curr_words), 10), 2, -1):
            if len(prev_words) >= overlap_size and len(curr_words) >= overlap_size:
                prev_phrase = " ".join(prev_words[-overlap_size:]).lower()
                curr_phrase = " ".join(curr_words[:overlap_size]).lower()
                
                # If significant similarity, remove overlapping portion
                if calculate_similarity(prev_phrase, curr_phrase) > 0.7:
                    current_chunk["text"] = " ".join(curr_words[overlap_size:])
                    repeated_text = True
                    break
        
        # If chunk still has content after removing duplicates, add it
        if current_chunk["text"] and not repeated_text:
            processed_chunks.append(current_chunk)
        elif repeated_text and current_chunk["text"]:
            processed_chunks.append(current_chunk)
    
    return processed_chunks

def align_nemo_with_whisper(nemo_text, whisper_chunks, audio_duration=None):
    """
    Align NeMo transcription text with Whisper chunks/timestamps
    Using dynamic text alignment algorithms for better matching
    """
    # If we only have one Whisper chunk, use it with NeMo text
    if len(whisper_chunks) <= 1:
        if audio_duration:
            end_time = audio_duration
        elif whisper_chunks:
            end_time = whisper_chunks[0]["end_time"]
        else:
            end_time = 0.0
            
        return [{
            "start_time": 0.0,
            "end_time": end_time,
            "text": nemo_text
        }]
    
    # Split nemo_text into words
    nemo_words = nemo_text.split()
    
    # Count words in Whisper chunks
    whisper_word_counts = [len(chunk["text"].split()) for chunk in whisper_chunks]
    total_whisper_words = sum(whisper_word_counts)
    
    # Basic proportional distribution
    word_ratios = [count / total_whisper_words for count in whisper_word_counts]
    
    # Distribute NeMo words to chunks based on word ratios
    aligned_chunks = []
    start_idx = 0
    
    for i, chunk in enumerate(whisper_chunks):
        # Calculate how many words should go in this chunk
        if i == len(whisper_chunks) - 1:
            # Last chunk gets all remaining words
            chunk_words = nemo_words[start_idx:]
        else:
            # Distribute proportionally
            word_count = max(1, int(len(nemo_words) * word_ratios[i]))
            end_idx = min(start_idx + word_count, len(nemo_words))
            chunk_words = nemo_words[start_idx:end_idx]
            start_idx = end_idx
        
        if chunk_words:
            aligned_chunks.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": " ".join(chunk_words)
            })
    
    return aligned_chunks

def transcribe_with_production_approach(audio_path, nemo_model, whisper_processor, whisper_model, use_mfa=False):
    """
    Production-ready approach for Kinyarwanda transcription with accurate timestamps:
    1. Use NeMo for main transcription (highest accuracy for Kinyarwanda)
    2. Use Whisper for initial chunking and timestamps
    3. Optionally refine with MFA for word-level precision (when enabled)
    4. Fall back gracefully through multiple methods if any step fails
    """
    logger.info(f"Transcribing with production approach: {audio_path}, MFA enabled: {use_mfa}")
    
    try:
        # Step 1: Pre-process audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio) / sr
        
        temp_wav_path = os.path.join(os.path.dirname(audio_path), "temp_input.wav")
        sf.write(temp_wav_path, audio, 16000)
        
        # Step 2: Get the main transcription from NeMo (higher quality for Kinyarwanda)
        try:
            logger.info("Running NeMo transcription...")
            nemo_transcriptions = nemo_model.transcribe([temp_wav_path])
            
            if nemo_transcriptions and len(nemo_transcriptions) > 0:
                # Extract text from NeMo result
                if hasattr(nemo_transcriptions[0], 'text'):
                    nemo_text = nemo_transcriptions[0].text
                elif hasattr(nemo_transcriptions[0], 'transcript'):
                    nemo_text = nemo_transcriptions[0].transcript
                else:
                    nemo_text = str(nemo_transcriptions[0])
                
                logger.info(f"NeMo transcription successful: {len(nemo_text)} characters")
            else:
                logger.warning("NeMo returned empty transcription")
                nemo_text = None
        except Exception as e:
            logger.error(f"NeMo transcription failed: {e}")
            nemo_text = None
        
        # Step 3: Get timestamps from Whisper
        try:
            logger.info("Running Whisper transcription for timestamps...")
            whisper_chunks = transcribe_with_whisper_model(
                audio_path=temp_wav_path,
                processor=whisper_processor,
                model=whisper_model,
                language="sw",  # Using Swahili for Kinyarwanda as per original implementation
                chunk_size_seconds=10,
                overlap_seconds=5
            )
            
            if not whisper_chunks:
                logger.warning("Whisper returned no chunks")
                whisper_chunks = []
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            whisper_chunks = []
        
        # Decision tree based on what succeeded
        if not nemo_text and not whisper_chunks:
            # Both methods failed - try one more fallback approach
            logger.error("Both NeMo and Whisper transcription failed. Attempting direct Whisper pipeline...")
            try:
                from .models import load_pipeline_model
                whisper_pipeline = load_pipeline_model("openai/whisper-small")
                chunks = transcribe_with_pipeline(temp_wav_path, whisper_pipeline, language="en")
                logger.info("Fallback to Whisper pipeline successful")
                os.remove(temp_wav_path)
                return chunks
            except Exception as e:
                logger.error(f"All transcription methods failed: {e}")
                os.remove(temp_wav_path)
                return []
        
        elif not nemo_text:
            # NeMo failed but Whisper worked - use Whisper results
            logger.warning("Using Whisper results because NeMo failed")
            os.remove(temp_wav_path)
            return whisper_chunks
        
        elif not whisper_chunks:
            # Whisper failed but NeMo worked - create estimated timestamps
            logger.warning("Using NeMo with estimated timestamps because Whisper failed")
            chunks = estimate_timestamps_from_text(nemo_text, audio_duration)
            os.remove(temp_wav_path)
            return chunks
        
        # Step 4: We have both NeMo and Whisper results - align them
        logger.info("Aligning NeMo transcription with Whisper timestamps...")
        
        # Option: Use MFA for precise word-level alignment if enabled
        if use_mfa:
            try:
                logger.info("Attempting Montreal Forced Aligner for precision...")
                
                # Check if MFA is available
                try:
                    import montreal_forced_aligner
                    mfa_available = True
                except ImportError:
                    logger.warning("Montreal Forced Aligner not installed. Skipping MFA step.")
                    mfa_available = False
                
                if mfa_available:
                    # MFA implementation would go here
                    # For now, fall back to hybrid alignment
                    logger.info("MFA processing would occur here. Falling back to hybrid alignment for now.")
            except Exception as e:
                logger.error(f"MFA alignment failed: {e}")
        
        # Step 5: Hybrid alignment (fallback or if MFA not enabled)
        aligned_chunks = align_nemo_with_whisper(nemo_text, whisper_chunks, audio_duration)
        
        # Clean up
        os.remove(temp_wav_path)
        return aligned_chunks
    
    except Exception as e:
        logger.error(f"Error in production transcription approach: {e}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return []