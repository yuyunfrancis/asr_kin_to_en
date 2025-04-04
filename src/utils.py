import os
import logging
import re

# Configure logging
logger = logging.getLogger("transcription_app")

def calculate_similarity(str1, str2):
    """Calculate string similarity for overlap detection"""
    # Simple case: exact match
    if str1 == str2:
        return 1.0
        
    # Calculate word overlap
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union

def format_transcription(chunks):
    """Format transcription results with timestamps"""
    formatted_text = ""
    for chunk in chunks:
        start_time = chunk["start_time"]
        end_time = chunk["end_time"]
        text = chunk.get("text", "") 
        formatted_text += f"[{start_time:.2f}:{end_time:.2f}] {text}\n"
    return formatted_text.strip()

def save_upload_temporarily(uploaded_file):
    """Save uploaded file to temporary directory"""
    temp_dir = "temp_dir"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_file_path

def clean_temp_file(file_path):
    """Remove temporary file"""
    if os.path.exists(file_path):
        os.remove(file_path)

def save_caption_file(content, file_name, format_type="srt"):
    """Save caption content to a file"""
    temp_dir = "temp_dir"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, f"{file_name}.{format_type}")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

def estimate_timestamps_from_text(text, audio_duration):
    """Estimate timestamps from text when Whisper fails"""
    # Split text into sentences
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
    
    if not sentences:
        sentences = [text]
    
    # Simple approach: distribute time evenly
    chunks = []
    chunk_duration = audio_duration / len(sentences)
    
    for i, sentence in enumerate(sentences):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        
        chunks.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": sentence
        })
    
    return chunks