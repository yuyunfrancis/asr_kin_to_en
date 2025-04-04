import logging
from datetime import timedelta

# Configure logging
logger = logging.getLogger("transcription_app")

def format_timestamp(seconds, format_type="srt"):
    """Convert seconds to formatted timestamp"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    elif format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_srt_captions(translated_chunks):
    """Generate SRT format captions"""
    srt_content = []
    
    for i, chunk in enumerate(translated_chunks, 1):
        # Skip invalid chunks
        if not chunk.get("original_text") or not chunk.get("translated_text"):
            continue
            
        start_time = format_timestamp(chunk["start_time"], "srt")
        end_time = format_timestamp(chunk["end_time"], "srt")
        
        # Add both original and translated text
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(f"{chunk['original_text']}")
        srt_content.append(f"{chunk['translated_text']}")
        srt_content.append("")  # Empty line between entries
    
    return "\n".join(srt_content)

def generate_vtt_captions(translated_chunks):
    """Generate WebVTT format captions"""
    # Start with VTT header
    vtt_content = ["WEBVTT", ""]
    
    for i, chunk in enumerate(translated_chunks):
        if not chunk.get("original_text") or not chunk.get("translated_text"):
            continue
            
        start_time = format_timestamp(chunk["start_time"], "vtt")
        end_time = format_timestamp(chunk["end_time"], "vtt")
        
        vtt_content.append(f"{start_time} --> {end_time}")
        vtt_content.append(f"{chunk['original_text']}")
        vtt_content.append(f"{chunk['translated_text']}")
        vtt_content.append("")
    
    return "\n".join(vtt_content)