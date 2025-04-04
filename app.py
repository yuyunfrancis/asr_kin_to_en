import streamlit as st
import time
import os
import logging
import torch
from dotenv import load_dotenv

# Configure environment
load_dotenv()
os.environ["USE_TORCH"] = "1"  # Ensure PyTorch-only mode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription_app")

# Import modules from src
from src.models import (
    load_pipeline_model, 
    load_whisper_model, 
    load_nemo_model, 
    load_translation_model, 
    initialize_transcription_models
)
from src.transcription import (
    transcribe_with_pipeline,
    transcribe_with_whisper_model, 
    transcribe_with_production_approach
)
from src.translation import translate_text, translate_chunks
from src.captions import generate_srt_captions, generate_vtt_captions
from src.utils import (
    format_transcription, 
    save_upload_temporarily, 
    clean_temp_file, 
    save_caption_file
)

# Set the page config
st.set_page_config(
    page_title="Kinyarwanda-English Transcription & Translation", 
    layout="centered", 
    initial_sidebar_state="auto"
)

# =========== UI COMPONENTS =============

def render_header():
    """Render app header and description"""
    st.markdown("<h1 style='color: #00bfff;'>Kinyarwanda-English Transcription & Translation</h1>", unsafe_allow_html=True)
    st.markdown("<p>Generate accurate transcriptions from audio files in Kinyarwanda or English, with bidirectional translation support.</p>", unsafe_allow_html=True)

def render_language_and_model_selection():
    """Render combined language and model selection"""
    st.subheader("Language & Model Selection")
    
    # Create two columns for language and model
    col1, col2 = st.columns(2)
    
    with col1:
        language_options = ["English", "Kinyarwanda"]
        selected_language = st.radio("Select Audio Language", language_options)
    
    # Map selected language to language code and model options
    if selected_language == "English":
        language_code = "en"
        with col2:
            model_options = {
                "Whisper Small": {"name": "openai/whisper-small", "type": "general"}
            }
            selected_model = st.radio("Select Model", list(model_options.keys()))
        
        # Enable English to Kinyarwanda translation
        translate_to_kinyarwanda = True
        translate_to_english = False
    else:  # Kinyarwanda
        language_code = "sw"  # For Whisper (uses Swahili for Kinyarwanda)
        with col2:
            model_options = {
                "NeMo Kinyarwanda": {"name": "mbazaNLP/Kinyarwanda_nemo_stt_conformer_model", "type": "kinyarwanda-nemo"},
                "Whisper Kinyarwanda": {"name": "mbazaNLP/Whisper-Small-Kinyarwanda", "type": "kinyarwanda-whisper"}
            }
            selected_model = st.radio("Select Model", list(model_options.keys()))
        
        # Enable Kinyarwanda to English translation
        translate_to_english = True
        translate_to_kinyarwanda = False
    
    model_info = model_options[selected_model]
    
    return language_code, model_info["name"], model_info["type"], translate_to_english, translate_to_kinyarwanda

def render_advanced_settings(model_type, translate_to_english, translate_to_kinyarwanda):
    """Render advanced settings based on model type"""
    with st.expander("Advanced Settings"):
        if "nemo" in model_type:
            # NeMo specific settings
            col1, col2 = st.columns(2)
            
            with col1:
                use_mfa = st.checkbox(
                    "Use Montreal Forced Aligner", 
                    value=False,
                    help="Enable precise word-level alignment (requires additional dependencies)"
                )
            
            with col2:
                fallback_enabled = st.checkbox(
                    "Enable Whisper fallback", 
                    value=True,
                    help="Fall back to Whisper if NeMo fails to transcribe"
                )
            
            # Add translation options when applicable
            if translate_to_english or translate_to_kinyarwanda:
                st.markdown("---")
                st.markdown("**Translation Options**")
                generate_captions = st.checkbox(
                    "Generate subtitle files", 
                    value=True,
                    help="Generate SRT and VTT subtitle files with original and translated text"
                )
            else:
                generate_captions = False
                
            return None, None, fallback_enabled, use_mfa, generate_captions
        else:
            # Whisper model settings
            col1, col2 = st.columns(2)
            
            with col1:
                chunk_size = st.slider(
                    "Chunk Size (seconds)", 
                    min_value=5, 
                    max_value=30, 
                    value=10, 
                    help="Size of audio chunks in seconds"
                )
            
            with col2:
                overlap_size = st.slider(
                    "Overlap Size (seconds)", 
                    min_value=1, 
                    max_value=10, 
                    value=5,
                    help="Overlap between chunks in seconds"
                )
            
            # Add translation options when applicable
            if translate_to_english or translate_to_kinyarwanda:
                st.markdown("---")
                st.markdown("**Translation Options**")
                generate_captions = st.checkbox(
                    "Generate subtitle files", 
                    value=True,
                    help="Generate SRT and VTT subtitle files with original and translated text"
                )
            else:
                generate_captions = False
            
            return chunk_size, overlap_size, True, False, generate_captions  # Always enable fallback for consistency

# =========== MAIN APP =============

def main():
    # Render header
    render_header()
    
    # Language and model selection
    language_code, model_name, model_type, translate_to_english, translate_to_kinyarwanda = render_language_and_model_selection()
    
    # Advanced settings
    chunk_size, overlap_size, fallback_enabled, use_mfa, generate_captions = render_advanced_settings(
        model_type, translate_to_english, translate_to_kinyarwanda
    )
    
    # File upload
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Action button
        if st.button("Start Transcription", type="primary"):
            with st.spinner("Processing audio..."):
                start_time = time.time()
                
                # Save uploaded file temporarily
                temp_file_path = save_upload_temporarily(uploaded_file)
                
                try:
                    # Display progress indication
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process based on model selection
                    if "nemo" in model_type:
                        # Initialize all required models
                        status_text.text("Loading transcription models...")
                        progress_bar.progress(10)
                        
                        models = initialize_transcription_models(model_type, language_code)
                        
                        if models["nemo_loaded"]:
                            status_text.text("Transcribing with advanced hybrid approach...")
                            progress_bar.progress(30)
                            
                            # Use the production approach
                            chunks = transcribe_with_production_approach(
                                temp_file_path,
                                models["nemo_model"],
                                models["whisper_processor"],
                                models["whisper_model"],
                                use_mfa=use_mfa
                            )
                            
                            progress_bar.progress(80)
                            
                            # If transcription failed completely
                            if not chunks and fallback_enabled:
                                status_text.text("Hybrid approach failed. Falling back to Whisper...")
                                progress_bar.progress(50)
                                
                                # Use Whisper as fallback
                                processor, whisper_model = models["whisper_processor"], models["whisper_model"]
                                chunks = transcribe_with_whisper_model(
                                    audio_path=temp_file_path,
                                    processor=processor,
                                    model=whisper_model,
                                    language="sw",
                                    chunk_size_seconds=10,
                                    overlap_seconds=5
                                )
                                
                                if chunks:
                                    st.warning("NeMo processing failed. Transcription was done using Whisper model instead.")
                        else:
                            # NeMo failed to load, use Whisper fallback
                            status_text.text("Failed to load NeMo model. Using Whisper model instead...")
                            progress_bar.progress(20)
                            
                            # Load Kinyarwanda-specific Whisper model
                            processor, whisper_model = models["whisper_processor"], models["whisper_model"]
                            
                            # Transcribe with Whisper
                            chunks = transcribe_with_whisper_model(
                                audio_path=temp_file_path,
                                processor=processor,
                                model=whisper_model,
                                language="sw",
                                chunk_size_seconds=10,
                                overlap_seconds=5
                            )
                            
                            st.warning("NeMo model could not be loaded. Used Whisper model instead.")
                    
                    elif "kinyarwanda-whisper" in model_type:
                        # Use Whisper model fine-tuned for Kinyarwanda
                        status_text.text("Loading Kinyarwanda-specific Whisper model...")
                        progress_bar.progress(10)
                        
                        # Initialize models
                        models = initialize_transcription_models(model_type, language_code)
                        processor, model = models["whisper_processor"], models["whisper_model"]
                        
                        status_text.text("Transcribing Kinyarwanda audio...")
                        progress_bar.progress(30)
                        
                        # Transcribe with specialized model
                        chunks = transcribe_with_whisper_model(
                            audio_path=temp_file_path,
                            processor=processor,
                            model=model,
                            language="sw", 
                            chunk_size_seconds=chunk_size,
                            overlap_seconds=overlap_size
                        )
                    
                    else:
                        # Use general Whisper model for English
                        status_text.text("Loading Whisper model for English...")
                        progress_bar.progress(10)
                        
                        # Initialize models
                        models = initialize_transcription_models(model_type, language_code)
                        model = models["whisper_model"]
                        
                        status_text.text("Transcribing English audio...")
                        progress_bar.progress(30)
                        
                        # Transcribe with general model
                        chunks = transcribe_with_pipeline(
                            temp_file_path,
                            model,
                            language="en"
                        )
                    
                    # Post-processing
                    status_text.text("Post-processing transcription...")
                    progress_bar.progress(90)
                    
                    # Format the transcription
                    formatted_transcription = format_transcription(chunks)
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    # Display results
                    st.success("✅ Transcription completed!")
                    
                    # Get plain text version
                    plain_text = " ".join([chunk["text"] for chunk in chunks])
                    
                    # Handle translation if needed
                    if translate_to_english:
                        with st.spinner("Translating to English..."):
                            # Load Kinyarwanda-to-English translation model
                            translation_start_time = time.time()
                            translation_tokenizer, translation_model = load_translation_model(direction="kin-to-en")
                            
                            # Translate the full text (Kinyarwanda to English)
                            translated_text = translate_text(
                                plain_text, 
                                translation_tokenizer, 
                                translation_model
                            )
                            
                            # Translate each chunk
                            translated_chunks = translate_chunks(
                                chunks, 
                                translation_tokenizer, 
                                translation_model
                            )
                            
                            translation_end_time = time.time()
                            st.success(f"✅ Translation completed in {round(translation_end_time - translation_start_time, 2)} seconds")
                            
                            # Generate caption files if requested
                            if generate_captions:
                                # Create SRT captions
                                srt_content = generate_srt_captions(translated_chunks)
                                srt_file_path = save_caption_file(srt_content, "captions", "srt")
                                
                                # Create VTT captions
                                vtt_content = generate_vtt_captions(translated_chunks)
                                vtt_file_path = save_caption_file(vtt_content, "captions", "vtt")
                                
                                st.success("✅ Caption files generated")
                        
                        # Create tabs for different views (including translation)
                        tab1, tab2, tab3, tab4 = st.tabs(["Original Text", "English Translation", "With Timestamps", "Captions"])
                        
                        with tab1:
                            # Original text without timestamps
                            st.text_area("Transcribed Text (Kinyarwanda)", value=plain_text, height=300)
                            st.download_button(
                                "Download Original Text", 
                                plain_text, 
                                file_name="transcription_original.txt"
                            )
                        
                        with tab2:
                            # Translated text
                            st.text_area("Translated Text (English)", value=translated_text, height=300)
                            st.download_button(
                                "Download Translation", 
                                translated_text, 
                                file_name="transcription_english.txt"
                            )
                        
                        with tab3:
                            # Timestamped format
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.text_area("Original with Timestamps", value=formatted_transcription, height=300)
                                st.download_button(
                                    "Download Original with Timestamps", 
                                    formatted_transcription, 
                                    file_name="transcription_timestamps.txt"
                                )
                            
                            with col2:
                                # Create timestamped translation
                                translated_timestamp_text = ""
                                for chunk in translated_chunks:
                                    translated_timestamp_text += f"[{chunk['start_time']:.2f}:{chunk['end_time']:.2f}] {chunk['translated_text']}\n"
                                
                                st.text_area("Translation with Timestamps", value=translated_timestamp_text, height=300)
                                st.download_button(
                                    "Download Translation with Timestamps", 
                                    translated_timestamp_text, 
                                    file_name="translation_timestamps.txt"
                                )
                        
                        with tab4:
                            # Caption files
                            if generate_captions:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.text_area("SRT Caption File", value=srt_content, height=300)
                                    with open(srt_file_path, "rb") as f:
                                        st.download_button(
                                            "Download SRT Captions", 
                                            f, 
                                            file_name="captions.srt"
                                        )
                                
                                with col2:
                                    st.text_area("VTT Caption File", value=vtt_content, height=300)
                                    with open(vtt_file_path, "rb") as f:
                                        st.download_button(
                                            "Download VTT Captions", 
                                            f, 
                                            file_name="captions.vtt"
                                        )
                            else:
                                st.info("Caption generation was not enabled. Enable it in Advanced Settings to generate subtitle files.")
                    
                    elif translate_to_kinyarwanda:
                        with st.spinner("Translating to Kinyarwanda..."):
                            # Load English-to-Kinyarwanda translation model
                            translation_start_time = time.time()
                            translation_tokenizer, translation_model = load_translation_model(direction="en-to-kin")
                            
                            # Translate the full text (English to Kinyarwanda)
                            translated_text = translate_text(
                                plain_text, 
                                translation_tokenizer, 
                                translation_model,
                                source_lang="eng_Latn",  # NLLB model uses specific language codes
                                target_lang="kin_Latn"   # Kinyarwanda in Latin script
                            )
                            
                            # Translate each chunk
                            translated_chunks = translate_chunks(
                                chunks, 
                                translation_tokenizer, 
                                translation_model,
                                source_lang="eng_Latn",
                                target_lang="kin_Latn"
                            )
                            
                            translation_end_time = time.time()
                            st.success(f"✅ Translation completed in {round(translation_end_time - translation_start_time, 2)} seconds")
                            
                            # Generate caption files if requested
                            if generate_captions:
                                # Create SRT captions
                                srt_content = generate_srt_captions(translated_chunks)
                                srt_file_path = save_caption_file(srt_content, "captions", "srt")
                                
                                # Create VTT captions
                                vtt_content = generate_vtt_captions(translated_chunks)
                                vtt_file_path = save_caption_file(vtt_content, "captions", "vtt")
                                
                                st.success("✅ Caption files generated")
                        
                        # Create tabs for different views (including translation)
                        tab1, tab2, tab3, tab4 = st.tabs(["Original Text", "Kinyarwanda Translation", "With Timestamps", "Captions"])
                        
                        with tab1:
                            # Original text without timestamps
                            st.text_area("Transcribed Text (English)", value=plain_text, height=300)
                            st.download_button(
                                "Download Original Text", 
                                plain_text, 
                                file_name="transcription_original.txt"
                            )
                        
                        with tab2:
                            # Translated text
                            st.text_area("Translated Text (Kinyarwanda)", value=translated_text, height=300)
                            st.download_button(
                                "Download Translation", 
                                translated_text, 
                                file_name="transcription_kinyarwanda.txt"
                            )
                        
                        with tab3:
                            # Timestamped format
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.text_area("Original with Timestamps", value=formatted_transcription, height=300)
                                st.download_button(
                                    "Download Original with Timestamps", 
                                    formatted_transcription, 
                                    file_name="transcription_timestamps.txt"
                                )
                            
                            with col2:
                                # Create timestamped translation
                                translated_timestamp_text = ""
                                for chunk in translated_chunks:
                                    translated_timestamp_text += f"[{chunk['start_time']:.2f}:{chunk['end_time']:.2f}] {chunk['translated_text']}\n"
                                
                                st.text_area("Translation with Timestamps", value=translated_timestamp_text, height=300)
                                st.download_button(
                                    "Download Translation with Timestamps", 
                                    translated_timestamp_text, 
                                    file_name="translation_timestamps.txt"
                                )
                        
                        with tab4:
                            # Caption files
                            if generate_captions:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.text_area("SRT Caption File", value=srt_content, height=300)
                                    with open(srt_file_path, "rb") as f:
                                        st.download_button(
                                            "Download SRT Captions", 
                                            f, 
                                            file_name="captions.srt"
                                        )
                                
                                with col2:
                                    st.text_area("VTT Caption File", value=vtt_content, height=300)
                                    with open(vtt_file_path, "rb") as f:
                                        st.download_button(
                                            "Download VTT Captions", 
                                            f, 
                                            file_name="captions.vtt"
                                        )
                            else:
                                st.info("Caption generation was not enabled. Enable it in Advanced Settings to generate subtitle files.")
                    
                    else:
                        # Create regular tabs for English transcription without translation
                        tab1, tab2, tab3 = st.tabs(["Plain Text", "Paragraphs", "With Timestamps"])
                        
                        with tab1:
                            # Plain text without timestamps
                            st.text_area("Transcribed Text", value=plain_text, height=300)
                            st.download_button(
                                "Download Plain Text", 
                                plain_text, 
                                file_name="transcription_plain.txt"
                            )
                        
                        with tab2:
                            # Paragraphs format
                            paragraph_text = "\n\n".join([chunk["text"] for chunk in chunks])
                            st.text_area("Paragraphs", value=paragraph_text, height=300)
                            st.download_button(
                                "Download Paragraphs", 
                                paragraph_text, 
                                file_name="transcription_paragraphs.txt"
                            )
                        
                        with tab3:
                            # Timestamped format
                            st.text_area("With Timestamps", value=formatted_transcription, height=300)
                            st.download_button(
                                "Download with Timestamps", 
                                formatted_transcription, 
                                file_name="transcription_timestamps.txt"
                            )
                    
                    # Show processing time
                    end_time = time.time()
                    st.write(f"⏱️ Total processing time: {round(end_time - start_time, 2)} seconds")
                    
                finally:
                    # Clean up
                    clean_temp_file(temp_file_path)

if __name__ == "__main__":
    main()