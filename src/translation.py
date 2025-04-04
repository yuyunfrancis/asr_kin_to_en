import logging
import re
import torch

# Configure logging
logger = logging.getLogger("transcription_app")

def translate_text(text, tokenizer, model, source_lang="kin", target_lang="eng_Latn", batch_size=512):
    """
    Translate text between Kinyarwanda and English
    
    Args:
        text: Text to translate
        tokenizer: The translation tokenizer
        model: The translation model
        source_lang: Source language code (for NLLB model)
        target_lang: Target language code (for NLLB model)
        batch_size: Maximum number of characters to process at once
        
    Returns:
        Translated text
    """
    logger.info(f"Translating text of length {len(text)} from {source_lang} to {target_lang}")
    
    # Check if using NLLB model which requires specific language codes
    is_nllb_model = "nllb" in model.config._name_or_path.lower()
    
    # If text is short enough, translate it all at once
    if len(text) < batch_size:
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return translation
    
    # For longer text, split into sentences and translate in batches
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into batches
    batches = []
    current_batch = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > batch_size and current_batch:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_length = len(sentence)
        else:
            current_batch.append(sentence)
            current_length += len(sentence)
    
    if current_batch:
        batches.append(" ".join(current_batch))
    
    # Translate each batch
    translations = []
    for batch in batches:
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(batch, return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        batch_translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        translations.append(batch_translation)
    
    # Join all translations
    complete_translation = " ".join(translations)
    
    return complete_translation

def translate_chunks(chunks, tokenizer, model, source_lang="kin", target_lang="eng_Latn"):
    """
    Translate each chunk separately while preserving timing information
    
    Args:
        chunks: List of dictionaries with start_time, end_time, and text
        tokenizer: The translation tokenizer
        model: The translation model
        source_lang: Source language code (for NLLB model)
        target_lang: Target language code (for NLLB model)
        
    Returns:
        List of dictionaries with start_time, end_time, original_text, and translated_text
    """
    logger.info(f"Translating {len(chunks)} chunks from {source_lang} to {target_lang}")
    
    # Check if using NLLB model which requires specific language codes
    is_nllb_model = "nllb" in model.config._name_or_path.lower()
    
    translated_chunks = []
    
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.get("text", "").strip():
            continue
            
        # Prepare inputs based on model type
        if is_nllb_model:
            # NLLB model uses forced_bos_token_id directly
            inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True)
            # Get the language token ID - NLLB uses different methods than other models
            if hasattr(tokenizer, 'lang_code_to_id'):
                # For older versions of the tokenizer
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            else:
                # For newer versions, use the convert_tokens_to_ids method
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation with forced BOS token
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id
                )
        else:
            # Regular models like MarianMT
            inputs = tokenizer(chunk["text"], return_tensors="pt", padding=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                output_ids = model.generate(**inputs)
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        translated_chunks.append({
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "original_text": chunk["text"],
            "translated_text": translation
        })
    
    return translated_chunks