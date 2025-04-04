# Kinyarwanda-English Transcription & Translation App

A full-featured application for transcribing audio in Kinyarwanda and English with bidirectional translation capabilities and subtitle generation.

![Screenshot of the application](https://example.com/app_screenshot.png)

## Features

- **Advanced Audio Transcription**:

  - Transcribe Kinyarwanda audio using state-of-the-art models
  - Transcribe English audio with high accuracy
  - Supports both NeMo and Whisper models for Kinyarwanda

- **Bidirectional Translation**:

  - Translate Kinyarwanda → English
  - Translate English → Kinyarwanda
  - Uses specialized models for each direction

- **Caption Generation**:

  - Create SRT captions for video subtitling
  - Create WebVTT captions for web videos
  - Includes both original and translated text

- **Robust Processing**:
  - Smart chunking for better timestamps
  - Duplicate content removal
  - Multiple fallback mechanisms

## System Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.9 or higher (Python 3.11 recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB of free space for models
- **GPU**: Optional but recommended for faster processing (NVIDIA with CUDA support)
- **Internet Connection**: Required for initial model downloads

## Installation Methods

### Method 1: Standard Installation

1. Clone the repository:

```bash
git clone https://github.com/yuyunfrancis/asr_kin_to_en.git
cd asr_kin_to_en
```

2. Create and activate a virtual environment:

```bash
# For Linux/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Method 2: Using Conda Environment

1. Clone the repository:

```bash
git clone https://github.com/yuyunfrancis/asr_kin_to_en.git
cd asr_kin_to_en
```

2. Create and activate a conda environment:

```bash
conda create -n transcription python=3.11
conda activate transcription
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

### Method 3: Manual Installation with Custom NeMo Setup

1. Clone the repository:

```bash
git clone https://github.com/yuyunfrancis/asr_kin_to_en.git
cd asr_kin_to_en
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch (with CUDA if available):

```bash
# For CUDA support (check compatibility with your GPU)
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchaudio
```

4. Install NeMo Toolkit manually:

```bash
# Install NeMo with all dependencies
pip install "nemo_toolkit[all]"

# If the above fails, try the following alternative
pip install Cython
pip install nemo_toolkit
```

5. Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with your Hugging Face token:

```
HUGGING_FACE_TOKEN=your_token_here
```

2. (Optional) Configure GPU usage by setting the following environment variables:

```bash
# For Linux/macOS
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management

# For Windows
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Running the Application

### Starting the App

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501 by default.

### Alternative Startup Options

```bash
# Run on a specific port
streamlit run app.py --server.port 8080

# Enable server headless mode
streamlit run app.py --server.headless true

# Specify a different browser
streamlit run app.py --browser.serverAddress="0.0.0.0"

# Debug mode
streamlit run app.py --logger.level=debug
```

## Usage Guide

1. **Select Language and Model**:

   - For Kinyarwanda audio: Choose either NeMo or Whisper Kinyarwanda
   - For English audio: Use Whisper Small

2. **Configure Advanced Settings**:

   - Adjust chunk size and overlap for Whisper models
   - Enable/disable Whisper fallback for NeMo
   - Enable/disable Montreal Forced Aligner for precision timestamps
   - Configure subtitle generation

3. **Upload Audio**:

   - Upload an audio file (MP3, WAV, or OGG format)
   - Maximum file size is determined by your Streamlit configuration

4. **Start Processing**:

   - Click "Start Transcription"
   - Monitor progress through the status indicators

5. **View and Download Results**:
   - Original transcription text
   - Translation (if enabled)
   - Time-stamped text for both original and translation
   - Caption files (SRT/VTT if enabled)

## Troubleshooting

### Common Issues and Solutions

#### NeMo Installation Problems

If you encounter issues with NeMo installation:

```bash
# Try installing specific versions
pip install "nemo_toolkit[all]==1.17.0"

# Or install core components separately
pip install nemo_toolkit
pip install sentencepiece
pip install matplotlib
```

#### GPU Memory Errors

```bash
# Reduce batch size to save memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Or force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

#### Model Download Failures

```bash
# Retry with specific cache directory
export HF_HOME=/path/to/huggingface/cache
pip install -U huggingface_hub
huggingface-cli login --token your_token_here
```

#### Streamlit Connection Issues

If the app starts but the browser doesn't open:

```bash
# Try accessing manually at
http://localhost:8501

# Or specify address explicitly
streamlit run app.py --server.address 127.0.0.1
```

### Logs and Debugging

Log files are created in the project directory:

- `app.log`: General application logs
- `models.log`: Model loading and inference logs
- `errors.log`: Error messages and stack traces

Enable verbose logging by creating a `logging.conf` file or setting:

```bash
export LOGLEVEL=DEBUG
```

## Project Structure

```
project_root/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (for tokens)
├── temp_dir/               # Temporary directory for uploads
└── src/
    ├── __init__.py         # Makes src a proper package
    ├── models.py           # Model loading and initialization
    ├── transcription.py    # Transcription functions
    ├── translation.py      # Translation functions
    ├── utils.py            # Utility functions
    └── captions.py         # Caption generation functions
```

## Models Used

- **Kinyarwanda Transcription**:

  - NeMo Kinyarwanda STT Conformer Model: `mbazaNLP/Kinyarwanda_nemo_stt_conformer_model`
  - Whisper Small Kinyarwanda: `mbazaNLP/Whisper-Small-Kinyarwanda`

- **English Transcription**:

  - OpenAI Whisper Small: `openai/whisper-small`

- **Translation**:
  - Kinyarwanda → English: `RogerB/marian-finetuned-multidataset-kin-to-en`
  - English → Kinyarwanda: `mbazaNLP/Nllb_finetuned_education_en_kin`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MbazaNLP](https://huggingface.co/mbazaNLP) for the Kinyarwanda models
- [OpenAI](https://github.com/openai/whisper) for the Whisper model
- [NVIDIA](https://github.com/NVIDIA/NeMo) for the NeMo toolkit
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [Streamlit](https://streamlit.io/) for the user interface framework

## Contact

Project Link: [https://github.com/yuyunfrancis/asr_kin_to_en](https://github.com/yuyunfrancis/asr_kin_to_en)

## Citation

If you use this application in your research, please cite:

```
@software{asr_kin_to_en,
  author = {Francis Yuyun},
  title = {Kinyarwanda-English Transcription & Translation App},
  year = {2023},
  url = {https://github.com/yuyunfrancis/asr_kin_to_en}
}
```
