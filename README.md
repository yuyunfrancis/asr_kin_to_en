# Audio Transcription & Translation API

A robust API service for audio transcription, translation, and caption generation, specializing in English and Kinyarwanda languages.

## Features

- **Advanced Audio Transcription**:
  - Support for English and Kinyarwanda audio
  - Multiple transcription models (Whisper, NeMo, specialized Kinyarwanda models)
  - Accurate timestamps and chunking

- **Bidirectional Translation**:
  - Kinyarwanda → English
  - English → Kinyarwanda
  - Preservation of timestamps during translation

- **Caption Generation**:
  - SRT captions for video subtitling
  - WebVTT captions for web videos
  - Support for bilingual captions

- **Full Processing Pipeline**:
  - Combined transcription → translation → caption generation
  - Asynchronous job processing
  - Status tracking and result retrieval

## Deployment Guide

### Hardware Requirements

| Component | Recommended Specification |
|-----------|--------------------------|
| CPU | 8+ cores |
| RAM | 16GB+ |
| GPU | NVIDIA GPU with 8GB+ VRAM (T4 or better) |
| Storage | 100GB+ SSD |
| Network | 1Gbps+ connection |

### Software Requirements

- Ubuntu 20.04 LTS or later
- Docker and Docker Compose
- NVIDIA drivers and NVIDIA Docker
- Redis (for queue management)
- Nginx (for reverse proxy)

### Deployment Steps

1. **Prepare the server**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
sudo apt install docker.io docker-compose -y

# Install NVIDIA drivers and container toolkit
sudo apt install nvidia-driver-525 -y
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Set up the application**

```bash
# Clone repository
git clone https://github.com/your-repo/audio-transcription-api.git
cd audio-transcription-api

# Create required directories
mkdir -p models_cache temp_files

# Set up environment file
cp .env.example .env
# Edit .env with your Hugging Face token
```

3. **Configure Docker Compose**

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./models_cache:/app/models_cache
      - ./temp_files:/app/temp_files
    environment:
      - HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}
      - MODELS_CACHE_DIR=/app/models_cache
      - TEMP_FILE_DIR=/app/temp_files
      - USE_GPU=True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:6.2-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  nginx:
    image: nginx:1.21-alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - api

volumes:
  redis-data:
```

4. **Configure Nginx**

Create `nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 50M;
        proxy_read_timeout 300s;
    }
}
```

5. **Deploy and run the API**

```bash
# Start the services
docker-compose up -d

# Check logs
docker-compose logs -f
```

6. **Test the API**

```bash
# Test the API status
curl http://localhost/api/status

# The API documentation is available at:
# http://localhost/docs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process-audio` | POST | Complete pipeline: transcription → translation → captions |
| `/api/transcribe` | POST | Audio transcription only |
| `/api/translate` | POST | Text translation between languages |
| `/api/generate-captions` | POST | Generate captions from transcription results |
| `/api/jobs/{job_id}` | GET | Check the status of a job |

## Example API Usage

### Process Audio (Full Pipeline)

```javascript
// JavaScript example using fetch
async function processAudio(audioFile) {
  const formData = new FormData();
  formData.append('file', audioFile);
  formData.append('source_language', 'english');
  formData.append('transcription_model', 'whisper');
  formData.append('translation_options', JSON.stringify({
    enabled: true,
    target_language: 'kinyarwanda'
  }));
  formData.append('caption_options', JSON.stringify({
    enabled: true,
    formats: ['srt', 'vtt']
  }));

  const response = await fetch('http://your-api-domain/api/process-audio', {
    method: 'POST',
    body: formData
  });

  const job = await response.json();
  return job.job_id; // Use this ID to check status
}

// Check job status
async function checkJobStatus(jobId) {
  const response = await fetch(`http://your-api-domain/api/jobs/${jobId}`);
  return await response.json();
}
```

## Performance Optimization Tips

1. **Model Caching**: The API automatically caches models, but for faster startup after deployment, consider warming up the cache by making an initial request for each language you plan to support.

2. **Resource Allocation**: If processing many files concurrently, increase the number of worker processes and adjust the GPU memory allocation.

3. **File Management**: The API automatically cleans up temporary files, but monitor disk usage if processing large volumes of audio.

## Monitoring and Maintenance

- Monitor GPU usage: `nvidia-smi`
- Check system resource utilization: `htop`
- View API logs: `docker-compose logs -f api`
- Restart services: `docker-compose restart`
- Update deployment: `git pull && docker-compose down && docker-compose up -d --build`

