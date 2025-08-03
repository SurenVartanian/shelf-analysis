# Configuration Guide

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Required API Keys (at least one for vision analysis)

```bash
# OpenAI API Key (for GPT-4 models)
OPENAI_API_KEY=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

# Google Cloud Configuration (for Vertex AI models)
# Uses gcloud authentication - make sure you're logged in with: gcloud auth application-default login
VERTEX_PROJECT=392356656271
VERTEX_LOCATION=us-central1
```

### API Configuration

```bash
API_HOST=0.0.0.0
API_PORT=8000
```

### YOLO Configuration

```bash
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
```

### Optional: Google Cloud

```bash
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
```

## Available Vision Models

The system supports multiple vision models through LiteLLM:

### Google Cloud Vertex AI Models (requires gcloud auth)
- **gemini-flash** - Gemini 2.5 Flash (fastest, most efficient)
- **gemini-pro** - Gemini 2.5 Pro (more capable, slower)
- **gemini-flash-lite** - Gemini 2.5 Flash Lite (fastest for high volume)

### OpenAI Models (requires OPENAI_API_KEY)
- **gpt-4o** - GPT-4 Omni (most capable)
- **gpt-4o-mini** - GPT-4 Omni Mini (faster, cheaper)
- **gpt-4-vision-preview** - GPT-4 Vision Preview (specialized for images)

## Getting API Keys

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key

### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key

### Google Cloud (Vertex AI)
1. Install Google Cloud CLI: `brew install google-cloud-sdk`
2. Authenticate: `gcloud auth login`
3. Set up application default credentials: `gcloud auth application-default login`
4. Set your project: `gcloud config set project 392356656271`

## Testing the Setup

After setting up your API keys, you can test the vision service:

```bash
# Run the tests
pytest tests/test_litellm_vision.py -v

# Start the server
python -m src.shelf_analyzer.main

# Test via web interface
# Visit http://localhost:8000
```

## Troubleshooting

### "Service not initialized" error
- Check that at least one API key is set
- Verify the API key is valid
- Check the logs for initialization errors

### Model not available
- Ensure the corresponding API key is set
- Check that the model name is correct
- Verify the model is supported by your API plan

### Rate limiting
- LiteLLM automatically handles retries
- Consider using different models to distribute load
- Check your API provider's rate limits 