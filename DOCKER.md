# 🐳 Docker Setup for Shelf Analysis

This guide explains how to run the Shelf Analysis system using Docker.

## 📋 Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)
- API keys for OpenAI or Google Cloud (optional but recommended)

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SurenVartanian/shelf-analysis.git
   cd shelf-analysis
   ```

2. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env and add your API keys
   nano .env
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t shelf-analysis .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e OPENAI_API_KEY=your_key_here \
     -e GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/logs:/app/logs \
     shelf-analysis
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | - |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google service account JSON | - |
| `API_HOST` | Host to bind the server to | `0.0.0.0` |
| `API_PORT` | Port to run the server on | `8000` |
| `DEBUG` | Enable debug mode | `True` |
| `YOLO_MODEL` | YOLO model to use | `yolov8n.pt` |
| `CONFIDENCE_THRESHOLD` | YOLO confidence threshold | `0.5` |

### Volumes

The following directories are mounted as volumes:

- `./data:/app/data` - Persistent data storage
- `./logs:/app/logs` - Application logs
- `./models:/app/models` - YOLO models (optional)

## 🏗️ Development

### Building for Development

```bash
# Build with development dependencies
docker build -t shelf-analysis:dev .

# Run with volume mounts for live code changes
docker run -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/static:/app/static \
  shelf-analysis:dev
```

### Using Docker Compose for Development

```bash
# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change the port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Permission denied:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER data/ logs/
   ```

3. **Out of memory:**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Or use a smaller YOLO model
   ```

4. **API key not working:**
   ```bash
   # Check environment variables
   docker-compose exec shelf-analysis env | grep API
   ```

### Health Check

The container includes a health check that verifies the API is responding:

```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:8000/health
```

## 📊 Monitoring

### View Logs

```bash
# All logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Specific service logs
docker-compose logs shelf-analysis
```

### Resource Usage

```bash
# Check container resource usage
docker stats

# Check disk usage
docker system df
```

## 🚀 Production Deployment

### Using Docker Compose

```bash
# Production build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With custom environment file
docker-compose --env-file .env.prod up -d
```

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml shelf-analysis
```

### Using Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=shelf-analysis
```

## 🔒 Security Considerations

1. **Never commit API keys** - Use environment variables
2. **Use secrets management** in production
3. **Regular security updates** - Keep base images updated
4. **Network security** - Use reverse proxy in production
5. **Resource limits** - Set memory and CPU limits

## 📝 Examples

### Custom Configuration

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  shelf-analysis:
    environment:
      - DEBUG=False
      - CONFIDENCE_THRESHOLD=0.7
    volumes:
      - ./custom-models:/app/models
```

### Multi-stage Build

```dockerfile
# Dockerfile.prod
FROM python:3.12-slim as builder
# ... build stage

FROM python:3.12-slim as runtime
# ... runtime stage
```

## 🤝 Contributing

When contributing to the Docker setup:

1. Test your changes locally
2. Update documentation
3. Ensure backward compatibility
4. Add appropriate labels and tags 