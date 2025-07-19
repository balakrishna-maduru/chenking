#!/bin/bash
set -e

echo "🐳 Setting up Chenking Local Embedding Service"
echo "=============================================="

# Create models directory
mkdir -p models
mkdir -p ollama_data

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Build custom embedding service
echo "🔨 Building custom embedding service..."
docker compose build custom-embedding-api

echo "🚀 Starting embedding services..."

# Choose which service to start
echo ""
echo "Choose embedding service to start:"
echo "1) Custom FastAPI service (recommended)"
echo "2) Hugging Face Text Embeddings Inference"
echo "3) Ollama with embeddings"
echo "4) All services"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Starting custom FastAPI embedding service..."
        docker compose --profile custom up -d
        ;;
    2)
        echo "Starting Hugging Face Text Embeddings Inference..."
        docker compose up -d embedding-service
        ;;
    3)
        echo "Starting Ollama embedding service..."
        docker compose --profile ollama up -d
        sleep 10
        echo "Installing embedding model in Ollama..."
        docker exec chenking-ollama ollama pull nomic-embed-text
        ;;
    4)
        echo "Starting all services..."
        docker compose --profile custom --profile ollama --profile cache up -d
        ;;
    *)
        echo "Invalid choice. Starting custom service..."
        docker compose --profile custom up -d
        ;;
esac

echo ""
echo "🎉 Embedding services are starting up!"
echo ""
echo "📋 Service Endpoints:"
echo "• Custom FastAPI:     http://localhost:8000"
echo "• HuggingFace TEI:    http://localhost:8080"
echo "• Ollama:            http://localhost:11434"
echo ""
echo "🔍 Health checks:"
echo "• curl http://localhost:8000/health"
echo "• curl http://localhost:8080/health"
echo ""
echo "⏱️  Services may take 1-2 minutes to fully start..."

# Wait for services and test
echo "Waiting for services to be ready..."
sleep 30

echo ""
echo "🧪 Testing embedding service..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Custom FastAPI service is ready!"
    echo ""
    echo "🔗 Update your Chenking configuration:"
    echo "processor = Processor('http://localhost:8000/chenking/embedding')"
else
    echo "⚠️  Service may still be starting. Check logs with:"
    echo "docker compose logs -f"
fi

echo ""
echo "🛠️  Useful commands:"
echo "• View logs:          docker compose logs -f"
echo "• Stop services:      docker compose down"
echo "• Restart services:   docker compose restart"
echo "• Check status:       docker compose ps"
