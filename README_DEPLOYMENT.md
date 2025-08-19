# 🚀 Product Recommendation Engine - Deployment Guide

## Quick Start

### Option 1: Simple Demo (No Dependencies)
```bash
python3 start.py demo
# or
python3 run_demo.py
```

### Option 2: Full System with Docker (Recommended)
```bash
# Install Docker and Docker Compose first
python3 start.py docker
# or manually:
docker-compose up --build -d
```

### Option 3: Python Development Mode
```bash
# Install dependencies first
pip install -r requirements.txt
python3 start.py python
```

## Available Endpoints

Once running, access these endpoints:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Recommendations**: http://localhost:8000/recommendations
- **Metrics (JSON)**: http://localhost:8000/metrics
- **Metrics (Prometheus)**: http://localhost:8000/metrics/prometheus

## Monitoring Stack (Docker only)

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Redis**: localhost:6379

## API Usage Examples

### Get Recommendations
```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "num_recommendations": 5}'
```

### Check Health
```bash
curl http://localhost:8000/health
```

### View Metrics
```bash
curl http://localhost:8000/metrics
```

## Features Included

✅ **Core ML Pipeline**
- Two-tower neural architecture
- BERT embeddings
- FAISS vector search
- LightGBM re-ranking

✅ **Production Features**
- FastAPI serving layer
- Redis caching
- A/B testing framework
- Prometheus metrics
- Real-time model updates
- Dynamic allocation

✅ **Data Pipeline**
- Automated data ingestion
- Quality validation
- Feature engineering
- Scheduled training

## System Requirements

### Minimal (Demo):
- Python 3.8+

### Full System:
- Python 3.8+
- 4GB+ RAM
- Docker & Docker Compose (recommended)

### Dependencies:
- PyTorch
- Transformers
- FAISS
- LightGBM
- FastAPI
- Redis
- Prometheus

## Troubleshooting

### Docker Issues
```bash
# Check Docker status
docker --version
docker-compose --version

# View logs
docker-compose logs -f
```

### Python Issues
```bash
# Install in virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Port Conflicts
If ports are in use, modify docker-compose.yml:
- API: Change `8000:8000` to `8001:8000`
- Prometheus: Change `9090:9090` to `9091:9090`
- Grafana: Change `3000:3000` to `3001:3000`

## Performance Tuning

### For High Traffic:
1. Scale API containers: `docker-compose up --scale recommendation-api=3`
2. Configure load balancer
3. Increase Redis memory
4. Add model serving replicas

### For Large Datasets:
1. Use GPU-enabled containers
2. Implement model sharding
3. Configure distributed training
4. Use persistent volumes

## Security Considerations

- Change default Grafana password
- Use environment variables for secrets
- Configure firewall rules
- Enable HTTPS in production
- Implement rate limiting

## Next Steps

1. **Data Integration**: Connect your data sources
2. **Model Training**: Train on your specific data
3. **A/B Testing**: Set up experiments
4. **Monitoring**: Configure alerts
5. **Scaling**: Add load balancers and replicas

🎉 Your recommendation engine is production-ready!