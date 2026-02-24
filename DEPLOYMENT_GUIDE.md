# Semantic Inference System - Deployment Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Backup and Recovery](#backup-and-recovery)

---

## 1. System Requirements

### 1.1 Hardware Requirements

**Minimum Configuration**:
- CPU: 2 cores @ 2.0 GHz
- RAM: 4 GB
- Storage: 10 GB available space
- Network: Internet connection for LLM API access

**Recommended Configuration**:
- CPU: 4+ cores @ 2.5+ GHz
- RAM: 8-16 GB
- Storage: 50+ GB SSD
- Network: High-speed internet (1+ Mbps)

**For Production**:
- CPU: 8+ cores
- RAM: 16-32 GB
- Storage: 100+ GB SSD with RAID
- Network: Dedicated connection with redundancy

### 1.2 Software Requirements

**Operating System**:
- Linux (Ubuntu 20.04+ recommended)
- macOS 11+
- Windows 10/11 with WSL2

**Required Software**:
- Docker 20.10+ and Docker Compose 2.0+
- OR Python 3.9+ and Node.js 14+ (for local development)
- Git (for repository cloning)

---

## 2. Installation Methods

### 2.1 Docker Deployment (Recommended)

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd rag-semantic-inference-deployment
```

**Step 2: Configure Environment**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
nano .env
```

Required environment variables:
```bash
NEO4J_PASSWORD=your_secure_password_here
GEMINI_API_KEY=your_google_gemini_api_key
```

**Step 3: Update Docker Compose**
```bash
# Edit docker-compose.yml to set passwords
nano docker-compose.yml

# Update these lines:
# - NEO4J_AUTH=neo4j/your_password_here
# - NEO4J_PASSWORD=your_password_here
```

**Step 4: Start Services**
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

**Step 5: Verify Installation**
```bash
# Check Neo4j
curl http://localhost:7474

# Check API
curl http://localhost:5001/health
```

### 2.2 Local Development Setup

**Backend Setup**:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export GEMINI_API_KEY="your_api_key"

# Start Neo4j (if not using Docker)
# Download and install from https://neo4j.com/download/

# Start Flask API
python semantic_search_api.py
```

**Chainlit Interface**:
```bash
# In the same directory
chainlit run app.py -w
```

**React Frontend**:
```bash
# Navigate to frontend directory
cd semantic-inference

# Install dependencies
npm install

# Start development server
npm start
```

Access at http://localhost:3000

---

## 3. Configuration

### 3.1 Neo4j Configuration

**Memory Settings** (docker-compose.yml):
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=512m
  - NEO4J_dbms_memory_heap_max__size=2G
  - NEO4J_dbms_memory_pagecache_size=1G
```

**For Large Datasets**:
```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=1G
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=2G
```

**APOC Plugin**:
Enabled by default in docker-compose.yml:
```yaml
- NEO4J_PLUGINS=["apoc"]
```

### 3.2 Flask API Configuration

**File**: `semantic_search_api.py`

**Key Configuration Points**:

```python
# Neo4j Connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Flask Server
app.run(host='127.0.0.1', port=5001, debug=False)

# CORS (for production, restrict origins)
CORS(app, origins=["http://localhost:3000"])

# Embedding Model
model_name = "all-MiniLM-L6-v2"

# Batch Sizes
BATCH_SIZE = 500  # Graph creation
EMBEDDING_BATCH_SIZE = 50  # Embedding generation
```

### 3.3 Chainlit Configuration

**File**: `chainlit.yaml`

```yaml
project:
  name: "RAG Knowledge Base System"

ui:
  name: "HumAIne Assistant"
  default_theme: light
  
settings:
  port: 8000
```

### 3.4 LLM Configuration

**Google Gemini API**:
```python
# In semantic_search_api.py
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model selection
model = genai.GenerativeModel('gemini-pro')
```

**Alternative LLM Support**:
To use OpenAI instead:
```python
# Uncomment OpenAI sections in semantic_search_api.py
# Set OPENAI_API_KEY environment variable
```

---

## 4. Deployment Scenarios

### 4.1 Development Deployment

**Purpose**: Local testing and development

**Configuration**:
- Single server
- Docker Compose with development settings
- Debug mode enabled
- CORS allowed from localhost

**Steps**:
```bash
docker-compose up
# Access at http://localhost:7474 (Neo4j), http://localhost:5001 (API)
```

### 4.2 Staging Deployment

**Purpose**: Pre-production testing

**Configuration**:
- Production-like environment
- Security enabled
- Performance monitoring
- Limited CORS

**Steps**:
```bash
# Update docker-compose.yml for staging
docker-compose -f docker-compose.staging.yml up -d

# Enable healthchecks
docker-compose ps  # Verify all services healthy
```

### 4.3 Production Deployment

**Prerequisites**:
- Domain name and SSL certificate
- Reverse proxy (Nginx/Traefik)
- Monitoring system (Prometheus/Grafana)
- Backup strategy

**Recommended Architecture**:
```
[Users] → [Load Balancer] → [Nginx Reverse Proxy] → [Flask API]
                                                    → [Neo4j]
```

**Production docker-compose.yml additions**:
```yaml
services:
  api:
    environment:
      - FLASK_ENV=production
    restart: always
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

**Nginx Configuration**:
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://api:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4.4 Cloud Deployment

**AWS Deployment**:

1. **EC2 Instance**:
   - Instance type: t3.large or larger
   - AMI: Ubuntu 22.04 LTS
   - Security groups: Allow 80, 443, 7687 (Neo4j Bolt)

2. **RDS Alternative** (if not using containerized Neo4j):
   - Not directly supported; use EC2 with persistent volumes

3. **S3 for Backups**:
```bash
# Install AWS CLI
pip install awscli

# Backup Neo4j data
aws s3 sync /var/lib/neo4j/data s3://your-bucket/neo4j-backup/
```

**Google Cloud Deployment**:

1. **Compute Engine**:
   - Machine type: n1-standard-2 or larger
   - OS: Ubuntu 22.04

2. **Container Registry**:
```bash
# Build and push image
docker build -t gcr.io/your-project/semantic-inference:latest .
docker push gcr.io/your-project/semantic-inference:latest
```

**Kubernetes Deployment**:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-inference-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-inference
  template:
    metadata:
      labels:
        app: semantic-inference
    spec:
      containers:
      - name: api
        image: semantic-inference:latest
        ports:
        - containerPort: 5001
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j:7687"
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-key
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-inference-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5001
  selector:
    app: semantic-inference
```

---

## 5. Monitoring and Maintenance

### 5.1 Health Checks

**API Health Endpoint**:
```bash
curl http://localhost:5001/health
# Expected: {"status": "healthy"}
```

**Neo4j Health**:
```bash
docker exec -it rag-neo4j cypher-shell -u neo4j -p password "CALL dbms.components()"
```

### 5.2 Logging

**Docker Logs**:
```bash
# View all logs
docker-compose logs

# Follow API logs
docker-compose logs -f api

# Neo4j logs
docker-compose logs -f neo4j
```

**Application Logs**:
```python
# Logs are written to console by default
# For file logging, update semantic_search_api.py:

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 5.3 Performance Monitoring

**Neo4j Metrics**:
```cypher
// Query performance
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Transactions")

// Memory usage
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Memory Pools")
```

**Python Profiling**:
```python
# Add to semantic_search_api.py for profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

### 5.4 Resource Usage

**Monitor Docker Resources**:
```bash
docker stats
```

**Set Resource Limits**:
```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
```

---

## 6. Backup and Recovery

### 6.1 Neo4j Backup

**Manual Backup**:
```bash
# Stop Neo4j
docker-compose stop neo4j

# Backup data directory
docker run --rm \
  -v rag-semantic-inference-deployment_neo4j_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/neo4j-backup-$(date +%Y%m%d).tar.gz /data

# Restart Neo4j
docker-compose start neo4j
```

**Automated Backup Script**:
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d-%H%M%S)

# Backup Neo4j data
docker-compose exec -T neo4j \
  neo4j-admin dump --to=/tmp/neo4j-$DATE.dump

docker cp rag-neo4j:/tmp/neo4j-$DATE.dump \
  $BACKUP_DIR/

# Backup embeddings and configuration
tar czf $BACKUP_DIR/config-$DATE.tar.gz \
  .env docker-compose.yml

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.dump" -mtime +7 -delete
```

**Scheduled Backups** (cron):
```bash
# Add to crontab
0 2 * * * /path/to/backup.sh
```

### 6.2 Restore from Backup

```bash
# Stop services
docker-compose down

# Restore Neo4j dump
docker run --rm \
  -v $(pwd)/backups:/backups \
  -v rag-semantic-inference-deployment_neo4j_data:/data \
  neo4j:5.15-community \
  neo4j-admin load --from=/backups/neo4j-20260216.dump --force

# Restore configuration
tar xzf backups/config-20260216.tar.gz

# Restart services
docker-compose up -d
```

### 6.3 Disaster Recovery Plan

**Preparation**:
1. Document all configuration settings
2. Store credentials securely (password manager)
3. Maintain offsite backup copies
4. Test restore procedures regularly

**Recovery Steps**:
1. Provision new infrastructure
2. Install Docker and dependencies
3. Restore code from Git repository
4. Restore Neo4j data from backup
5. Restore configuration files
6. Verify data integrity
7. Test search functionality
8. Switch DNS/traffic to new instance

**RTO/RPO**:
- Recovery Time Objective: < 4 hours
- Recovery Point Objective: < 24 hours (daily backups)

---

## 7. Upgrade Procedures

### 7.1 Application Upgrade

```bash
# Backup current state
./backup.sh

# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Stop old version
docker-compose down

# Start new version
docker-compose up -d

# Verify functionality
curl http://localhost:5001/health
```

### 7.2 Neo4j Upgrade

**Minor Version**:
```bash
# Update docker-compose.yml
# Change: neo4j:5.15-community → neo4j:5.16-community

docker-compose pull neo4j
docker-compose up -d neo4j
```

**Major Version**:
```bash
# Backup data
./backup.sh

# Read migration guide
# https://neo4j.com/docs/upgrade-migration-guide/

# Test in staging first
# Update docker-compose.yml
# Follow Neo4j migration procedures
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue**: Container won't start
```bash
# Check logs
docker-compose logs api

# Common causes:
# - Port already in use
# - Missing environment variables
# - Insufficient memory
```

**Issue**: Neo4j connection refused
```bash
# Wait for Neo4j to fully start (30-60 seconds)
docker-compose logs neo4j | grep "Started"

# Check network
docker network ls
docker network inspect rag-semantic-inference-deployment_rag-network
```

**Issue**: Out of memory
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Reduce Neo4j heap size
# Edit docker-compose.yml memory settings
```

### 8.2 Performance Issues

**Slow Searches**:
1. Check Neo4j memory configuration
2. Verify embeddings are generated
3. Reduce similarity threshold
4. Optimize Cypher queries

**High CPU Usage**:
1. Limit concurrent requests
2. Optimize batch sizes
3. Use caching for frequent queries

---

## 9. Security Hardening

### 9.1 Production Checklist

- [ ] Change default Neo4j password
- [ ] Restrict CORS to specific domains
- [ ] Enable HTTPS/TLS
- [ ] Implement rate limiting
- [ ] Add authentication middleware
- [ ] Restrict Neo4j network access
- [ ] Regular security updates
- [ ] Secure API key storage (secrets management)
- [ ] Enable audit logging
- [ ] Implement input validation

### 9.2 Network Security

```yaml
# docker-compose.yml - Production network settings
networks:
  rag-network:
    driver: bridge
    internal: true  # Isolate from external networks
    
  external:
    driver: bridge

services:
  api:
    networks:
      - rag-network
      - external  # Only API exposed
```

---

## 10. Support and Resources

### 10.1 Getting Help

- **Documentation**: Check TECHNICAL_DOCUMENTATION.md
- **Logs**: Review application and container logs
- **Neo4j Browser**: http://localhost:7474 for database inspection

### 10.2 Useful Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart service
docker-compose restart [service_name]

# Execute command in container
docker-compose exec api python script.py

# Shell access
docker-compose exec api bash

# Neo4j shell
docker-compose exec neo4j cypher-shell -u neo4j -p password

# Backup
docker-compose exec neo4j neo4j-admin dump --to=/tmp/backup.dump

# Check service health
docker-compose ps
docker-compose exec api curl localhost:5001/health
```

---

**Document Version**: 1.0  
**Last Updated**: February 16, 2026  
**For**: HumAIne Project - Semantic Inference System
