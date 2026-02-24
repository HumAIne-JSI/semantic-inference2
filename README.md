# HumAIne Semantic Inference System

## Project Overview

**Project**: HumAIne - Human-Centered AI Network  
**Exploitable Result**: Semantic Inference for Industrial Knowledge Graphs  
**Version**: 1.0  
**Date**: February 2026

## Executive Summary

The Semantic Inference System is an AI-powered platform that transforms complex industrial data into queryable knowledge graphs, enabling natural language interaction with technical specifications, asset information, and operational data. By combining Neo4j graph database technology, semantic embeddings, and large language models, the system provides intuitive access to structured industrial data.

### Key Capabilities

- 🔍 **Natural Language Search**: Query technical data using everyday language
- 📊 **Multi-Format Support**: CSV, AAS (Asset Administration Shell), and RDF data ingestion
- 🧠 **AI-Enhanced Responses**: Context-aware answers using LLM technology
- 🚀 **Scalable Architecture**: Handles large datasets with 100,000+ entities
- 📈 **Comprehensive Evaluation**: Built-in quality metrics and benchmarking

---

## Repository Structure

```
rag-semantic-inference-deployment/
├── README.md                          # This file
├── TECHNICAL_DOCUMENTATION.md         # Comprehensive technical documentation
├── DEPLOYMENT_GUIDE.md                # Installation and deployment instructions
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Docker orchestration configuration
├── Dockerfile                         # Container build instructions
├── .env.example                       # Environment variables template
│
├── Core Backend Components
├── semantic_search_api.py             # Main Flask API (2778 lines)
├── create_graph.py                    # Neo4j graph creation from AAS
├── create_graph_rdf.py                # RDF data processing
├── gen_embeddings.py                  # Embedding generation for AAS
├── gen_embeddings_rdf.py              # Embedding generation for RDF
│
├── User Interfaces
├── app.py                             # Chainlit chat interface
├── chainlit.yaml                      # Chainlit configuration
├── semantic-inference/                # React web application
│   ├── src/
│   │   ├── App.js                     # Main React component
│   │   ├── SemanticSearch.js          # Search interface component
│   │   └── neo4j.js                   # Neo4j connection utilities
│   ├── public/                        # Static assets
│   ├── package.json                   # Node.js dependencies
│   └── README.md                      # React app documentation
│
├── Evaluation Framework
├── evaluation.py                      # Automated evaluation script
├── evaluation.json                    # Test queries and expected results
├── evaluation.ipynb                   # Evaluation notebook
├── eval_visualization.ipynb           # Results visualization
├── eval_cypher.py                     # Cypher query evaluation
├── eval_with_streaming.py             # Streaming evaluation tests
│
├── Data and Examples
├── dataset/
│   ├── products.csv                   # Sample CSV dataset (35MB)
│   └── products_cleaned.csv           # Cleaned product data
├── example_aas.json                   # Sample AAS specification (1.3MB)
│
├── Experimental Features
├── mcp_rag_server.py                  # MCP (Model Context Protocol) server
├── mcp_server.py                      # Alternative MCP implementation
├── main.ipynb                         # Development notebook
│
└── UI Assets
    └── public/                        # Branding and UI assets
        ├── avatars/
        ├── logo_dark.png
        ├── logo_light.png
        └── custom-background.png
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.9+ and Node.js 14+ (for local development)
- Google Gemini API key (for LLM integration)

### Installation (Docker - Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd rag-semantic-inference-deployment

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Update docker-compose.yml with secure passwords
nano docker-compose.yml
# Set NEO4J_AUTH and NEO4J_PASSWORD

# 4. Start the system
docker-compose up -d

# 5. Verify installation
curl http://localhost:7474  # Neo4j Browser
curl http://localhost:5001/health  # API health check
```

### Access Points

- **Neo4j Browser**: http://localhost:7474
- **Flask API**: http://localhost:5001
- **React Web App**: http://localhost:3000 (if started separately)
- **Chainlit Interface**: Run `chainlit run app.py`

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                       │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  React Web App   │      │  Chainlit Chat   │        │
│  │  - Visual UI     │      │  - Conversational│        │
│  │  - 3 Tab Layout  │      │  - Commands      │        │
│  └────────┬─────────┘      └────────┬─────────┘        │
└───────────┼──────────────────────────┼──────────────────┘
            │                          │
            └──────────┬───────────────┘
                       │ HTTP/REST
            ┌──────────▼──────────────┐
            │   Flask API Backend     │
            │  semantic_search_api.py │
            │                         │
            │  🔍 Semantic Search     │
            │  📊 Graph Management    │
            │  🔢 Embedding Gen       │
            │  🤖 LLM Integration     │
            └──────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
 ┌──────▼──────┐              ┌───────▼───────┐
 │   Neo4j     │              │ Google Gemini │
 │  Database   │              │   LLM API     │
 │             │              │               │
 │ - Knowledge │              │ - Response    │
 │   Graph     │              │   Generation  │
 │ - Vector    │              │ - Context     │
 │   Similarity│              │   Enhancement │
 └─────────────┘              └───────────────┘
```

### Technology Stack

**Backend**:
- Python 3.9+
- Flask 2.3.3 (Web framework)
- Neo4j 5.13.0 (Graph database)
- Sentence Transformers (Embeddings)
- Google Generative AI (LLM)

**Frontend**:
- React 18+
- Chainlit 1.0.0
- Axios (HTTP client)

**Infrastructure**:
- Docker & Docker Compose
- Neo4j Community Edition 5.15

---

## Core Features

### 1. Multi-Format Data Ingestion

**Supported Formats**:

- **CSV**: Standard tabular data with automatic column detection
- **AAS (Asset Administration Shell)**: Industry 4.0 standard for asset description
  - Supports assetAdministrationShells and submodels
  - Handles derivedFrom relationships (HAS_MODEL)
  - Extracts manufacturer, energy consumption, capabilities
- **RDF**: Resource Description Framework for semantic web data
  - Triple-based data representation
  - Ontology-aware processing

**Example AAS Processing**:
```json
{
  "id": "https://example.com/ids/aas/5018_7091_8002_8858",
  "idShort": "Drilling Machine #5",
  "description": [{"text": "High-precision drilling machine"}],
  "assetKind": "Instance",
  "submodels": [
    {"keys": [{"value": "https://example.com/ids/sm/Manufacturer"}]},
    {"keys": [{"value": "https://example.com/ids/sm/Drilling"}]}
  ]
}
```

### 2. Semantic Search Pipeline

**Process Flow**:

1. **Query Embedding**: User query → 384-dimensional vector (all-MiniLM-L6-v2)
2. **Vector Similarity Search**: Cosine similarity in Neo4j graph
3. **Result Retrieval**: Top-N similar entities (default: 15)
4. **Reranking**: Cross-encoder for improved relevance (optional)
5. **Context Gathering**: Retrieve connected nodes and relationships
6. **LLM Enhancement**: Generate natural language response using Gemini
7. **Streaming Response**: Real-time output via Server-Sent Events

**Search Query Example**:
```
User: "Which drilling machines have the highest energy consumption?"

System Process:
1. Embed query → [0.234, -0.891, 0.445, ...]
2. Find similar Asset nodes in Neo4j
3. Retrieve connected EnergyConsumption nodes
4. Rank by relevance
5. Generate response:

"Based on the knowledge graph, the top 3 drilling machines by energy 
consumption are:

1. Drilling Machine #12: 276W (230V, 1.2A)
2. Drilling Machine #8: 242W (220V, 1.1A)
3. Drilling Machine #15: 216W (240V, 0.9A)

These machines require higher power due to their larger drilling 
diameters (10-12mm) and depths (50-60mm)."
```

### 3. Knowledge Graph Schema

**AAS Format Schema**:
```cypher
(:Asset {
    id: string,
    idShort: string,
    description: string,
    assetKind: string,
    availability: string,
    embedding: float[]  // 384-dimensional vector
})
-[:MANUFACTURED_BY]->(:Manufacturer {name: string})
-[:HAS_ENERGY_CONSUMPTION]->(:EnergyConsumption {details: string})
-[:HAS_DRILLING]->(:Drilling {details: string})
-[:HAS_CIRCLE_CUTTING]->(:CircleCutting {details: string})
-[:HAS_SAWING]->(:Sawing {details: string})
-[:HAS_AVAILABILITY]->(:Availability {status: string})
-[:HAS_MODEL]->(:Asset)  // Template/instance relationship
```

### 4. Evaluation Framework

**Implemented Metrics**:

| Metric | Type | Purpose |
|--------|------|---------|
| BLEU | Lexical | N-gram precision |
| ROUGE-1/2/L | Lexical | Recall-oriented matching |
| Cosine Similarity | Lexical | TF-IDF vector comparison |
| BERT Score | Semantic | Contextualized embedding similarity |
| BLEURT | Semantic | Learned metric (BERT-based) |
| Sentence-BERT | Semantic | Sentence embedding comparison |
| Precision/Recall/F1 | Token-based | Exact token matching |

**Evaluation Process**:
```bash
python evaluation.py
# Outputs: query_results.csv with per-query metrics
# Logs: Average scores across all test queries
```

---

## Usage Guide

### Creating a Knowledge Graph

**Option 1: Via Chainlit Interface**
```
# Start Chainlit
chainlit run app.py

# In chat:
/upload-kg
[Upload your AAS JSON file]

# System processes file and creates graph
# Then generate embeddings
```

**Option 2: Via React Web App**
```
1. Navigate to http://localhost:3000
2. Click "Manage Knowledge Graph" tab
3. Select format (CSV/AAS/RDF)
4. Upload file or paste text
5. Click "Upload"
6. Monitor progress bar
7. Go to "Generate Embeddings" tab
8. Click "Start Generating Embeddings"
```

**Option 3: Via API**
```bash
# Upload file
curl -X POST http://localhost:5001/create-graph \
  -F "file=@example_aas.json" \
  -F "graphFormat=aas"

# Generate embeddings
curl -X POST http://localhost:5001/generate-embeddings \
  -H "Content-Type: application/json" \
  -d '{"graphFormat": "aas"}'
```

### Performing Searches

**Example Queries**:

```
1. "List all available drilling machines"
   → Returns assets with Availability status = True and Drilling capability

2. "What manufacturers produce machines with circle cutting?"
   → Joins Asset → Manufacturer via relationships

3. "Show me machines with energy consumption above 250W"
   → Filters based on EnergyConsumption node details

4. "Which machines are derived from template TMP-001?"
   → Follows HAS_MODEL relationships

5. "Find all sawing machines made by Siemens"
   → Multi-hop query: Asset → MANUFACTURED_BY → Manufacturer
```

### API Reference

**Semantic Search**:
```bash
POST /semantic-search
Content-Type: application/json

{
  "query": "your search query here",
  "n": 10  // number of results
}

# Returns: Server-Sent Events stream with response chunks
```

**Graph Management**:
```bash
# Create/Update Graph
POST /create-graph
Content-Type: multipart/form-data

file: <binary file data>
graphFormat: "csv" | "aas" | "rdf"

# Delete Graph
POST /delete-graph

# Set Graph Format
POST /set-graph-format
Content-Type: application/json

{"graphFormat": "aas"}
```

**Embeddings**:
```bash
# Generate Embeddings
POST /generate-embeddings
Content-Type: application/json

{"graphFormat": "aas"}

# Check Progress
GET /generate-embeddings-status

# Returns:
{
  "status": "In Progress" | "Completed",
  "progress": {
    "processed": 150,
    "total": 500
  }
}
```

---

## Configuration

### Environment Variables

Create `.env` file:
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# LLM Configuration
GEMINI_API_KEY=your_google_gemini_api_key

# Flask Configuration (optional)
FLASK_ENV=production
FLASK_DEBUG=False
```

### Performance Tuning

**Neo4j Memory** (docker-compose.yml):
```yaml
# For large graphs (100K+ nodes):
- NEO4J_dbms_memory_heap_max__size=4G
- NEO4J_dbms_memory_pagecache_size=2G

# For small graphs (<10K nodes):
- NEO4J_dbms_memory_heap_max__size=1G
- NEO4J_dbms_memory_pagecache_size=512m
```

**Batch Sizes** (semantic_search_api.py):
```python
# Graph creation batch size
BATCH_SIZE = 500  # Increase for faster import on powerful machines

# Embedding generation batch size
EMBEDDING_BATCH_SIZE = 50  # Increase if you have sufficient memory
```

**Search Parameters**:
```python
# Number of results to retrieve
n_results = 15  # Higher = more comprehensive, slower

# Similarity threshold
similarity_threshold = 0.5  # Lower = more results, less relevant
```

---

## Testing and Validation

### Unit Tests

```bash
# Run evaluation suite
python evaluation.py

# Expected output:
# - Processing 20 test queries
# - Comparing against ground truth
# - Generating metrics CSV
# - Logging average scores
```

### Performance Benchmarks

**System Capabilities** (tested configuration):
- Dataset size: 1000 assets with relationships
- Graph creation: ~10 seconds
- Embedding generation: ~20 seconds (all nodes)
- Search latency: 1-3 seconds per query
- Concurrent users: 10+ simultaneous searches

### Quality Metrics

**Evaluation Results** (example):
```
BLEU Score: 0.32
ROUGE-L: 0.57
BERT F1: 0.71
Sentence-BERT Similarity: 0.82
Overall Accuracy: 78%
```

---

## Troubleshooting

### Common Issues

**Issue 1: Neo4j Won't Start**
```bash
# Check if port 7687 is in use
lsof -i :7687

# View Neo4j logs
docker-compose logs neo4j

# Solution: Ensure no other Neo4j instance running
```

**Issue 2: API Connection Refused**
```bash
# Verify API is running
curl http://localhost:5001/health

# Check logs
docker-compose logs api

# Solution: Ensure Flask started successfully, check firewall
```

**Issue 3: Embeddings Generation Fails**
```bash
# Check memory availability
free -h

# Reduce batch size in gen_embeddings.py
EMBEDDING_BATCH_SIZE = 25  # Reduced from 50

# Restart embedding generation
```

**Issue 4: Slow Search Performance**
```bash
# Verify embeddings are generated
docker-compose exec neo4j cypher-shell -u neo4j -p password
> MATCH (n:Asset) WHERE n.embedding IS NOT NULL RETURN count(n);

# If 0, regenerate embeddings
# If >0, check Neo4j memory settings
```

---

## Development

### Running in Development Mode

**Backend**:
```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask in debug mode
export FLASK_ENV=development
python semantic_search_api.py
```

**Frontend**:
```bash
cd semantic-inference
npm start
# Hot reload enabled, changes reflect immediately
```

**Chainlit**:
```bash
chainlit run app.py -w
# -w flag enables watch mode for live reload
```

### Code Structure

**Main Files**:
- `semantic_search_api.py`: Core backend (2778 lines)
  - Lines 1-100: Imports and configuration
  - Lines 100-500: Cypher query generation
  - Lines 500-1500: Search logic and LLM integration
  - Lines 1500-2778: API endpoints and utilities

- `create_graph.py`: AAS graph creation (178 lines)
  - Neo4jBatchImporter class
  - Batch processing logic
  - AAS JSON parsing

- `gen_embeddings.py`: Embedding generation (210 lines)
  - EnhancedKGEmbeddings class
  - Node text preparation
  - Batch embedding creation

### Adding New Features

**Example: Adding a New Data Format**

1. Create parser: `create_graph_newformat.py`
2. Create embedding generator: `gen_embeddings_newformat.py`
3. Update API endpoint in `semantic_search_api.py`:
```python
@app.route('/create-graph', methods=['POST'])
def create_graph():
    graph_format = request.form.get('graphFormat', 'csv')
    if graph_format == 'newformat':
        process_newformat_file(file_path)
```
4. Update frontend format selector

---

## Contributing

### Guidelines

1. **Code Style**: Follow PEP 8 for Python, ESLint for JavaScript
2. **Documentation**: Update README and TECHNICAL_DOCUMENTATION.md
3. **Testing**: Add tests to evaluation framework
4. **Commit Messages**: Use conventional commits format

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "feat: add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Create Pull Request with description

---

## License

[Specify License - See LICENSE file]

---

## Support and Contact

**Documentation**:
- Technical Documentation: `TECHNICAL_DOCUMENTATION.md`
- Deployment Guide: `DEPLOYMENT_GUIDE.md`

**Resources**:
- Neo4j Documentation: https://neo4j.com/docs/
- Sentence Transformers: https://www.sbert.net/
- Chainlit: https://docs.chainlit.io/

**Project Team**: HumAIne Consortium

---

## Acknowledgments

This project is part of the HumAIne (Human-Centered AI Network) initiative, funded by [Funding Organization]. It builds upon research in:

- Retrieval-Augmented Generation (RAG)
- Knowledge Graphs for Industrial Applications
- Semantic Web Technologies
- Natural Language Processing

**Key Technologies**:
- Neo4j graph database platform
- Sentence-BERT embeddings
- Google Gemini large language model
- React and Chainlit user interfaces

---

## Changelog

### Version 1.0 (February 2026)
- Initial release
- Support for CSV, AAS, and RDF formats
- Semantic search with LLM enhancement
- React and Chainlit interfaces
- Comprehensive evaluation framework
- Docker deployment support

---

**Document Version**: 1.0  
**Last Updated**: February 16, 2026  
**Status**: Production Ready
