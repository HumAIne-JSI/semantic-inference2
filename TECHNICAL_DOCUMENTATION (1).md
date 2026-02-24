# HumAIne Semantic Inference System - Technical Documentation

## Executive Summary

The **Semantic Inference System** is a Retrieval-Augmented Generation (RAG) platform that enables intelligent querying of structured data through natural language interfaces. The system integrates knowledge graph technology with large language models to provide context-aware, accurate responses based on industrial asset data and technical specifications.

**Version:** 1.0  
**Last Updated:** February 2026  
**ER Category:** AI-Powered Knowledge Management  

---

## 1. System Overview

### 1.1 Purpose and Objectives

The Semantic Inference System addresses the challenge of making complex industrial data accessible through natural language queries. It combines:

- **Knowledge Graph Construction**: Automated transformation of structured data (CSV, AAS, RDF) into graph representations
- **Semantic Search**: Vector similarity-based retrieval using sentence embeddings
- **AI-Powered Inference**: LLM-enhanced response generation with context enrichment
- **Multi-Format Support**: Flexible data ingestion from various industrial standards

### 1.2 Key Features

1. **Multiple Data Format Support**:
   - CSV (Comma-Separated Values) for tabular data
   - AAS (Asset Administration Shell) for industrial asset specifications
   - RDF (Resource Description Framework) for semantic web data

2. **Advanced Search Capabilities**:
   - Semantic similarity search using sentence transformers
   - Cypher query generation from natural language
   - Cross-encoder reranking for improved relevance
   - Streaming response generation

3. **Dual User Interfaces**:
   - **React Web Application**: Modern SPA for visual interaction
   - **Chainlit Interface**: Conversational AI chatbot for queries

4. **Scalable Architecture**:
   - Neo4j graph database for efficient relationship traversal
   - Batch processing for large datasets
   - Docker containerization for deployment

5. **Comprehensive Evaluation**:
   - Multi-metric evaluation framework (BLEU, ROUGE, BERT-Score, etc.)
   - Quality assurance pipeline

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  React Frontend  │         │ Chainlit Chat UI │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼──────────────┐
            │    Flask API Backend        │
            │  (semantic_search_api.py)   │
            │                             │
            │  - Semantic Search          │
            │  - Graph Management         │
            │  - Embedding Generation     │
            │  - LLM Integration          │
            └──────────────┬──────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
     ┌──────▼──────┐            ┌────────▼────────┐
     │   Neo4j     │            │  Google Gemini  │
     │  Database   │            │   LLM API       │
     │             │            │                 │
     │ - Knowledge │            │ - Response Gen  │
     │   Graph     │            │ - Enhancement   │
     │ - Vector    │            │                 │
     │   Search    │            │                 │
     └─────────────┘            └─────────────────┘
```

### 2.2 Core Modules

#### 2.2.1 Knowledge Graph Construction
**File**: `create_graph.py`, `create_graph_rdf.py`

**Purpose**: Transform raw data into Neo4j graph structures

**Key Classes**:
- `Neo4jBatchImporter`: Handles batch import with retry logic and connection management
- `process_json_in_chunks()`: Memory-efficient processing of large AAS JSON files
- `process_rdf_file()`: RDF triple processing into graph nodes and relationships

**Features**:
- Batch processing (configurable batch size, default 500)
- Automatic retry on connection failures
- Support for AAS derivation relationships (HAS_MODEL)
- Property node creation (Manufacturer, EnergyConsumption, Drilling, etc.)

**Graph Schema** (AAS Format):
```cypher
(:Asset {id, idShort, description, assetKind, availability})
  -[:MANUFACTURED_BY]->(:Manufacturer {name})
  -[:HAS_ENERGY_CONSUMPTION]->(:EnergyConsumption {details})
  -[:HAS_DRILLING]->(:Drilling {details})
  -[:HAS_CIRCLE_CUTTING]->(:CircleCutting {details})
  -[:HAS_SAWING]->(:Sawing {details})
  -[:HAS_AVAILABILITY]->(:Availability {status})
  -[:HAS_MODEL]->(:Asset)
```

#### 2.2.2 Embedding Generation
**File**: `gen_embeddings.py`, `gen_embeddings_rdf.py`

**Purpose**: Generate semantic embeddings for all graph nodes

**Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- Embedding dimension: 384
- Fast inference time
- Good balance of quality and speed

**Process**:
1. Extract all nodes from Neo4j with relationships
2. Prepare contextual text representation for each node
3. Generate embeddings in batches (default: 50 nodes)
4. Store embeddings as node properties in Neo4j

**Text Preparation Strategy**:
- **Asset nodes**: Combine ID, description, asset kind, and related properties
- **Property nodes**: Include type, details/name, and connected assets
- **Relationship context**: Include incoming/outgoing relationship information

**Example**:
```python
# Asset node text representation:
"ID: Drilling Machine #5 Description: High-precision drilling machine 
Asset Kind: Instance Availability Status: True HAS_DRILLING: 
Diameter: 10mm, Depth: 50mm"
```

#### 2.2.3 Semantic Search API
**File**: `semantic_search_api.py` (2778 lines)

**Core Endpoint**: `/semantic-search`

**Search Pipeline**:
1. **Query Embedding**: Convert user query to vector using sentence transformer
2. **Vector Similarity Search**: 
   - Cosine similarity computation in Cypher
   - Configurable similarity threshold (default: 0.5)
   - Returns top-N results (default: 15)
3. **Result Reranking**: Cross-encoder for improved relevance ordering
4. **Context Enrichment**: Gather related nodes and relationships
5. **LLM Enhancement**: Generate natural language response using Google Gemini
6. **Streaming Response**: Server-Sent Events (SSE) for real-time output

**Advanced Features**:

**Dynamic Query Generation**:
```python
def generate_structured_cypher_query(schema: dict, 
                                    similarity_threshold: float = 0.5, 
                                    n_results: int = 15) -> str
```
- Analyzes graph schema at runtime
- Generates optimized Cypher queries
- Handles optional relationships
- Aggregates connected models

**Fallback Mechanisms**:
- Universal fallback query for schema-agnostic search
- Text-based fuzzy matching when embeddings unavailable
- Graceful degradation on API failures

**API Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/semantic-search` | POST | Execute semantic search with streaming response |
| `/create-graph` | POST | Upload and create knowledge graph |
| `/delete-graph` | POST | Clear Neo4j database |
| `/generate-embeddings` | POST | Start embedding generation |
| `/generate-embeddings-status` | GET | Check embedding progress |
| `/set-graph-format` | POST | Change active graph format |
| `/progress` | GET | SSE stream for file upload progress |

#### 2.2.4 User Interfaces

**Chainlit Interface** (`app.py`):
- Chat-based interaction
- Special commands: `/delete-kg`, `/upload-kg`, `/graph-format`
- Real-time progress tracking
- Session-based format selection

**React Frontend** (`semantic-inference/src/`):
- Three-tab interface:
  1. **Semantic Search**: Query input with streaming responses
  2. **Manage Knowledge Graph**: File upload, format selection, deletion
  3. **Generate Embeddings**: Trigger and monitor embedding creation
- Progress indicators for all operations
- Format selection (CSV/AAS/RDF)

### 2.3 Data Flow

**Knowledge Graph Creation Flow**:
```
User Upload → Flask API → Format Detection → 
Batch Processing → Neo4j Import → Embedding Generation → Ready
```

**Search Query Flow**:
```
User Query → Embedding → Neo4j Vector Search → 
Result Retrieval → Reranking → Context Gathering → 
LLM Enhancement → Streaming Response → User
```

---

## 3. Technical Specifications

### 3.1 Technology Stack

**Backend**:
- Python 3.9+
- Flask 2.3.3 (Web framework)
- Neo4j 5.13.0 (Graph database)
- Sentence Transformers 2.2.2 (Embeddings)
- Google Generative AI 0.3.2 (LLM)

**Frontend**:
- React 18+ (Web framework)
- Axios (HTTP client)
- Chainlit 1.0.0 (Chat interface)

**Infrastructure**:
- Docker & Docker Compose
- Neo4j Community Edition 5.15

### 3.2 Database Schema

**Neo4j Configuration**:
- Memory heap: 512MB initial, 2GB max
- Page cache: 1GB
- APOC plugin enabled
- Vector search capability

**Constraints and Indexes**:
```cypher
// Unique constraint on Asset ID
CREATE CONSTRAINT asset_id_unique IF NOT EXISTS 
FOR (a:Asset) REQUIRE a.id IS UNIQUE;

// Index on embeddings (implicit via vector search)
// Index on frequently queried properties
CREATE INDEX asset_idshort IF NOT EXISTS 
FOR (a:Asset) ON (a.idShort);
```

### 3.3 API Configuration

**Environment Variables**:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
GEMINI_API_KEY=<api_key>
```

**Flask Configuration**:
- Host: 127.0.0.1
- Port: 5001
- CORS enabled for frontend communication

### 3.4 Performance Characteristics

**Scalability**:
- Supports knowledge graphs with 100,000+ nodes
- Batch processing handles large files (tested up to 35MB CSV)
- Concurrent query handling via Flask threading

**Response Times** (typical):
- Semantic search: 1-3 seconds
- Embedding generation: ~0.5 seconds per batch of 50 nodes
- Graph creation: Varies by file size (1000 records ≈ 10 seconds)

**Resource Requirements**:
- RAM: Minimum 4GB, recommended 8GB
- Storage: ~100MB per 10,000 nodes (including embeddings)
- CPU: Multi-core recommended for parallel processing

---

## 4. Evaluation Framework

### 4.1 Metrics Implemented
**File**: `evaluation.py`

The system includes comprehensive evaluation against ground truth:

**Lexical Metrics**:
- **BLEU Score**: N-gram precision with smoothing
- **ROUGE** (1/2/L): Recall-oriented n-gram and longest common subsequence
- **Cosine Similarity**: TF-IDF vector comparison

**Semantic Metrics**:
- **BERT Score**: Contextualized embedding similarity (Precision, Recall, F1)
- **BLEURT**: Learned metric based on BERT
- **Sentence-BERT**: Sentence embedding cosine similarity

**Token-based Metrics**:
- Precision, Recall, F1-Score
- Exact Match accuracy

### 4.2 Evaluation Process

1. Load evaluation dataset (`evaluation.json`)
2. Query API for each test case
3. Compute all metrics against expected output
4. Generate CSV report with per-query results
5. Calculate and log averaged metrics

**Example Results**:
```
BLEU: 0.3245
ROUGE-L: 0.5678
BERT F1: 0.7123
Sentence-BERT Similarity: 0.8234
```

### 4.3 Visualization
**Files**: `eval_visualization.ipynb`, `evaluation.ipynb`

Jupyter notebooks for:
- Metric distribution analysis
- Query-by-query performance visualization
- Comparative analysis across different configurations

---

## 5. Deployment

### 5.1 Docker Deployment

**Services**:
1. **Neo4j**: Graph database with APOC plugin
2. **API**: Flask backend in containerized environment

**Start System**:
```bash
docker-compose up -d
```

**Access Points**:
- Neo4j Browser: http://localhost:7474
- Flask API: http://localhost:5001
- React Frontend: http://localhost:3000 (if started separately)

### 5.2 Local Development

**Prerequisites**:
- Python 3.9+
- Node.js 14+ (for React frontend)
- Neo4j Desktop or Docker

**Backend Setup**:
```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask API
python semantic_search_api.py

# Start Chainlit interface (alternative)
chainlit run app.py
```

**Frontend Setup**:
```bash
cd semantic-inference
npm install
npm start
```

### 5.3 Configuration

**Graph Format Selection**:
Three modes supported, selectable at runtime:
- `csv`: Standard tabular data
- `aas`: Asset Administration Shell JSON
- `rdf`: RDF triples (JSONL format)

**Embedding Model**:
Default: `all-MiniLM-L6-v2`
- Can be changed in `gen_embeddings.py` and `gen_embeddings_rdf.py`
- Alternative: `all-mpnet-base-v2` (higher quality, slower)

**LLM Model**:
Uses Google Gemini API (configurable model version in `semantic_search_api.py`)

---

## 6. Usage Examples

### 6.1 Creating a Knowledge Graph

**Via Chainlit**:
```
User: /upload-kg
[Upload AAS JSON file]
System: Processing... 1000 of 1000 items processed
System: Knowledge graph created successfully!
```

**Via React UI**:
1. Navigate to "Manage Knowledge Graph" tab
2. Select graph format (CSV/AAS/RDF)
3. Upload file or paste text
4. Monitor progress bar
5. Generate embeddings if needed

### 6.2 Performing Semantic Search

**Example Query**:
```
User: "Which drilling machines have energy consumption above 200W?"
```

**System Response**:
```
Based on the knowledge graph, I found 3 drilling machines with high energy consumption:

1. Drilling Machine #5 (ID: DM-005)
   - Energy Consumption: Voltage: 230V, Current: 1.2A (276W)
   - Drilling Capacity: Diameter: 10mm, Depth: 50mm
   - Availability: Online

2. Drilling Machine #12 (ID: DM-012)
   - Energy Consumption: Voltage: 220V, Current: 1.1A (242W)
   - Drilling Capacity: Diameter: 8mm, Depth: 40mm
   - Availability: Online

3. Drilling Machine #8 (ID: DM-008)
   - Energy Consumption: Voltage: 240V, Current: 0.9A (216W)
   - Drilling Capacity: Diameter: 12mm, Depth: 60mm
   - Availability: Offline

These machines exceed the 200W threshold and are suitable for heavy-duty operations.
```

### 6.3 Advanced Queries

**Relationship Traversal**:
```
User: "Show me all machines manufactured by Siemens with circle cutting capability"
```

**Aggregation**:
```
User: "What is the average drilling depth across all drilling machines?"
```

**Availability Check**:
```
User: "List all available sawing machines"
```

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **LLM Dependency**: Requires external API (Google Gemini) for enhanced responses
2. **Single-Language Support**: Currently optimized for English queries
3. **Static Schema**: Graph schema is predefined per format type
4. **Embedding Update**: Full regeneration required on data updates
5. **No Authentication**: Production deployment requires authentication layer

### 7.2 Planned Enhancements

1. **Multi-modal Support**: Image and document integration
2. **Real-time Updates**: Incremental embedding updates
3. **Advanced Analytics**: Dashboard with usage statistics
4. **Custom Schema Builder**: GUI for defining custom graph schemas
5. **Federated Learning**: Privacy-preserving collaborative learning
6. **Multi-tenancy**: Support for multiple isolated knowledge bases

---

## 8. Troubleshooting

### 8.1 Common Issues

**Neo4j Connection Errors**:
- Verify Neo4j is running: `docker ps` or Neo4j Desktop
- Check credentials in `.env` file
- Ensure firewall allows port 7687

**Embedding Generation Fails**:
- Check available memory (requires ~2GB for large graphs)
- Verify sentence-transformers installation
- Monitor progress in logs

**Slow Search Performance**:
- Reduce similarity threshold (fewer results)
- Optimize Neo4j memory settings
- Consider using cross-encoder reranking selectively

### 8.2 Debugging

**Enable Detailed Logging**:
```python
logging.basicConfig(level=logging.DEBUG)
```

**Neo4j Query Profiling**:
```cypher
PROFILE MATCH (n:Asset) 
WHERE n.embedding IS NOT NULL 
RETURN n LIMIT 10;
```

---

## 9. Security Considerations

### 9.1 Data Privacy

- All data stored locally in Neo4j
- No data sent to external services except LLM API calls
- API keys should be stored in environment variables

### 9.2 Production Deployment

Recommended additions:
1. Authentication middleware (JWT/OAuth)
2. Rate limiting on API endpoints
3. HTTPS/TLS encryption
4. Input validation and sanitization
5. CORS restriction to specific origins
6. Database access controls

---

## 10. References and Resources

### 10.1 Key Technologies

- **Neo4j**: https://neo4j.com/docs/
- **Sentence Transformers**: https://www.sbert.net/
- **Flask**: https://flask.palletsprojects.com/
- **Chainlit**: https://docs.chainlit.io/
- **React**: https://react.dev/

### 10.2 Research Papers

1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)
2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
3. Asset Administration Shell Specification (Platform Industrie 4.0)

---

## 11. Conclusion

The HumAIne Semantic Inference System represents a robust, scalable solution for making complex industrial data queryable through natural language. By combining knowledge graphs, semantic search, and large language models, it bridges the gap between structured data and human-friendly interfaces.

The system's modular architecture, comprehensive evaluation framework, and support for multiple data formats make it adaptable to various industrial use cases beyond the initial Asset Administration Shell focus.

---

**Document Version**: 1.0  
**Last Updated**: February 16, 2026  
**Maintained By**: HumAIne Project Team  
**License**: See LICENSE file in repository
