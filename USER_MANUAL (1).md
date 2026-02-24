# Semantic Inference System - User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [User Interfaces](#user-interfaces)
4. [Working with Knowledge Graphs](#working-with-knowledge-graphs)
5. [Performing Searches](#performing-searches)
6. [Advanced Features](#advanced-features)
7. [Tips and Best Practices](#tips-and-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

### 1.1 What is the Semantic Inference System?

The Semantic Inference System is an AI-powered tool that helps you find information in complex industrial databases using simple, natural language questions. Instead of writing database queries or navigating complex menus, you can simply ask questions like:

- "Which drilling machines are currently available?"
- "Show me all machines manufactured by Siemens"
- "What's the energy consumption of Machine #5?"

The system understands your question, searches the knowledge base, and provides a comprehensive, easy-to-understand answer.

### 1.2 Key Concepts

**Knowledge Graph**: A database that stores information as a network of connected entities. For example, machines are connected to their manufacturers, capabilities, and specifications.

**Semantic Search**: Smart search that understands the meaning of your question, not just keyword matching.

**Embeddings**: Mathematical representations of data that help the system understand similarity and relevance.

**Streaming Response**: Answers appear word-by-word as they're generated, like a conversation.

### 1.3 What You Can Do

- ✅ Upload industrial asset data (CSV, AAS, RDF formats)
- ✅ Create searchable knowledge graphs automatically
- ✅ Ask questions in natural language
- ✅ Get AI-enhanced answers with full context
- ✅ Manage and update your knowledge base

---

## 2. Getting Started

### 2.1 Accessing the System

**Web Interface**:
1. Open your web browser
2. Navigate to: `http://localhost:3000`
3. You'll see the Semantic Search interface

**Chat Interface** (Chainlit):
1. Ask your administrator to start Chainlit
2. Open browser to the provided URL
3. Start chatting with the assistant

### 2.2 System Requirements

**To Use the System**:
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for AI features)
- No special software needed

**For Administrators**:
- See DEPLOYMENT_GUIDE.md for installation

### 2.3 First-Time Setup

If this is a new installation:

1. **Upload Data**: Start by uploading your first dataset
2. **Generate Embeddings**: Create searchable vectors
3. **Test Search**: Try a simple query
4. **Explore**: Learn what types of questions work best

---

## 3. User Interfaces

### 3.1 React Web Application

The web app has three main tabs:

#### Tab 1: Semantic Search

**Purpose**: Ask questions and get answers

**Layout**:
```
┌─────────────────────────────────────────┐
│  [Search Box]              [Search]     │
├─────────────────────────────────────────┤
│                                         │
│  Response Area                          │
│  (Streaming answer appears here)        │
│                                         │
└─────────────────────────────────────────┘
```

**How to Use**:
1. Type your question in the search box
2. Click "Search" or press Enter
3. Watch the answer appear in real-time
4. Read the complete response

#### Tab 2: Manage Knowledge Graph

**Purpose**: Upload and manage your data

**Sections**:

1. **Delete Knowledge Graph**
   - Button to clear all data
   - Use with caution!
   - Useful for starting fresh

2. **Graph Format Selection**
   - Dropdown menu
   - Options: CSV, AAS, RDF
   - Choose before uploading

3. **Upload File**
   - Click "Choose File"
   - Select your data file
   - Click "Upload"
   - Watch progress bar

4. **Enter Text**
   - Alternative to file upload
   - Paste CSV or text data
   - Click "Submit"

#### Tab 3: Generate Embeddings

**Purpose**: Make your data searchable

**Process**:
1. Click "Start Generating Embeddings"
2. Progress bar shows status
3. Wait for completion (may take several minutes)
4. Success message appears

### 3.2 Chainlit Chat Interface

**Purpose**: Conversational interaction

**Features**:
- Natural conversation flow
- Special commands (start with `/`)
- Real-time responses
- Session history

**Special Commands**:
```
/delete-kg          - Delete knowledge graph
/upload-kg          - Upload a file
/graph-format csv   - Change to CSV format
/graph-format aas   - Change to AAS format
/graph-format rdf   - Change to RDF format
```

**Example Conversation**:
```
You: /upload-kg
Assistant: Please upload your knowledge graph file...
[You upload file]
Assistant: Processing... 500 of 1000 items processed...
Assistant: Knowledge graph created successfully!

You: Which machines are available?
Assistant: Based on the knowledge graph, there are 15 
available machines...
```

---

## 4. Working with Knowledge Graphs

### 4.1 Understanding Data Formats

#### CSV Format

**When to Use**: Simple tabular data

**Example**:
```csv
name,manufacturer,energy_consumption,availability
Machine A,Siemens,250W,True
Machine B,Bosch,180W,False
```

**Requirements**:
- First row should be headers
- Comma-separated values
- UTF-8 encoding

#### AAS Format (Asset Administration Shell)

**When to Use**: Industry 4.0 asset descriptions

**Characteristics**:
- JSON format
- Contains assets and submodels
- Rich metadata and relationships

**Example Structure**:
```json
{
  "assetAdministrationShells": [...],
  "submodels": [...]
}
```

#### RDF Format

**When to Use**: Semantic web data, ontologies

**Characteristics**:
- Triple-based (subject-predicate-object)
- JSONLD format
- Supports complex relationships

### 4.2 Uploading Your Data

#### Step-by-Step Process

**Using Web Interface**:

1. **Prepare Your File**
   - Ensure data is clean
   - Check file format
   - Verify file size (< 100MB recommended)

2. **Navigate to Manage Tab**
   - Click "Manage Knowledge Graph"

3. **Select Format**
   - Choose CSV, AAS, or RDF
   - Must match your file

4. **Upload**
   - Click "Choose File"
   - Select your file
   - Click "Upload"

5. **Monitor Progress**
   - Progress bar shows status
   - Don't close the browser
   - Wait for completion message

6. **Generate Embeddings**
   - Switch to "Generate Embeddings" tab
   - Click "Start Generating Embeddings"
   - Wait for completion

**Using Chainlit**:

```
You: /upload-kg
Assistant: Please upload your knowledge graph file...
[Upload dialog appears]
[Select and upload file]
Assistant: Processing your file...
Assistant: Knowledge graph created successfully!
```

### 4.3 Managing Your Data

#### Viewing Data

**Via Neo4j Browser** (if you have access):
1. Navigate to http://localhost:7474
2. Run query: `MATCH (n) RETURN n LIMIT 100`
3. Visualize the graph

#### Updating Data

**Option 1: Delete and Re-upload**
- Safest method
- Click "Delete Knowledge Graph"
- Upload new file

**Option 2: Incremental Update**
- Upload additional data
- New nodes are added
- Existing nodes are updated

#### Deleting Data

**Warning**: This cannot be undone!

**Steps**:
1. Go to "Manage Knowledge Graph" tab
2. Click "Delete Knowledge Graph"
3. Confirm the action
4. All data is removed

---

## 5. Performing Searches

### 5.1 Writing Good Queries

#### Query Types

**1. Simple Fact Questions**
```
✅ Good: "What is the energy consumption of Machine #5?"
✅ Good: "List all available drilling machines"
❌ Poor: "machine 5"
❌ Poor: "list"
```

**2. Filtered Searches**
```
✅ Good: "Which machines have energy consumption above 200W?"
✅ Good: "Show me unavailable sawing machines"
❌ Poor: "high energy"
❌ Poor: "broken machines"
```

**3. Relationship Queries**
```
✅ Good: "What machines are manufactured by Siemens?"
✅ Good: "Show all machines with circle cutting capability"
❌ Poor: "Siemens stuff"
❌ Poor: "cutting things"
```

**4. Aggregation Questions**
```
✅ Good: "How many drilling machines are there?"
✅ Good: "What's the average energy consumption?"
❌ Poor: "count"
❌ Poor: "average"
```

### 5.2 Understanding Results

#### Response Structure

A typical response includes:

1. **Direct Answer**: Main information requested
2. **Supporting Details**: Specifications, relationships
3. **Context**: Additional relevant information

**Example**:
```
Query: "Which drilling machines are available?"

Response:
"Based on the knowledge graph, there are 3 available 
drilling machines:

1. Drilling Machine #5 (ID: DM-005)
   - Manufacturer: Siemens
   - Energy Consumption: 276W (230V, 1.2A)
   - Drilling Capacity: Diameter 10mm, Depth 50mm
   - Availability: Online

2. Drilling Machine #12 (ID: DM-012)
   - Manufacturer: Bosch
   - Energy Consumption: 242W (220V, 1.1A)
   - Drilling Capacity: Diameter 8mm, Depth 40mm
   - Availability: Online

3. Drilling Machine #8 (ID: DM-008)
   - Manufacturer: Festo
   - Energy Consumption: 216W (240V, 0.9A)
   - Drilling Capacity: Diameter 12mm, Depth 60mm
   - Availability: Online

All three machines are currently operational and ready 
for use."
```

#### Streaming Responses

- Answers appear word-by-word
- You can start reading immediately
- Cursor (▋) shows active generation
- Final answer appears when complete

### 5.3 Query Examples by Use Case

#### Asset Discovery
```
"List all machines in the knowledge base"
"What types of machines are available?"
"Show me all assets manufactured in 2023"
```

#### Technical Specifications
```
"What are the specifications of Machine #10?"
"Show drilling capabilities of all machines"
"Which machines support circle cutting?"
```

#### Operational Status
```
"Which machines are currently available?"
"Show me offline machines"
"List unavailable drilling machines"
```

#### Comparative Analysis
```
"Which machine has the highest energy consumption?"
"Compare drilling depths across all machines"
"What's the difference between Machine #5 and #10?"
```

#### Manufacturer Information
```
"What machines are made by Siemens?"
"List all manufacturers in the database"
"Show Bosch drilling machines"
```

---

## 6. Advanced Features

### 6.1 Format-Specific Queries

#### CSV Queries
- Focus on column names in your data
- Reference exact values when filtering

#### AAS Queries
```
"Show machines with derivedFrom template TMP-001"
"What are the submodels of Asset X?"
"List all instance assets"
```

#### RDF Queries
- Can query ontology relationships
- Reference class hierarchies
- Use property paths

### 6.2 Multi-Hop Queries

**Definition**: Questions that require traversing multiple relationships

**Examples**:
```
"Show machines manufactured by Siemens with drilling capability"
(Asset → Manufacturer + Asset → Drilling)

"What's the energy consumption of all available sawing machines?"
(Asset → Availability + Asset → Sawing + Asset → EnergyConsumption)

"List machines with circle cutting depth > 5cm made by Bosch"
(Asset → Manufacturer + Asset → CircleCutting)
```

### 6.3 Handling Ambiguity

**System Behavior**: When a query is ambiguous, the system:
1. Interprets based on available data
2. Provides the most relevant results
3. May include multiple interpretations

**Example**:
```
Query: "Show me machines"
[Ambiguous - what type?]

Response includes:
- All machine types found
- Counts by category
- Top relevant assets
```

**Best Practice**: Be specific to get better results

---

## 7. Tips and Best Practices

### 7.1 Query Optimization

**DO**:
- ✅ Use complete sentences
- ✅ Include specific details
- ✅ Reference exact names/IDs when known
- ✅ Ask one thing at a time
- ✅ Use technical terms from your domain

**DON'T**:
- ❌ Use single keywords
- ❌ Write overly complex questions
- ❌ Include multiple unrelated queries
- ❌ Use abbreviations without context
- ❌ Expect the system to guess missing info

### 7.2 Data Quality

**For Best Results**:

1. **Clean Data**
   - Remove duplicates before upload
   - Fix formatting issues
   - Ensure consistent naming

2. **Rich Metadata**
   - Include descriptions
   - Add relationship information
   - Use standard fields

3. **Regular Updates**
   - Keep data current
   - Remove obsolete entries
   - Update availability status

### 7.3 Performance Tips

**Fast Searches**:
- Simple queries are faster
- Specific IDs/names give quick results
- Aggregations may take longer

**Large Datasets**:
- Wait for embeddings to complete
- First search may be slower (caching)
- Subsequent searches are faster

**When to Regenerate Embeddings**:
- After major data updates (>10% changes)
- When search quality degrades
- After changing data format

### 7.4 Common Workflows

#### Daily Operations
1. Check system status
2. Perform routine searches
3. Update availability status (if applicable)

#### Weekly Maintenance
1. Review data accuracy
2. Update changed assets
3. Check for orphaned nodes

#### Monthly Tasks
1. Full data refresh
2. Regenerate embeddings
3. Review search logs
4. Optimize based on usage patterns

---

## 8. Troubleshooting

### 8.1 Search Issues

#### Problem: No Results

**Possible Causes**:
- Knowledge graph is empty
- Query too specific
- Embeddings not generated

**Solutions**:
1. Verify data is uploaded
2. Check embeddings are generated
3. Try a broader query
4. Check spelling/names

#### Problem: Irrelevant Results

**Possible Causes**:
- Query too vague
- Limited data in graph
- Embeddings outdated

**Solutions**:
1. Be more specific in query
2. Add more data
3. Regenerate embeddings
4. Try different phrasing

#### Problem: Slow Responses

**Possible Causes**:
- Large dataset
- Complex query
- Server load

**Solutions**:
1. Wait patiently (up to 30 seconds)
2. Simplify query
3. Contact administrator if persistent

### 8.2 Upload Issues

#### Problem: Upload Fails

**Possible Causes**:
- File too large
- Wrong format
- Corrupted file
- Network issue

**Solutions**:
1. Check file size (< 100MB)
2. Verify format matches selection
3. Try smaller file
4. Check internet connection

#### Problem: Wrong Data Format Error

**Solutions**:
1. Verify file is correct format
2. Check format selector matches file
3. Validate file structure
4. Try example file first

### 8.3 Interface Issues

#### Problem: Page Not Loading

**Solutions**:
1. Check URL is correct
2. Verify server is running
3. Clear browser cache
4. Try different browser

#### Problem: Streaming Stops

**Solutions**:
1. Refresh page
2. Try query again
3. Check internet connection
4. Report to administrator

### 8.4 Getting Help

**Self-Help Resources**:
1. Check this user manual
2. Review example queries
3. Try similar searches
4. Test with known data

**When to Contact Support**:
- Persistent errors
- System crashes
- Data corruption
- Performance degradation

**What to Include in Support Request**:
- Error message (if any)
- What you were trying to do
- Steps to reproduce
- Screenshots (if applicable)
- Your query text

---

## 9. Reference

### 9.1 Quick Command Reference

**Chainlit Commands**:
```
/delete-kg               Delete all data
/upload-kg               Upload new file
/graph-format csv        Switch to CSV mode
/graph-format aas        Switch to AAS mode
/graph-format rdf        Switch to RDF mode
```

### 9.2 Common Query Patterns

**List Queries**:
```
"List all [entity type]"
"Show me [entity type] with [condition]"
"What [entity type] are [state]?"
```

**Count Queries**:
```
"How many [entity type] are there?"
"Count [entity type] with [condition]"
```

**Filter Queries**:
```
"Show [entity type] where [property] > [value]"
"Find [entity type] with [property] = [value]"
```

**Relationship Queries**:
```
"What [entities] are connected to [entity]?"
"Show [relationship] for [entity]"
```

### 9.3 Data Format Specifications

**CSV Requirements**:
- UTF-8 encoding
- Comma delimiter
- Header row required
- No special characters in headers

**AAS Requirements**:
- Valid JSON
- Contains assetAdministrationShells array
- Contains submodels array
- Follows AAS specification

**RDF Requirements**:
- JSONLD format
- Valid triples
- Standard RDF namespaces

---

## 10. Glossary

**Asset**: An industrial machine or equipment in the knowledge graph

**Embedding**: A numerical representation of data that enables semantic search

**Knowledge Graph**: A network database storing entities and their relationships

**Neo4j**: The graph database technology powering the system

**Query**: A question or search request in natural language

**RAG**: Retrieval-Augmented Generation - combining search with AI

**Semantic Search**: Search that understands meaning, not just keywords

**Streaming Response**: Answer that appears progressively as it's generated

**Submodel**: In AAS, a component containing specific asset information

---

## Appendix A: Example Datasets

### Sample Queries for Products Dataset

```
"Show all products in the electronics category"
"What products cost less than $100?"
"List products with rating above 4.5"
"Find products from brand XYZ"
```

### Sample Queries for AAS Dataset

```
"Which drilling machines are available?"
"Show energy consumption for all assets"
"List machines manufactured by Siemens"
"What are the capabilities of Machine #5?"
"Find all instance assets derived from template TMP-001"
```

---

## Appendix B: Video Tutorials

**Getting Started** (5 minutes):
- System overview
- First login
- Basic navigation

**Creating Your First Knowledge Graph** (10 minutes):
- Preparing data
- Upload process
- Embedding generation

**Advanced Searching** (15 minutes):
- Query techniques
- Interpreting results
- Multi-hop queries

**Data Management** (8 minutes):
- Updating data
- Deleting entries
- Best practices

---

**User Manual Version**: 1.0  
**Last Updated**: February 16, 2026  
**For**: HumAIne Semantic Inference System  
**Feedback**: Please report issues or suggestions to the technical team
