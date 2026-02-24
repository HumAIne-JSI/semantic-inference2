import os
import google.generativeai as genai

# Try to import OpenAI with version compatibility
try:
    from openai import OpenAI  # openai >= 1.0.0
    OPENAI_VERSION = "new"
    print("new openai")
except ImportError:
    try:
        import openai  # openai < 1.0.0
        OPENAI_VERSION = "old"
        print("old openai")
    except ImportError:
        openai = None
        OPENAI_VERSION = None
        print("no openai")

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship
import pandas as pd
from tqdm import tqdm
import re
import logging
from fuzzywuzzy import fuzz
import threading
from queue import Queue
import json
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
import tempfile
from create_graph import (
    Neo4jBatchImporter, 
    process_json_in_chunks, 
    _process_batch
)
from gen_embeddings import EnhancedKGEmbeddings
from create_graph_rdf import process_file_in_batches as process_rdf_file
from gen_embeddings_rdf import generate_and_store_embeddings
from neomodel import config as neomodel_config, db as neomodel_db

import httpx


def get_graph_schema():
    with driver.session() as session:
        nodes_query = "CALL db.schema.nodeTypeProperties()"
        rels_query = "CALL db.schema.relTypeProperties()"
        nodes = session.run(nodes_query).data()
        rels = session.run(rels_query).data()

    node_schema = {}
    for n in nodes:
        label = n["nodeType"]
        prop = n["propertyName"]
        if label not in node_schema:
            node_schema[label] = set()
        node_schema[label].add(prop)

    rel_schema = {}
    for r in rels:
        # Different Neo4j versions / procedures may expose different key names.
        # Collect safe fallbacks.
        rel_type = r.get("relType") or r.get("relationshipType") or r.get("relationship")
        start = r.get("sourceNodeType") or r.get("from") or r.get("startNodeType")
        end = r.get("targetNodeType") or r.get("to") or r.get("endNodeType")
        if not rel_type:
            # Skip rows we can't interpret
            continue
        if rel_type not in rel_schema:
            rel_schema[rel_type] = []
        rel_schema[rel_type].append((start, end))

    return {
        "nodes": {k: list(v) for k, v in node_schema.items()},
        "relationships": rel_schema
    }

def generate_structured_cypher_query(schema: dict, similarity_threshold: float = 0.5, n_results: int = 15) -> str:
    """Generate a structured Cypher query based on the graph schema that includes connected models."""
    if not schema or not schema.get("nodes"):
        # Fallback to basic embedding query
        return """
        MATCH (n)
        WITH n,
            CASE
                WHEN 'embedding' IN keys(n) THEN n.embedding
                ELSE head([k IN keys(n) WHERE toLower(k) CONTAINS 'embedding' | n[k]])
            END AS emb
        WHERE emb IS NOT NULL AND size(emb) = size($embedding)
        WITH n, emb,
            REDUCE(dot = 0.0, i IN RANGE(0, size(emb)-1) | dot + emb[i] * $embedding[i]) AS dot,
            SQRT(REDUCE(n1 = 0.0, i IN RANGE(0, size(emb)-1) | n1 + emb[i]*emb[i])) AS n1,
            SQRT(REDUCE(n2 = 0.0, i IN RANGE(0, size($embedding)-1) | n2 + $embedding[i]*$embedding[i])) AS n2
        WITH n, CASE WHEN n1 = 0 OR n2 = 0 THEN 0.0 ELSE dot / (n1 * n2) END AS similarity
        WHERE similarity > $similarity_threshold
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT $n
        """
    
    nodes = schema.get("nodes", {})
    relationships = schema.get("relationships", {})
    
    # Find the primary entity nodes (those that have embedding properties)
    primary_nodes = []
    for node_type, properties in nodes.items():
        if "embedding" in properties:
            # Clean node type (remove `: ` prefix if present)
            clean_type = node_type.replace(":`", "").replace("`", "").strip(":")
            primary_nodes.append(clean_type)
    
    if not primary_nodes:
        # No embedding-enabled nodes found, use fallback
        return generate_structured_cypher_query({}, similarity_threshold, n_results)
    
    # Use the first primary node as the main entity (usually Asset in AAS graphs)
    main_entity = primary_nodes[0]
    main_var = main_entity.lower()[0]  # 'a' for Asset, etc.
    
    # Build the base query with embedding similarity
    query_parts = [
        f"MATCH ({main_var}:{main_entity})",
        f"WHERE {main_var}.embedding IS NOT NULL",
        f"WITH {main_var},",
        f"REDUCE(dot = 0.0, i IN RANGE(0, SIZE({main_var}.embedding)-1) |",
        f"    dot + {main_var}.embedding[i] * $embedding[i]",
        f") / (",
        f"    SQRT(REDUCE(norm1 = 0.0, i IN RANGE(0, SIZE({main_var}.embedding)-1) |",
        f"        norm1 + {main_var}.embedding[i] * {main_var}.embedding[i]",
        f"    )) *",
        f"    SQRT(REDUCE(norm2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) |",
        f"        norm2 + $embedding[i] * $embedding[i]",
        f"    ))",
        f") AS similarity",
        f"WHERE similarity > {similarity_threshold}",
        ""
    ]
    
    # Add optional matches for relationships and connected entities
    optional_matches = []
    return_fields = []
    collect_parts = []
    
    # First add properties of main entity
    main_entity_key = None
    for key in nodes.keys():
        if main_entity in key:
            main_entity_key = key
            break
    
    if main_entity_key:
        main_props = nodes.get(main_entity_key, [])
        for prop in main_props:
            if prop != "embedding":  # Skip embedding property
                return_fields.append(f"{main_var}.{prop} AS {prop}")
    
    # Add similarity score
    return_fields.append("similarity AS score")
    
    # Process relationships to find connected entities
    model_relationships = []  # For HAS_MODEL type relationships
    other_relationships = []  # For other relationships
    
    for rel_type in relationships.keys():
        rel_name = rel_type.replace(":`", "").replace("`", "").strip(":")
        if "MODEL" in rel_name.upper():
            model_relationships.append(rel_name)
        else:
            other_relationships.append(rel_name)
    
    # Add optional matches for related entities (non-model relationships)
    used_vars = {main_var}
    for rel_name in other_relationships:
        # Find target node types that this relationship could connect to
        for target_node_key in nodes.keys():
            target_type = target_node_key.replace(":`", "").replace("`", "").strip(":")
            if target_type != main_entity:
                # Create unique variable name
                var_name = target_type.lower()[:3]
                counter = 1
                while var_name in used_vars:
                    var_name = f"{target_type.lower()[:2]}{counter}"
                    counter += 1
                used_vars.add(var_name)
                
                optional_matches.append(f"OPTIONAL MATCH ({main_var})-[:{rel_name}]->({var_name}:{target_type})")
                
                # Add fields from connected entities
                target_props = nodes.get(target_node_key, [])
                for prop in target_props:
                    if prop != "embedding":
                        return_fields.append(f"{var_name}.{prop} AS {rel_name.lower()}_{prop}")
    
    # Handle model relationships specially for connected_models collection
    for rel_name in model_relationships:
        # Model relationships often connect same type entities (instance -> template)
        instance_var = "instance"
        optional_matches.append(f"OPTIONAL MATCH ({instance_var}:{main_entity})-[:{rel_name}]->({main_var})")
        
        # Create collection for connected models
        if main_entity_key:
            instance_props = nodes.get(main_entity_key, [])
            collect_fields = []
            for prop in instance_props:
                if prop != "embedding":
                    collect_fields.append(f"{prop}: {instance_var}.{prop}")
            
            if collect_fields:
                collect_parts.append(f"COLLECT(DISTINCT {{{', '.join(collect_fields)}}}) AS connected_models")
    
    # If no model relationships found, create a simpler connected_models based on any reverse relationships
    if not collect_parts:
        # Look for any relationship that could represent instances
        for rel_name in other_relationships:
            if any(keyword in rel_name.upper() for keyword in ["HAS", "CONTAINS", "INCLUDES"]):
                # Try reverse relationship for instances
                instance_var = "inst"
                optional_matches.append(f"OPTIONAL MATCH ({instance_var}:{main_entity})-[:{rel_name}]->({main_var})")
                
                if main_entity_key:
                    instance_props = nodes.get(main_entity_key, [])
                    collect_fields = []
                    for prop in instance_props:
                        if prop != "embedding":
                            collect_fields.append(f"{prop}: {instance_var}.{prop}")
                    
                    if collect_fields:
                        collect_parts.append(f"COLLECT(DISTINCT {{{', '.join(collect_fields)}}}) AS connected_models")
                break
    
    # If still no connected_models, add an empty one
    if not collect_parts:
        collect_parts.append("[] AS connected_models")
    
    # Combine all return fields
    all_returns = return_fields + collect_parts
    
    # Build the complete query
    complete_query = "\n".join(query_parts) + "\n" + "\n".join(optional_matches) + "\n\n" + \
                    f"RETURN\n    {',\\n    '.join(all_returns)}\n" + \
                    f"ORDER BY similarity DESC\n" + \
                    f"LIMIT $n"
    
    return complete_query

def generate_universal_fallback_query(query_text: str) -> str:
    """
    Generate a universal fallback query that will always find something.
    This is the last resort when all other queries fail.
    """
    words = [word.lower().strip() for word in query_text.split() if len(word.strip()) > 2]
    
    if not words:
        # If no meaningful words, just return any nodes
        return """
        MATCH (n)
        RETURN n, 0.1 as score, labels(n)[0] as node_type
        LIMIT $n
        """
    
    # Create a very broad fuzzy search
    word_conditions = []
    for word in words[:3]:  # Limit to first 3 words to avoid overly complex queries
        word_conditions.append(f"toString(n) CONTAINS '{word}'")
        word_conditions.append(f"any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS '{word}')")
    
    where_clause = " OR ".join(word_conditions)
    
    return f"""
    MATCH (n)
    WHERE {where_clause}
    RETURN n, 0.3 as score, labels(n)[0] as node_type
    ORDER BY score DESC
    LIMIT $n
    """

def generate_fallback_fuzzy_query(query_text: str, schema: dict) -> str:
    """
    Generate a fallback fuzzy query when schema-aware generation fails.
    Creates a comprehensive fuzzy search across all available node types.
    """
    query_words = [word.lower().strip() for word in query_text.split() if len(word.strip()) > 2]
    
    # Get all node labels from schema
    node_labels = list(schema.get("nodes", {}).keys())
    if not node_labels:
        # Ultimate fallback if no schema available
        return """
        MATCH (n)
        WHERE toString(n) CONTAINS $query_text OR any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $query_text)
        RETURN n, 0.5 as score
        LIMIT 15
        """
    
    # Build fuzzy query for each node type
    match_clauses = []
    for label in node_labels:
        properties = schema["nodes"][label]
        
        # Create fuzzy conditions for this node type
        fuzzy_conditions = []
        
        # Add embedding similarity if available
        if "embedding" in properties:
            fuzzy_conditions.append(f"""
            REDUCE(dot = 0.0, i IN RANGE(0, SIZE(n.embedding)-1) | 
                dot + n.embedding[i] * $embedding[i]
            ) / (
                SQRT(REDUCE(norm1 = 0.0, i IN RANGE(0, SIZE(n.embedding)-1) | 
                    norm1 + n.embedding[i] * n.embedding[i]
                )) * 
                SQRT(REDUCE(norm2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
                    norm2 + $embedding[i] * $embedding[i]
                ))
            ) > 0.2""")
        
        # Add text-based fuzzy matching for string properties
        text_props = [prop for prop in properties if prop not in ["embedding", "id"]]
        if text_props:
            text_conditions = []
            for prop in text_props[:3]:  # Limit to first 3 properties to avoid overly complex queries
                text_conditions.extend([
                    f"toLower(toString(n.{prop})) CONTAINS toLower('{word}')" 
                    for word in query_words[:3]  # Limit to first 3 words
                ])
            
            if text_conditions:
                fuzzy_conditions.append(f"({' OR '.join(text_conditions)})")
        
        # If no specific conditions, fall back to generic search
        if not fuzzy_conditions:
            fuzzy_conditions.append(f"any(word in split(toLower($query_text), ' ') WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS word))")
        
        # Combine conditions for this label
        where_clause = " OR ".join(fuzzy_conditions)
        match_clauses.append(f"""
        MATCH (n:{label})
        WHERE {where_clause}
        RETURN n, 0.6 as score, '{label}' as node_type
        """)
    
    # Combine all matches with UNION
    full_query = " UNION ".join(match_clauses)
    full_query += """
    ORDER BY score DESC
    LIMIT $n
    """
    
    return full_query

def generate_schema_aware_cypher(query: str, schema: dict) -> str:
    prompt = f"""
You are a Cypher expert. Given the user's question and the Neo4j schema, generate a VALID Cypher query.

Schema:
{json.dumps(schema, indent=2)}

User Query:
{query}

Rules:
- DO NOT guess labels or properties not in the schema.
- Only use available node labels, relationships, and properties.
- Focus on accuracy, not creativity.
- Return ONLY the Cypher query.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    print(prompt)
    try:
        response = model.generate_content(prompt)
        cypher = response.text.strip().replace("```cypher", "").replace("```", "")
        return cypher
    except Exception as e:
        print(f"[ERROR] Schema-aware Cypher generation failed: {e}")
        return None



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI Provider Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Options: "gemini" or "openai"
logger.info(f"AI Provider configured: {AI_PROVIDER}")

# Set up environment variables for API keys
os.environ["GOOGLE_API_KEY"] = "your-api-key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")

# Configure AI providers
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai_client = None

if OPENAI_API_KEY and OPENAI_VERSION:
    try:
        if OPENAI_VERSION == "new":
            # OpenAI >= 1.0.0
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"OpenAI client initialized successfully (version: new, >=1.0.0)")
        elif OPENAI_VERSION == "old":
            # OpenAI < 1.0.0
            openai.api_key = OPENAI_API_KEY
            openai_client = openai  # Use the module directly for old version
            logger.info(f"OpenAI configured successfully (version: old, <1.0.0)")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI: {e}")
        openai_client = None
elif not OPENAI_API_KEY:
    logger.info("OpenAI API key not provided - OpenAI provider unavailable")

# Neo4j connection setup (replace with your connection details)
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "your-neo4j-password"
driver = GraphDatabase.driver(uri, auth=(username, password))
graph = Graph(uri, auth=(username, password))

# Configure neomodel connection (used by /semantic-search). Neomodel expects bolt:// scheme.
neomodel_config.DATABASE_URL = f"bolt://{username}:{password}@localhost:7687"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global queue to store progress updates
progress_queue = Queue()

# Initialize embedding models
model = SentenceTransformer('all-MiniLM-L6-v2')  # Bi-encoder for fast retrieval
embedding_queue = Queue()

# Initialize cross-encoder for re-ranking (more accurate but slower)
logger.info("Loading Cross-Encoder model for precision re-ranking...")
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("Cross-Encoder loaded successfully - precision re-ranking enabled")
except Exception as e:
    logger.warning(f"Failed to load Cross-Encoder: {e}. Falling back to bi-encoder only.")
    cross_encoder = None


global grpah_format
graph_format = "aas"

def generate_ai_response(prompt: str, stream: bool = True, provider: str = None):
    """
    Universal AI response generator that works with both Gemini and OpenAI (both old and new versions).
    
    Args:
        prompt: The prompt to send to the AI
        stream: Whether to stream the response
        provider: Override the global AI_PROVIDER (optional)
    
    Yields/Returns:
        For streaming: yields chunks with .text or .choices[0].delta.content
        For non-streaming: returns complete response text
    """
    global AI_PROVIDER, openai_client, OPENAI_VERSION
    
    # Use override provider if specified, otherwise use global
    current_provider = provider if provider else AI_PROVIDER
    
    try:
        if current_provider.lower() == "openai":
            if not openai_client:
                raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
            
            # Handle different OpenAI versions
            if OPENAI_VERSION == "new":
                # OpenAI >= 1.0.0 (new client-based API)
                response = openai_client.chat.completions.create(
                    model="gpt-4o",  # or "gpt-4o-mini" for faster/cheaper
                    messages=[{"role": "user", "content": prompt}],
                    stream=stream,
                    temperature=0.1
                )
                
                if stream:
                    return response
                else:
                    return response.choices[0].message.content
                    
            elif OPENAI_VERSION == "old":
                # OpenAI < 1.0.0 (old module-based API)
                response = openai_client.ChatCompletion.create(
                    model="gpt-4",  # Use gpt-4 for old version compatibility
                    messages=[{"role": "user", "content": prompt}],
                    stream=stream,
                    temperature=0.1
                )
                
                if stream:
                    return response
                else:
                    return response['choices'][0]['message']['content']
            else:
                raise ValueError("OpenAI version not detected properly")
        
        else:  # Default to Gemini
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                stream=stream,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            
            if stream:
                return response
            else:
                return response.text
    
    except Exception as e:
        logger.error(f"AI response generation failed with {current_provider}: {e}")
        raise


@app.route('/set-ai-provider', methods=['POST'])
@cross_origin()
def set_ai_provider():
    """API endpoint to switch between AI providers (gemini/openai)."""
    global AI_PROVIDER
    
    try:
        data = request.json
        provider = data.get('provider', 'gemini').lower()
        
        if provider not in ['gemini', 'openai']:
            return jsonify({
                "status": "error",
                "message": "Invalid provider. Choose 'gemini' or 'openai'"
            }), 400
        
        # Check if OpenAI is available when switching to it
        if provider == 'openai' and not openai_client:
            return jsonify({
                "status": "error",
                "message": "OpenAI client not available. Please set OPENAI_API_KEY environment variable."
            }), 400
        
        AI_PROVIDER = provider
        logger.info(f"AI provider switched to: {provider}")
        
        return jsonify({
            "status": "success",
            "message": f"AI provider set to {provider}",
            "current_provider": AI_PROVIDER
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get-ai-provider', methods=['GET'])
@cross_origin()
def get_ai_provider():
    """Get current AI provider and available providers."""
    return jsonify({
        "current_provider": AI_PROVIDER,
        "available_providers": ["gemini", "openai"],
        "openai_available": openai_client is not None
    }), 200


# ============================================================================
# Cross-Encoder Re-ranking Functions
# ============================================================================

def format_node_for_reranking(node_data):
    """
    Format a node for cross-encoder scoring.
    Creates a natural language representation emphasizing important properties.
    
    Args:
        node_data (dict): Node data with 'labels', 'properties', 'node_id', etc.
    
    Returns:
        str: Formatted text representation of the node
    """
    labels = node_data.get('labels', [])
    props = node_data.get('properties', {})
    
    # Build a natural language representation
    parts = []
    
    # Label information
    if labels:
        parts.append(f"Type: {', '.join(labels)}")
    
    # Key properties that are most relevant for matching
    important_keys = ['name', 'title', 'description', 'value', 'type', 
                      'label', 'text', 'content', 'idShort', 'id']
    
    for key in important_keys:
        if key in props and props[key]:
            parts.append(f"{key}: {props[key]}")
    
    # Add other properties
    for key, val in props.items():
        if key not in important_keys and val:
            # Limit property value length to avoid overwhelming the cross-encoder
            val_str = str(val)[:100]
            parts.append(f"{key}: {val_str}")
    
    return " | ".join(parts)


def rerank_with_cross_encoder(query_text, candidates, top_k=None):
    """
    Re-rank candidate nodes using cross-encoder for higher precision.
    
    This is the second stage of a two-stage retrieval:
    1. Bi-encoder (fast): Get initial candidates (~100 nodes)
    2. Cross-encoder (accurate): Re-rank to final top_k
    
    Args:
        query_text (str): User's search query
        candidates (list): List of node dicts from bi-encoder retrieval
        top_k (int): Number of top results to return (None = return all re-ranked)
    
    Returns:
        list: Re-ranked candidates with updated 'rerank_score' field
    """
    if not cross_encoder:
        logger.warning("Cross-encoder not available, skipping re-ranking")
        return candidates
    
    if not candidates:
        return candidates
    
    try:
        # Format candidates for cross-encoder
        formatted_candidates = [format_node_for_reranking(c) for c in candidates]
        
        # Create query-candidate pairs
        pairs = [[query_text, cand_text] for cand_text in formatted_candidates]
        
        # Get cross-encoder scores (relevance scores)
        logger.info(f"Re-ranking {len(candidates)} candidates with cross-encoder...")
        scores = cross_encoder.predict(pairs)
        
        # Combine scores with original data
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])
            # Keep original score for reference
            if 'score' in candidate:
                candidate['original_score'] = candidate['score']
        
        # Sort by re-ranking score (descending)
        reranked = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        # Return top_k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        logger.info(f"Re-ranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        return reranked
        
    except Exception as e:
        logger.error(f"Error in cross-encoder re-ranking: {e}")
        return candidates  # Fallback to original order


@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    """Semantic search via neomodel with streaming output; schema-agnostic and no py2neo in this path."""
    data = request.json or {}
    query_text = data.get('query', '')
    # Read requested top-K robustly and default to 16 if not provided
    # Accept from JSON body and querystring; support common synonyms
    raw_n = (
        data.get('n')
        or data.get('k')
        or data.get('top_k')
        or data.get('limit')
        or request.args.get('n')
        or request.args.get('k')
        or request.args.get('top_k')
        or request.args.get('limit')
        or request.args.get('topK')
    )
    n_source = 'body'
    if raw_n is None:
        n_source = 'default'
    elif request.args.get('n') or request.args.get('k') or request.args.get('top_k') or request.args.get('limit') or request.args.get('topK'):
        n_source = 'querystring'
    try:
        n_results = int(raw_n) if raw_n is not None else 10
    except Exception:
        n_results = 10 ##################################################ooo000#####################
    # Enforce a server-side minimum and maximum that can be tuned via env vars
    try:
        min_results = int(os.getenv('SEMANTIC_SEARCH_MIN_RESULTS', '10'))
    except Exception:
        min_results = 10
    # Optionally ignore client-provided n entirely (default: true)
    ignore_client_n = str(os.getenv('SEMANTIC_SEARCH_IGNORE_CLIENT_N', 'true')).lower() in {'1','true','yes','on'}
    # Allow explicit override via force_n (query/body) to respect client n
    force_n_flag = False
    try:
        fn_q = request.args.get('force_n')
        fn_b = data.get('force_n') if isinstance(data, dict) else None
        if fn_q is not None:
            force_n_flag = fn_q.lower() in {'1','true','yes','on'}
        elif fn_b is not None:
            force_n_flag = bool(fn_b)
    except Exception:
        force_n_flag = False
    if force_n_flag:
        ignore_client_n = False
    if ignore_client_n:
        n_results = min_results
        n_source = 'server'
    try:
        max_results = int(os.getenv('SEMANTIC_SEARCH_MAX_RESULTS', '10'))
    except Exception:
        max_results = 10
    # Cap to max_results to ensure we return at most 11 by default
    effective_n = min(max(n_results, min_results), max_results)
    if ignore_client_n:
        print(f"/semantic-search enforced_n_results: {effective_n} (min={min_results}, max={max_results}) force_n={force_n_flag}")
    else:
        print(f"/semantic-search requested n_results: {n_results} (source={n_source}, raw={raw_n}); capped_to: {effective_n} (min={min_results}, max={max_results}) force_n={force_n_flag}")


    if isinstance(query_text, dict) and 'output' in query_text:
        query_text = query_text['output']
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    if graph_format == "rdf":
        return fix_semantic_search_for_rdf(data)

    def generate():
        query_embedding = get_embedding(query_text)
        if not query_embedding:
            yield "data: " + json.dumps({"error": "Failed to get query embedding"}) + "\n\n"
            return

        # Emit initial meta about n handling
        try:
            meta_n = {
                'meta': {
                    'requested_n': n_results,
                    'effective_n': effective_n,
                    'min_results': min_results,
                    'max_results': max_results,
                    'ignore_client_n': ignore_client_n,
                    'force_n': force_n_flag
                }
            }
            yield f"data: {json.dumps(meta_n)}\n\n"
        except Exception:
            pass

        def nm_run(cypher: str, params: dict):
            try:
                rows, meta = neomodel_db.cypher_query(cypher, params)
                cols = [m['name'] for m in meta] if meta and isinstance(meta[0], dict) and 'name' in meta[0] else meta
                results = []
                for row in rows:
                    item = {}
                    for i, col in enumerate(cols):
                        val = row[i] if i < len(row) else None
                        if isinstance(val, (list, tuple)):
                            cleaned = []
                            for v in val:
                                if isinstance(v, dict):
                                    cleaned.append(v)
                                else:
                                    try:
                                        cleaned.append(dict(v))
                                    except Exception:
                                        cleaned.append(str(v))
                            val = cleaned
                        elif not isinstance(val, (str, int, float, bool, dict)) and val is not None:
                            try:
                                val = dict(val)
                            except Exception:
                                val = str(val)
                        item[col] = val
                    results.append(item)
                return results
            except Exception as e:
                logging.error(f"neomodel cypher_query failed: {e}")
                return []

        # ----------------------
        # Schema-agnostic, multi-pass retrieval
        # ----------------------
        def tokenize(text: str):
            import re
            raw = [t for t in re.split(r"[^a-zA-Z0-9]+", str(text).lower()) if t]
            return [t for t in raw if len(t) >= 4][:12]

        def count_token_hits(item: dict, tokens: list[str]) -> int:
            if not tokens:
                return 0
            def scan_obj(obj):
                score = 0
                if isinstance(obj, str):
                    s = obj.lower()
                    for tok in tokens:
                        if tok in s:
                            score += 1
                elif isinstance(obj, dict):
                    for v in obj.values():
                        score += scan_obj(v)
                elif isinstance(obj, list):
                    for v in obj:
                        score += scan_obj(v)
                return score
            score = 0
            if isinstance(item.get('props'), dict):
                score += scan_obj(item['props'])
            if isinstance(item.get('connected_nodes'), list):
                for cn in item['connected_nodes']:
                    if isinstance(cn, dict) and isinstance(cn.get('props'), dict):
                        score += scan_obj(cn['props'])
            if isinstance(item.get('labels'), list):
                score += scan_obj(" ".join(item['labels']))
            return score

        def dedupe_by_signature(items: list[dict]) -> list[dict]:
            seen = set()
            out = []
            for it in items:
                try:
                    sig = json.dumps({'labels': it.get('labels', []), 'props': it.get('props', {})}, sort_keys=True, default=str)
                except Exception:
                    sig = str(it.get('labels')) + '|' + str(it.get('props'))
                if sig not in seen:
                    seen.add(sig)
                    out.append(it)
            return out

        def run_embed_search(sim_threshold: float, per_label_limit: int) -> list[dict]:
            schema_q_local = (
                "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes "
                "WITH nodeLabels, propertyName, propertyTypes "
                "WHERE toLower(propertyName) CONTAINS 'embedding' "
                "   OR any(t IN propertyTypes WHERE toLower(t) CONTAINS 'float' AND toLower(t) CONTAINS 'list') "
                "UNWIND nodeLabels AS label "
                "RETURN label, collect(DISTINCT propertyName) AS props "
                "ORDER BY size(props) DESC"
            )
            meta_local = nm_run(schema_q_local, {})
            all_res = []
            if meta_local:
                for node_meta in meta_local:
                    label = node_meta.get('label')
                    props = node_meta.get('props') or []
                    emb = next((p for p in props if 'embedding' in p.lower()), (props[0] if props else None))
                    if label and emb:
                        q = f"""
                        MATCH (n:`{label}`)
                        WHERE n.`{emb}` IS NOT NULL
                        WITH n,
                             REDUCE(dot = 0.0, i IN RANGE(0, SIZE(n.`{emb}`)-1) |
                                 dot + n.`{emb}`[i] * $embedding[i]
                             ) / (
                                 SQRT(REDUCE(n1 = 0.0, i IN RANGE(0, SIZE(n.`{emb}`)-1) | n1 + n.`{emb}`[i] * n.`{emb}`[i])) *
                                 SQRT(REDUCE(n2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | n2 + $embedding[i] * $embedding[i]))
                             ) AS similarity
                        WHERE similarity > $threshold
                    OPTIONAL MATCH (n)-[r]-(m)
                    WITH n, similarity,
                        collect(DISTINCT {{labels: labels(m), props: properties(m), node_id: elementId(m), rel_type: type(r), direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END}}) AS connected_nodes
                    RETURN labels(n) AS labels, properties(n) AS props, connected_nodes, similarity AS score, elementId(n) AS node_id
                        ORDER BY score DESC
                        LIMIT $lim
                        """
                        label_res = nm_run(q, {"embedding": query_embedding, "threshold": float(sim_threshold), "lim": int(per_label_limit)})
                        all_res.extend(label_res)
            all_res.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_res

        tokens = tokenize(query_text)
        embed1 = run_embed_search(sim_threshold=0.3, per_label_limit=50)
        token_fallback_q = (
            "MATCH (n) "
            "WITH n, [k IN keys(n) WHERE n[k] IS NOT NULL AND size(toString(n[k]))>0 AND k <> 'embedding' | toLower(toString(n[k]))] AS texts "
            "WITH n, reduce(s=0, t IN texts | s + reduce(s2=0, tok IN $tokens | s2 + CASE WHEN t CONTAINS tok THEN 1 ELSE 0 END)) AS hits "
            "WHERE hits > 0 "
            "OPTIONAL MATCH (n)-[*1..2]-(m) "
            "WITH n, hits, collect(DISTINCT {labels: labels(m), props: properties(m), node_id: elementId(m)}) AS connected_nodes "
            "RETURN labels(n) AS labels, properties(n) AS props, connected_nodes, toFloat(hits) AS score, elementId(n) AS node_id "
            "ORDER BY score DESC LIMIT $limit"
        )
        token_fallback = nm_run(token_fallback_q, {"tokens": tokens, "limit": max(effective_n * 8, 200)}) if tokens else []
        combined = dedupe_by_signature(embed1 + token_fallback)
        if len(combined) < effective_n:
            embed2 = run_embed_search(sim_threshold=0.2, per_label_limit=150)
            combined = dedupe_by_signature(combined + embed2)
        else:
            embed2 = []
        token_hits_map = {}
        for it in combined:
            token_hits_map[id(it)] = count_token_hits(it, tokens)
        combined.sort(key=lambda x: (token_hits_map.get(id(x), 0), x.get('score', 0.0)), reverse=True)
        results_list = combined[:effective_n]

        # Promote connected nodes that show token evidence to primary results and fetch their neighborhoods
        #try:
        def fetch_node_full(nid: str):
            q = (
        "MATCH (n) WHERE elementId(n)=$id "
                "OPTIONAL MATCH (n)-[r]-(m) "
        "WITH n, collect(DISTINCT {labels: labels(m), props: properties(m), node_id: elementId(m), rel_type: type(r), direction: CASE WHEN startNode(r)=n THEN 'outgoing' ELSE 'incoming' END}) AS connected_nodes "
        "RETURN labels(n) AS labels, properties(n) AS props, connected_nodes, elementId(n) AS node_id"
            )
            res = nm_run(q, {"id": str(nid)})
            return res[0] if res else None

        def key_of(item: dict):
            if 'node_id' in item and item['node_id'] is not None:
                return ('id', str(item['node_id']))
            return ('sig', json.dumps({'labels': item.get('labels'), 'props': item.get('props')}, sort_keys=True, default=str))

        # Promotion of neighbors to primaries (skip when effective_n == 1 unless explicitly allowed)
        allow_promotion_single = False
        try:
            qp = request.args.get('allow_promotion_single')
            bp = data.get('allow_promotion_single') if isinstance(data, dict) else None
            if qp is not None:
                allow_promotion_single = qp.lower() in {'1','true','yes','on'}
            elif bp is not None:
                allow_promotion_single = bool(bp)
        except Exception:
            allow_promotion_single = False

        if effective_n == 1 and not allow_promotion_single:
            promoted_list = list(results_list)
            try:
                yield f"data: {json.dumps({'meta': {'promotion': 'skipped_for_single_result'}})}\n\n"
            except Exception:
                pass
        else:
            promoted = {key_of(it): it for it in results_list}
            for it in list(promoted.values()):
                for cn in (it.get('connected_nodes') or []):
                    # Simple evidence check using existing token scoring
                    nhits = count_token_hits({'props': cn.get('props', {}), 'labels': cn.get('labels', [])}, tokens)
                    if nhits >= 1 and cn.get('node_id') is not None:
                        full = fetch_node_full(cn['node_id'])
                        if full:
                            # carry over or combine score
                            full['score'] = max(full.get('score', 0.0), it.get('score', 0.0))
                            promoted[key_of(full)] = full

            promoted_list = list(promoted.values())
        # Re-rank by token evidence then similarity score
        promoted_list.sort(key=lambda x: (count_token_hits(x, tokens), x.get('score', 0.0)), reverse=True)
        
        # ============================================================================
        # Cross-Encoder Re-ranking (Stage 2: Precision)
        # ============================================================================
        # Two-stage retrieval:
        # 1. Bi-encoder (above): Fast retrieval of top candidates
        # 2. Cross-encoder (here): Accurate re-ranking for final results
        #
        # Strategy: Get more candidates than needed, then re-rank to effective_n
        # This improves precision by 15-25% without keyword matching
        rerank_pool_size = min(len(promoted_list), effective_n * 3)  # Re-rank 3x candidates
        rerank_candidates = promoted_list[:rerank_pool_size]
        
        if cross_encoder and len(rerank_candidates) > 1:
            logger.info(f"Re-ranking top {len(rerank_candidates)} candidates with cross-encoder...")
            reranked = rerank_with_cross_encoder(query_text, rerank_candidates, top_k=effective_n)
            
            # Check if re-ranking is helpful (hybrid approach)
            top_score = reranked[0].get('rerank_score', -999) if reranked else -999
            use_reranked = top_score >= -2.0  # Threshold for "useful" re-ranking
            
            if not use_reranked:
                logger.warning(f"Cross-encoder scores too low (top: {top_score:.4f}), falling back to semantic ranking")
                # Fall back to original semantic ranking
                results_list = promoted_list[:effective_n]
                fallback_used = True
            else:
                logger.info(f"Re-ranking successful. Top score: {top_score:.4f}")
                results_list = reranked[:effective_n]
                fallback_used = False
            
            # Send re-ranking metadata
            meta_rerank = {
                'meta': {
                    'reranking': {
                        'enabled': True,
                        'candidates_pool': len(rerank_candidates),
                        'top_rerank_score': top_score,
                        'fallback_to_semantic': fallback_used,
                        'score_improvement': (
                            reranked[0].get('rerank_score', 0) - reranked[0].get('original_score', 0)
                            if reranked and 'original_score' in reranked[0] and not fallback_used else 0
                        )
                    }
                }
            }
            yield f"data: {json.dumps(meta_rerank)}\n\n"
        else:
            # Fallback: cross-encoder not available or too few candidates
            results_list = promoted_list[:effective_n]
        
        #except Exception as _e:
            # Promotion is best-effort; proceed with base results if anything goes wrong
        #    pass

        #try:
        logger.info(f"Results list size after re-ranking/fallback: {len(results_list)}")
        meta_evt0 = {
            'meta': {
                'requested_n': n_results,
                'effective_n': effective_n,
                'embed1': len(embed1),
                'fallback_tokens': len(token_fallback),
                'embed2': len(embed2),
        'unique_after_merge': len(combined),
        'returned': len(results_list)
            }
        }
        yield f"data: {json.dumps(meta_evt0)}\n\n"
        #except Exception:
        #    pass

        logger.info(f"About to call fuzzy_match_results with {len(results_list)} results")
        matched_results = fuzzy_match_results(results_list, query_text)
        logger.info(f"After fuzzy_match_results: {len(matched_results)} matched results")
        
        # ------------------------------------------------------------------
        # Schema-agnostic serializer (preserve neighbors, trim verbosity)
        # - Keeps all connected_nodes (distance-1) intact
        # - Removes embeddings and volatile meta fields
        # - Trims very long string properties to a safe max length
        # - Drops None/empty values to save space
        # ------------------------------------------------------------------
        def _trim_value(v, max_len=400):
            try:
                if isinstance(v, str):
                    # Collapse whitespace/newlines to single spaces
                    s = " ".join(v.split())
                    if len(s) > max_len:
                        return s[:max_len] + "…"
                    return s
                return v
            except Exception:
                return v

        def _clean_props(props: dict, drop_keys=("embedding",), max_str_len=400):
            if not isinstance(props, dict):
                return props
            out = {}
            for k, v in props.items():
                if k in drop_keys:
                    continue
                if v is None:
                    continue
                # Skip empty strings or containers
                if isinstance(v, str) and v.strip() == "":
                    continue
                if isinstance(v, (list, dict)) and len(v) == 0:
                    continue
                out[k] = _trim_value(v, max_len=max_str_len)
            return out

        def serialize_results(items: list, instance_limit: int = 20) -> list:
            cleaned = []
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                new_it = {}
                # Always carry structure fields first
                if 'labels' in it:
                    new_it['labels'] = it['labels']
                if 'props' in it:
                    # Main node props: moderate cap
                    new_it['props'] = _clean_props(it.get('props') or {}, max_str_len=160)
                # Keep neighbors intact but cleaned
                cn = it.get('connected_nodes')
                if isinstance(cn, list):
                    new_cn = []
                    # Filter out generic instance nodes to reduce noise
                    type_nodes = []
                    instance_nodes = []
                    for cn_it in cn:
                        if not isinstance(cn_it, dict):
                            continue
                        props = cn_it.get('props') or {}
                        kind = props.get('assetKind', '')
                        # Separate Type nodes from Instance nodes
                        if kind == 'Type':
                            type_nodes.append(cn_it)
                        elif kind == 'Instance':
                            instance_nodes.append(cn_it)
                        else:
                            # Keep all non-Asset nodes (Manufacturer, Capability, etc.)
                            type_nodes.append(cn_it)
                    
                    # Limit instances based on parameter (default 20, but can be increased for counting queries)
                    filtered_cn = type_nodes + instance_nodes[:instance_limit]
                    
                    for cn_it in filtered_cn:
                        cn_new = {}
                        if 'labels' in cn_it:
                            cn_new['labels'] = cn_it['labels']
                        if 'props' in cn_it:
                            # Neighbor props: tighter cap (many entries)
                            cn_new['props'] = _clean_props(cn_it.get('props') or {}, max_str_len=80)
                        if 'rel_type' in cn_it:
                            cn_new['rel_type'] = cn_it['rel_type']
                        if 'direction' in cn_it:
                            cn_new['direction'] = cn_it['direction']
                        # Keep stable id if present (as string) for dedupe downstream
                        if cn_it.get('node_id') is not None:
                            cn_new['node_id'] = str(cn_it['node_id'])
                        if cn_new:
                            new_cn.append(cn_new)
                    new_it['connected_nodes'] = new_cn
                # Retain only minimal meta on the primary item
                # Keep node_id if present for dedup logic; drop other transient scores
                if it.get('node_id') is not None:
                    new_it['node_id'] = str(it['node_id'])
                cleaned.append(new_it)
            return cleaned

        # Apply serializer early to reduce payload before downstream steps
        # Determine instance limit based on query intent
        # Check if query needs instances (counting, availability queries)
        needs_instances_flag = False #ai_result.get('needs_instances', 'false') == 'true'
        query_lower = str(query_text).lower() if query_text else ''
        # Fallback: also check for counting/availability keywords
        if not needs_instances_flag:
            counting_keywords = ['count', 'how many', 'available', 'online', 'offline', 'unavailable']
            needs_instances_flag = any(kw in query_lower for kw in counting_keywords)
        
        instance_limit = 999999 if needs_instances_flag else 20  # Show all instances for counting queries
        
        serialized_before_chars = len(json.dumps(matched_results, default=str)) if matched_results else 0
        matched_results = serialize_results(matched_results, instance_limit=instance_limit)
        serialized_after_chars = len(json.dumps(matched_results, default=str)) if matched_results else 0
        try:
            yield f"data: {json.dumps({'meta': {'serializer': {'before_chars': serialized_before_chars, 'after_chars': serialized_after_chars, 'reduction_percent': (round((serialized_before_chars-serialized_after_chars)/serialized_before_chars*100,1) if serialized_before_chars else 0)}}})}\n\n"
        except Exception:
            pass

        # Compact encoding function (tokenized) - defined here to be available for relaxation steps
        def compact_encode_results(results: list) -> tuple:
            """
            Advanced compact encoding with token dictionaries to minimize repetition.
            Strategy:
              - Build dictionaries for property keys (K), labels (T), relationship types (R), directions (D)
              - Replace repeated strings with integer indices
              - Use short keys: L (labels idx list), P (prop key idx -> value), C (connections), r (rel idx), d (dir idx)
            Returns (encoded_json, original_len, encoded_len, dicts)
            """
            if not results:
                return ("[]", 0, 2, {})
            original_json = json.dumps(results, default=str)
            original_len = len(original_json)

            key_vocab = []  # property keys
            label_vocab = []  # node labels
            rel_vocab = []  # relationship types
            dir_vocab = []  # directions

            def idx_of(val, vocab):
                try:
                    if val not in vocab:
                        vocab.append(val)
                    return vocab.index(val)
                except Exception:
                    return None

            encoded_items = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                enc_item = {}
                # Labels
                labels = item.get('labels') or []
                if labels:
                    enc_item['L'] = [idx_of(lb, label_vocab) for lb in labels if lb is not None]
                # Props
                props = item.get('props') if isinstance(item.get('props'), dict) else {}
                if props:
                    ep = {}
                    for k, v in props.items():
                        ki = idx_of(k, key_vocab)
                        if ki is None:
                            continue
                        ep[str(ki)] = v
                    if ep:
                        enc_item['P'] = ep
                # Connections
                conns = []
                for cn in item.get('connected_nodes') or []:
                    if not isinstance(cn, dict):
                        continue
                    ecn = {}
                    clabels = cn.get('labels') or []
                    if clabels:
                        ecn['L'] = [idx_of(lb, label_vocab) for lb in clabels if lb is not None]
                    cprops = cn.get('props') if isinstance(cn.get('props'), dict) else {}
                    if cprops:
                        cp = {}
                        for k, v in cprops.items():
                            ki = idx_of(k, key_vocab)
                            if ki is None:
                                continue
                            cp[str(ki)] = v
                        if cp:
                            ecn['P'] = cp
                    rel_type = cn.get('rel_type')
                    if rel_type:
                        ecn['r'] = idx_of(rel_type, rel_vocab)
                    direction = cn.get('direction')
                    if direction:
                        ecn['d'] = idx_of(direction, dir_vocab)
                    if ecn:
                        conns.append(ecn)
                if conns:
                    enc_item['C'] = conns
                if enc_item:
                    encoded_items.append(enc_item)

            dicts = {
                'K': key_vocab,
                'T': label_vocab,
                'R': rel_vocab,
                'D': dir_vocab
            }
            payload = {'dicts': dicts, 'data': encoded_items}
            encoded_json = json.dumps(payload, default=str, separators=(',', ':'))
            encoded_len = len(encoded_json)
            return (encoded_json, original_len, encoded_len, dicts)

        # ------------------------------------------------------------------
        # Auto-relaxation: increment result count by 1 until compact length >= 25000
        # Skips if force_n is active or expansion disabled intentionally.
        # ------------------------------------------------------------------
        auto_relax_threshold = 25000
        auto_relax_enabled = True
        try:
            ar_q = request.args.get('auto_relax')
            ar_b = data.get('auto_relax') if isinstance(data, dict) else None
            if ar_q is not None:
                auto_relax_enabled = ar_q.lower() in {'1','true','yes','on'}
            elif ar_b is not None:
                auto_relax_enabled = bool(ar_b)
        except Exception:
            auto_relax_enabled = True
        # Do not auto-relax if force_n was used to enforce strict count
        if force_n_flag:
            auto_relax_enabled = False

        # Build a ranked source list (after promotion logic) for relaxing
        ranked_source = promoted_list if 'promoted_list' in locals() else results_list

        def _encode_for_length(items):
            enc, o_len, e_len, _dm = compact_encode_results(items)
            return enc, e_len

        # Initial encoding for current matched_results
        enc_str_initial, enc_len_initial = _encode_for_length(matched_results)
        if auto_relax_enabled and enc_len_initial < auto_relax_threshold:
            # Iteratively grow by 1 until threshold or source exhausted
            current_k = len(matched_results)
            iteration = 0
            while enc_len_initial < auto_relax_threshold and current_k < len(ranked_source):
                iteration += 1
                current_k += 1
                grow_slice = ranked_source[:current_k]
                # Re-run fuzzy + serialize for new slice
                grown = fuzzy_match_results(grow_slice, query_text)
                grown = serialize_results(grown, instance_limit=instance_limit)
                enc_str_new, enc_len_new = _encode_for_length(grown)
                meta_relax = {
                    'meta': {
                        'auto_relax_iteration': iteration,
                        'candidate_k': current_k,
                        'encoded_chars': enc_len_new,
                        'target_chars': auto_relax_threshold,
                        'met_target': enc_len_new >= auto_relax_threshold
                    }
                }
                try:
                    yield f"data: {json.dumps(meta_relax)}\n\n"
                except Exception:
                    pass
                if enc_len_new > enc_len_initial:
                    matched_results = grown
                    enc_str_initial = enc_str_new
                    enc_len_initial = enc_len_new
                if enc_len_initial >= auto_relax_threshold:
                    break
            # emit final meta state
            try:
                yield f"data: {json.dumps({'meta': {'auto_relax_final_k': len(matched_results), 'final_encoded_chars': enc_len_initial, 'target_chars': auto_relax_threshold}})}\n\n"
            except Exception:
                pass
        # Schema-agnostic compaction: keep only items that contain connected_nodes and dedupe
        #try:
        before_count = len(matched_results)
        logger.info(f"Before connected_nodes filtering: {before_count} results")
        filtered = []
        seen = set()
        for item in matched_results:
            if not isinstance(item, dict):
                continue
            cn = item.get('connected_nodes', None)
            if not (isinstance(cn, list) and len(cn) > 0):
                # Skip items without connected context to reduce overload
                logger.debug(f"Skipping item without connected_nodes: {item.get('labels', [])}")
                continue
            # Build a schema-agnostic signature using labels + props (minus embedding + volatile fields)
            labels = item.get('labels', [])
            props = dict(item.get('props', {}) or {})
            # Drop heavy/volatile keys from signature
            props.pop('embedding', None)
            for k in ['score', 'orig_score', 'similarity', 'node_id']:
                props.pop(k, None)
            try:
                sig = json.dumps({'labels': labels, 'props': props}, sort_keys=True, default=str)
            except Exception:
                sig = str(labels) + '|' + str(props)
            if sig in seen:
                continue
            seen.add(sig)
            filtered.append(item)

        # Always keep only items that have connected_nodes
        matched_results = filtered
        logger.info(f"After connected_nodes filtering: {len(matched_results)} results")
        
        # Check if we have empty results - this might be why response stops
        if len(matched_results) == 0:
            logger.error("NO RESULTS after filtering! This will cause empty response.")
            # Emergency fallback: use original results_list without filtering
            logger.warning("Emergency fallback: using results without connected_nodes filter")
            matched_results = results_list[:effective_n]
        
        # Emit diagnostic meta event for filtering
        #try:
        meta_evt_filter = {
            'meta': {
                'filtered_before': before_count,
                'filtered_after': len(matched_results),
                'filter_rule': 'keep_only_with_connected_nodes+dedupe',
                'emergency_fallback_used': len(filtered) == 0
            }
        }
        yield f"data: {json.dumps(meta_evt_filter)}\n\n"
        # except Exception:
            #     pass
        #except Exception:
            # Best-effort compaction; proceed if anything goes wrong
         #   pass
        # Optional stricter result-level compaction: collapse items with identical neighbor sets (schema-agnostic)
        strict_flag = False
        try:
            strict_flag = bool(data.get('compact_strict'))
        except Exception:
            pass
        if not strict_flag:
            qs_flag = request.args.get('compact_strict')
            if qs_flag is not None:
                strict_flag = qs_flag.lower() in {'1','true','yes','on'}

        if strict_flag and matched_results:
            def neighbor_set_signature(item: dict) -> str:
                acc = []
                for cn in item.get('connected_nodes') or []:
                    if not isinstance(cn, dict):
                        continue
                    rel = cn.get('rel_type')
                    # prefer stable neighbor identity
                    nid = cn.get('node_id')
                    if nid is None:
                        try:
                            nprops = dict(cn.get('props') or {})
                            nprops.pop('embedding', None)
                            nsig = json.dumps({'labels': cn.get('labels'), 'props': nprops}, sort_keys=True, default=str)
                        except Exception:
                            nsig = str(cn.get('labels')) + '|' + str(cn.get('props'))
                    else:
                        nsig = f"id:{nid}"
                    # ignore direction when forming signature to avoid bidirectional duplication
                    acc.append((rel, nsig))
                acc.sort()
                return json.dumps(acc, sort_keys=True)

            seen_ns = set()
            compacted = []
            for it in matched_results:
                ns = neighbor_set_signature(it)
                if ns in seen_ns:
                    continue
                seen_ns.add(ns)
                compacted.append(it)
            matched_results = compacted

            yield f"data: {json.dumps({'meta': {'strict_compaction': True, 'after': len(matched_results)}})}\n\n"
    # end strict compaction

        # Deduplicate connections: avoid bidirectional duplicates and nested repeats
        #try:
        seen_edges = set()  # undirected edge key: (min_id, max_id, rel_type) using elementId strings
        for item in matched_results:
            if not isinstance(item, dict):
                continue
            cn_list = item.get('connected_nodes')
            if not isinstance(cn_list, list):
                continue
            primary_id = item.get('node_id')
            local_seen = set()  # per-item seen by neighbor signature and rel_type
            deduped_cn = []
            for cn in cn_list:
                if not isinstance(cn, dict):
                    continue
                rel_type = cn.get('rel_type')
                # neighbor identity (prefer node_id, fallback to labels+props signature)
                n_id = cn.get('node_id')
                if n_id is None:
                    try:
                        n_sig = json.dumps({'labels': cn.get('labels'), 'props': cn.get('props')}, sort_keys=True, default=str)
                    except Exception:
                        n_sig = str(cn.get('labels')) + '|' + str(cn.get('props'))
                    neighbor_key = ('sig', n_sig)
                else:
                    neighbor_key = ('id', str(n_id))
                local_key = (neighbor_key, rel_type)
                if local_key in local_seen:
                    continue  # duplicate within same list
                # global undirected edge key, if both ids are available
                global_key = None
                if primary_id is not None and n_id is not None:
                    a, b = sorted([str(primary_id), str(n_id)])
                    global_key = (a, b, rel_type)
                    if global_key in seen_edges:
                        continue  # skip mirrored/bidirectional duplicate across results
                # accept this connection
                local_seen.add(local_key)
                if global_key is not None:
                    seen_edges.add(global_key)
                deduped_cn.append(cn)
            item['connected_nodes'] = deduped_cn
        #except Exception:
        #    pass

        # Strip only meta fields to shorten payload without losing domain info
        #try:
        remove_keys = {"node_id", "score", "orig_score", "similarity"}
        cleaned = []
        for item in matched_results:
            if not isinstance(item, dict):
                cleaned.append(item)
                continue
            base = {k: v for k, v in item.items() if k not in remove_keys}
            # Clean connected_nodes entries but keep their props intact
            if isinstance(base.get("connected_nodes"), list):
                new_cn = []
                for cn in base["connected_nodes"]:
                    if isinstance(cn, dict):
                        new_cn.append({k: v for k, v in cn.items() if k not in remove_keys})
                    else:
                        new_cn.append(cn)
                base["connected_nodes"] = new_cn
            cleaned.append(base)
        matched_results = cleaned
        #except Exception:
        #    pass
        # Emit a small metadata event to make counts visible to the client logs
        #try:
        meta_evt = {
            'meta': {
                'requested_n': n_results,
                'effective_n': effective_n,
                'returned': len(matched_results)
            }
        }
        yield f"data: {json.dumps(meta_evt)}\n\n"
        #except Exception:
        #    pass
        
    # Filter out embedding property from results before passing to LLM
        for result in matched_results:
            if 'props' in result and isinstance(result['props'], dict):
                result['props'].pop('embedding', None)
            if 'connected_nodes' in result and isinstance(result['connected_nodes'], list):
                for node in result['connected_nodes']:
                    if 'props' in node and isinstance(node['props'], dict):
                        node['props'].pop('embedding', None)
        
        print(matched_results)
        print("Number of fetched nodes:", len(matched_results))
              
        # ------------------------------------------------------------------
        # Adaptive expansion (guarded by allow_expansion): increase effective_n
        # until encoded payload reaches target size; otherwise respect enforced_n.
        # ------------------------------------------------------------------
        def encode_current(results):
            return compact_encode_results(results)

        # Determine whether expansion is allowed. Default: disallow when server
        # enforces n=1 (min=max=1) unless explicitly enabled by allow_expansion.
        allow_expansion = False
        try:
            qs_allow = request.args.get('allow_expansion')
            body_allow = data.get('allow_expansion') if isinstance(data, dict) else None
            if qs_allow is not None:
                allow_expansion = qs_allow.lower() in {'1','true','yes','on'}
            elif body_allow is not None:
                allow_expansion = bool(body_allow)
            else:
                # infer from server caps: if strictly capped to 1, don't expand
                allow_expansion = not (effective_n == 1 and os.getenv('SEMANTIC_SEARCH_MIN_RESULTS','1') == '1' and os.getenv('SEMANTIC_SEARCH_MAX_RESULTS','1') == '1')
        except Exception:
            allow_expansion = False

        if allow_expansion:
            target_chars = None
            try:
                target_chars = int(data.get('min_compact_chars') or request.args.get('min_compact_chars') or 70000)
            except Exception:
                target_chars = 70000
            expand_max = None
            try:
                expand_max = int(data.get('expand_max') or request.args.get('expand_max') or max(effective_n*5, effective_n+10))
            except Exception:
                expand_max = max(effective_n*5, effective_n+10)
            expand_step = None
            try:
                expand_step = int(data.get('expand_step') or request.args.get('expand_step') or max(5, effective_n//2 or 1))
            except Exception:
                expand_step = max(5, effective_n//2 or 1)

            if expand_max < effective_n:
                expand_max = effective_n
            if expand_step <= 0:
                expand_step = 5

            source_ranked = promoted_list

            def build_results(k):
                base = source_ranked[:k]
                fm = fuzzy_match_results(base, query_text)
                fm = serialize_results(fm, instance_limit=instance_limit)
                return fm

            matched_results_current = matched_results
            encoded_results, orig_len, enc_len, dict_maps = encode_current(matched_results_current)
            iteration = 0
            while enc_len < target_chars and len(source_ranked) > len(matched_results_current) and len(matched_results_current) < expand_max:
                new_k = min(len(matched_results_current) + expand_step, expand_max, len(source_ranked))
                iteration += 1
                expanded = build_results(new_k)
                enc_tmp, _, enc_len_tmp, dict_tmp = encode_current(expanded)
                meta_expand = {
                    'meta': {
                        'expansion_iteration': iteration,
                        'candidate_k': new_k,
                        'encoded_chars': enc_len_tmp,
                        'target_chars': target_chars,
                        'met_target': enc_len_tmp >= target_chars
                    }
                }
                try:
                    yield f"data: {json.dumps(meta_expand)}\n\n"
                except Exception:
                    pass
                if enc_len_tmp > enc_len:
                    matched_results_current = expanded
                    encoded_results, orig_len, enc_len, dict_maps = enc_tmp, orig_len, enc_len_tmp, dict_tmp
                if enc_len >= target_chars:
                    break

            matched_results = matched_results_current
            reduction_pct = ((orig_len - enc_len) / orig_len * 100) if orig_len > 0 else 0
        else:
            # No expansion; encode current matched_results only
            encoded_results, orig_len, enc_len, dict_maps = encode_current(matched_results)
            reduction_pct = ((orig_len - enc_len) / orig_len * 100) if orig_len > 0 else 0
        
        # Emit diagnostic for encoding
        try:
            meta_enc = {
                'meta': {
                    'compact_encoding': True,
                    'original_chars': orig_len,
                    'encoded_chars': enc_len,
                    'reduction_percent': round(reduction_pct, 1),
                    'dict_sizes': {k: len(v) for k, v in dict_maps.items()},
                    'target_chars': target_chars,
                    'target_met': enc_len >= target_chars,
                    'final_k': len(matched_results)
                }
            }
            yield f"data: {json.dumps(meta_enc)}\n\n"
        except Exception:
            pass
        
        # Decoding instructions for Gemini
        decoding_rules = """
COMPACT ENCODING FORMAT (Tokenized) - Decoding Guide:
- Top-level object: {"dicts": {"K": [...prop keys...], "T": [...labels...], "R": [...rel types...], "D": [...directions...]}, "data": [items...]}
- Item fields:
  * L: list of label indices -> lookup in dicts.T
  * P: map of property-key-index (string) -> value (lookup key index in dicts.K)
  * C: list of connection objects each with:
    - L: label indices of neighbor
    - P: neighbor properties (key index -> value)
    - r: relationship type index (lookup in dicts.R)
    - d: direction index (lookup in dicts.D)
- All indices are integers; values already trimmed.
- Reconstruct by substituting indices with their dictionary values.
"""
        
        is_counting_query = False #ai_result.get("status") == "counting"
        if graph_format == "aas":
            ai_input = (
                f"{decoding_rules}\n\n"
                f"Please answer the user query based on the compact-encoded results. IMPORTANT: Results may contain relevant entities at MULTIPLE levels:\n"
                f"- Top-level items in the data array\n"
                f"- Connected nodes (C) nested within any item\n"
                f"You MUST examine ALL levels to find complete answers. Check descriptions in both top-level properties (P) and in connected_nodes properties.\n\n"
                f"User query: {query_text}\n\nCompact Results:\n{encoded_results}\n"
            )
            if is_counting_query:
                summary, truncated_results = preprocess_results_for_gemini(matched_results, query_text)
                ai_input = (
                    f"Please answer the user query accurately based on this data. "
                    f"CRITICAL: For counting or availability queries, you MUST use the counts from the Summary section below. "
                    f"The Summary contains ACCURATE recursive counts of all machine instances at all nesting levels. "
                    f"DO NOT manually count items in the Sample Results - those are truncated for display only.\n\n"
                    f"User query: {query_text}\n\n"
                    f"Summary (USE THIS FOR COUNTS):\n{json.dumps(summary, indent=2)}\n\n"
                    f"Sample Results (for reference only, lists may be truncated):\n{truncated_results}\n"
                )
        elif graph_format == "csv":
            ai_input = f"{decoding_rules}\n\nPlese answer the user query based on the compact-encoded results or correct the results based on user query (each dict in Results list is an item with its attributes).\n\nUser query: {query_text}\n\nCompact Results:\n{encoded_results}\n"
        else:
            ai_input = (
                f"{decoding_rules}\n\n"
                f"Please answer the user query based on these compact-encoded knowledge graph entities and relationships.\n\n"
                f"User query: {query_text}\n\n"
                f"Compact Results (entities and their connections):\n{encoded_results}\n"
                f"Please be specific and direct in your answer based on the graph data."
            )

        try:
            print(ai_input)
            print(len(matched_results))
            # Check if user specified a provider override in the request
            provider_override = data.get('ai_provider') or request.args.get('ai_provider')
            response = generate_ai_response(ai_input, stream=True, provider=provider_override)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # Handle streaming response from either provider and both OpenAI versions
        for chunk in response:
            # Gemini format
            if hasattr(chunk, 'text') and chunk.text:
                yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
            # OpenAI new format (>=1.0.0)
            elif hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield f"data: {json.dumps({'chunk': delta.content})}\n\n"
            # OpenAI old format (<1.0.0) - dict-based response
            elif isinstance(chunk, dict) and 'choices' in chunk:
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content')
                if content:
                    yield f"data: {json.dumps({'chunk': content})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Content-Type': 'text/event-stream'
        }
    )
@app.route('/set-graph-format', methods=['POST'])
@cross_origin()
def set_graph_format():
    try:
        data = request.json
        selected_format = data.get('graphFormat', 'rdf')  # Default to 'rdf' if not provided
        
        # Validate the selected format
        if selected_format not in ['csv', 'aas', 'rdf']:
            return jsonify({"status": "error", "message": "Invalid graph format"}), 400
        
        # Update the global graph format
        global graph_format
        graph_format = selected_format

        print(f"Graph format set to: {graph_format}")
        return jsonify({"status": "success", "message": f"Graph format updated to {selected_format}"}), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/create-graph', methods=['POST'])
def create_graph():
    try:
        # Get the graph format from the request
        form_format = request.form.get('graphFormat')
        global graph_format
        
        # If a format was passed in the form, use it (allows override)
        selected_format = form_format if form_format else graph_format
        
        print(f"Using graph format: {selected_format}")
        #graph_format = request.form.get('graphFormat') or request.json.get('graphFormat', 'rdf')
        
        # For AAS JSON format
        if graph_format == 'aas':
            if 'file' in request.files:
                file = request.files['file']
                # Save the uploaded JSON file temporarily
                file_path = os.path.join(tempfile.gettempdir(), "temp_aas.json")
                file.save(file_path)
                
                # Start processing in a separate thread
                def process_aas_data():
                    try:
                        # Initialize the Neo4j importer
                        importer = Neo4jBatchImporter(
                            uri, 
                            username, 
                            password
                        )
                        
                        # Clear existing data
                        with importer.get_session() as session:
                            session.run("MATCH (n) DETACH DELETE n")
                        
                        # Process the AAS JSON file
                        total_assets = 0
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                total_assets = len(data.get("assetAdministrationShells", []))
                        except Exception as e:
                            progress_queue.put({'error': f"Error reading JSON file: {str(e)}"})
                            return
                        
                        # Setup data iterator and create graph
                        data_iterator = process_json_in_chunks(file_path)
                        
                        assets_processed = 0
                        batch = []
                        batch_size = 50  # Process in smaller batches
                        
                        for row in data_iterator:
                            if not row:
                                continue
                                
                            batch.append(row)
                            if len(batch) >= batch_size:
                                _process_batch(importer, batch)
                                assets_processed += len(batch)
                                progress_queue.put({
                                    'processed': assets_processed,
                                    'total': total_assets
                                })
                                batch = []
                        
                        # Process remaining records
                        if batch:
                            _process_batch(importer, batch)
                            assets_processed += len(batch)
                        

                        embedding_generator = EnhancedKGEmbeddings(uri=uri, user=username, password=password, model_name=model)
                        
                        try:
                            embedding_generator.generate_and_store_embeddings(batch_size=50)
                            logger.info("Embedding generation completed successfully!")
                        except Exception as e:
                            logger.error(f"Fatal error: {str(e)}")
                            raise
                        finally:
                            embedding_generator.close()
                        # Signal completion
                        progress_queue.put({
                            'processed': assets_processed,
                            'total': total_assets,
                            'status': 'completed'
                        })
                        
                        # Cleanup
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                    except Exception as e:
                        progress_queue.put({'error': str(e)})
                        print(f"Error in process_aas_data: {e}")
                    finally:
                        if 'importer' in locals():
                            importer.close()
                
                # Start processing thread
                threading.Thread(target=process_aas_data).start()
                #threading.Thread(target=run_aas_embedding_generation, args=(embedding_queue,)).start()
                return jsonify({"message": "Processing started"}), 200
            else:
                return jsonify({"error": "No file provided for AAS format"}), 400
        
        # For CSV format (existing functionality)
        elif graph_format == 'csv':
            # Check if the request contains a file
            if 'file' in request.files:
                file = request.files['file']
                df = pd.read_csv(file)
            elif 'csv_text' in request.json:
                csv_text = request.json['csv_text']
                from io import StringIO
                df = pd.read_csv(StringIO(csv_text))
            else:
                return jsonify({"error": "No file or text provided"}), 400

            total_products = len(df)
            
            # Start processing in a separate thread (existing CSV processing code)
            def process_data():
                try:
                    # Preprocess the data
                    df['currency'] = df['price'].str.extract(r'([^0-9]+)')
                    df['price_value'] = pd.to_numeric(df['price'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0.0)
                    df['stock_type'] = df['number_available_in_stock'].str.extract(r'([^0-9]+)').fillna('Out of stock')
                    df['stock_availability'] = pd.to_numeric(df['number_available_in_stock'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce').fillna(0)
                    df['average_review_rating'] = pd.to_numeric(df['average_review_rating'].str.replace(' out of 5 stars', ''), errors='coerce').fillna(0.0)
                    df['number_of_reviews'] = pd.to_numeric(df['number_of_reviews'].str.replace(',', ''), errors='coerce').fillna(0).astype(int)
                    
                    df['amazon_category_and_sub_category'] = df['amazon_category_and_sub_category'].fillna('').astype(str)
                    df['manufacturer'] = df['manufacturer'].fillna('Unknown')

                    def complete_product_description(row):
                        # Convert all values to strings explicitly
                        product_name = str(row['product_name']) if pd.notna(row['product_name']) else "Unknown"
                        product_desc = str(row['product_description']) if pd.notna(row['product_description']) else ""
                        product_info = str(row['product_information']) if pd.notna(row['product_information']) else ""
                        
                        description = "Product Title: " + product_name + "\n"
                        description += "Product Description: " + product_desc + "\n"
                        description += "Product Information: " + product_info + "\n"
                        return description

                    df['description_complete'] = df.apply(complete_product_description, axis=1)

                    print("Deleting existing data...")
                    graph.run("MATCH (n) DETACH DELETE n")

                    products_processed = 0
                    
                    # Create the knowledge graph
                    for _, row in df.iterrows():
                        try:
                            # Convert all values to appropriate types before creating the Node
                            product = Node("Product",
                                        uniq_id=str(row['uniq_id']),
                                        name=str(row['product_name']) if pd.notna(row['product_name']) else "Unknown",
                                        description=str(row['product_description']) if pd.notna(row['product_description']) else "",
                                        price=float(row['price_value']),
                                        currency=str(row['currency']) if pd.notna(row['currency']) else "",
                                        review_rating=float(row['average_review_rating']),
                                        review_count=int(row['number_of_reviews']),
                                        stock_type=str(row['stock_type']))
                            
                            if 'description_complete' in row and pd.notna(row['description_complete']):
                                product['description_complete'] = str(row['description_complete'])
                            
                            manufacturer = Node("Manufacturer", 
                                             name=str(row['manufacturer']) if pd.notna(row['manufacturer']) else "Unknown")
                            
                            categories = str(row['amazon_category_and_sub_category']).split(' > ')
                            previous_category = None
                            
                            for cat in categories:
                                if cat and not pd.isna(cat):  # Check if category is not empty or NaN
                                    category = Node("Category", name=str(cat))
                                    graph.merge(category, "Category", "name")
                                    if previous_category:
                                        graph.merge(Relationship(previous_category, "HAS_SUBCATEGORY", category))
                                    previous_category = category
                            
                            graph.merge(product, "Product", "uniq_id")
                            graph.merge(manufacturer, "Manufacturer", "name")
                            graph.merge(Relationship(product, "MANUFACTURED_BY", manufacturer))
                            if previous_category:
                                graph.merge(Relationship(product, "BELONGS_TO", previous_category))
                            
                            products_processed += 1
                            progress_queue.put({
                                'processed': products_processed,
                                'total': total_products
                            })
                            
                        except Exception as e:
                            print(f"Error processing product {row['uniq_id']}: {e}")
                            continue

                    # Signal completion
                    progress_queue.put({
                        'processed': total_products,
                        'total': total_products,
                        'status': 'completed'
                    })
                    
                except Exception as e:
                    progress_queue.put({'error': str(e)})
                    print(f"Error in process_data: {e}")

            # Start processing thread
            threading.Thread(target=process_data).start()
            
            return jsonify({"message": "Processing started"}), 200
        
        elif graph_format == 'rdf':
            if 'file' in request.files:
                file = request.files['file']
                # Save the uploaded RDF file temporarily
                file_path = os.path.join(tempfile.gettempdir(), "temp_rdf.jsonl")
                file.save(file_path)
                
                # Start processing in a separate thread
                def process_rdf_data():
                    try:
                        # Initialize the Neo4j importer
                        importer = Neo4jBatchImporter(
                            uri, 
                            username, 
                            password
                        )
                        
                        # Clear existing data
                        # with importer.get_session() as session:
                        #     session.run("MATCH (n) DETACH DELETE n")
                        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
                            with driver.session() as session:
                                session.run("MATCH (n) DETACH DELETE n")
                        # Process the RDF file
                        process_rdf_file(file_path, uri, (username, password), batch_size=1000)
                        
                        generate_and_store_embeddings(model, URI=uri, AUTH=(username, password), batch_size=100)
                        # Signal completion
                        progress_queue.put({
                            'status': 'completed'
                        })
                        logger.info("Embedding generation completed successfully!")
                        # Cleanup
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                    except Exception as e:
                        progress_queue.put({'error': str(e)})
                        print(f"Error in process_rdf_data: {e}")
                    finally:
                        if 'importer' in locals():
                            importer.close()
                
                # Start processing thread
                threading.Thread(target=process_rdf_data).start()
                # thread = Thread(target=run_rdf_embedding_generation, args=(embedding_queue,))
                # thread.daemon = True
                # thread.start()
                #generate_and_store_embeddings(model, uri, auth=(username, password), batch_size=100)
                return jsonify({"message": "Processing started"}), 200
            else:
                return jsonify({"error": "No file provided for RDF format"}), 400
        
        #generate_embeddings()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/delete-graph", methods=["POST"])
def delete_graph():
    """Delete the current knowledge graph."""
    try:
        graph.run("MATCH (n) DETACH DELETE n")
        return jsonify({"message": "Knowledge graph deleted successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete knowledge graph: {str(e)}"}), 500

def get_embedding(text):
    """Get the embedding for a single text using sentence-transformers."""
    try:
        embedding = model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def generate_enriched_cypher(query, cypher_query):
    """Send the user query and Cypher query to Gemini to generate an enriched Cypher query."""
    gemini_input = f"Please update the WHERE clause of this Cypher query based on the user query for a more accurate search and !!write just the entire resulting cypher query!!.\n\nUser query: {query}\nCypher Query: {cypher_query}\n"
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(gemini_input)
        enriched_cypher_query = response.text.strip()
        enriched_cypher_query = enriched_cypher_query.replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        print(f"Error generating enriched Cypher query with Gemini: {e}")
        enriched_cypher_query = cypher_query  # Fallback to original Cypher query
    return enriched_cypher_query


def fix_cypher_query(query, cypher_query):
    """
    Fixes common Cypher syntax issues before sending to Gemini for enrichment.
    """
    # Log the initial problematic query
    logging.info(f"Attempting to fix Cypher query:\n{cypher_query}")

    # 🚫 Fix #1: Invalid WHERE inside exists()
    cypher_query = re.sub(
        r"exists\(\(([^)]+)\)\s*WHERE\s*([^)]+)\)",
        r"exists { MATCH (\1) WHERE \2 }",
        cypher_query
    )

    # 🚫 Fix #2: Double WHERE clauses in sequence
    cypher_query = re.sub(r"\bWHERE\b.*?\bWHERE\b", "WHERE", cypher_query, flags=re.DOTALL)

    # 🚫 Fix #3: Remove trailing commas in RETURN clauses
    cypher_query = re.sub(r"RETURN\s+([^,]+),\s*\)", r"RETURN \1)", cypher_query)

    # 🚫 Fix #4: Remove empty OPTIONAL MATCH clauses
    cypher_query = re.sub(r"OPTIONAL MATCH\s*\(\)", "", cypher_query)

    # 🚫 Fix #5: Remove redundant parentheses
    cypher_query = re.sub(r"\(\s*\(", "(", cypher_query)
    cypher_query = re.sub(r"\)\s*\)", ")", cypher_query)

    logging.info(f"Cleaned Cypher query before Gemini:\n{cypher_query}")

    # ✅ Call Gemini for further enrichment
    gemini_input = f"Please fix and improve this Cypher query based on the user query.\nUser query: {query}\nCypher Query: {cypher_query}\n"
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        response = model.generate_content(gemini_input)
        enriched_cypher_query = response.text.strip()
        enriched_cypher_query = enriched_cypher_query.replace("```cypher", "").replace("```", "").strip()
        logging.info(f"Gemini fixed query:\n{enriched_cypher_query}")
    except Exception as e:
        logging.error(f"Error generating enriched Cypher query with Gemini: {e}")
        enriched_cypher_query = cypher_query  # Fallback

    return enriched_cypher_query
    




def run_cypher_query(session, cypher_query, query_text, params):
    """Universal Cypher execution & generic record shaping.

    Features:
    - Executes provided Cypher with params; retries with light auto-fix if it fails.
    - Returns list[dict] with primitive JSON-serializable values.
    - Flattens Neo4j records: converts Nodes -> dict(label, properties), Relationships -> dict(type, start_id, end_id, properties), Paths -> list of segment dicts.
    - Preserves any existing score / similarity fields if present; adds position-based fallback score otherwise.
    - Works for any schema (AAS, CSV, RDF, arbitrary) without graph_format branching.
    """
    if not cypher_query or not isinstance(cypher_query, str):
        logging.error("Empty or invalid Cypher query provided to run_cypher_query.")
        return []

    def _to_plain(value):
        from neo4j.graph import Node, Relationship, Path
        if isinstance(value, Node):
            # Exclude 'embedding' property; include internal id for neighbor enrichment
            return {
                "_type": "node",
                "id": value.id,
                "labels": list(value.labels),
                **{k: value.get(k) for k in value.keys() if k != 'embedding'}
            }
        if isinstance(value, Relationship):
            return {
                "_type": "relationship",
                "rel_type": value.type,
                "start_id": value.start_node.id,
                "end_id": value.end_node.id,
                **{k: value.get(k) for k in value.keys() if k != 'embedding'}
            }
        if isinstance(value, Path):
            segments = []
            for rel in value.relationships:
                segments.append(_to_plain(rel))
            return {"_type": "path", "segments": segments}
        if isinstance(value, list):
            return [_to_plain(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_plain(v) for k, v in value.items() if k != 'embedding'}
        return value

    try:
        logging.info(f"Running Cypher query:\n{cypher_query}")
        result_cursor = session.run(cypher_query, **params)
    except Exception as e:
        logging.error(f"Primary Cypher execution failed: {e}")
        fixed = fix_cypher_query(query_text, cypher_query) if cypher_query else None
        if fixed and fixed != cypher_query:
            try:
                logging.info("Retrying with auto-fixed Cypher")
                result_cursor = session.run(fixed, **params)
            except Exception as ee:
                logging.error(f"Retry with fixed query failed: {ee}")
                return []
        else:
            return []

    records = []
    node_ids = set()
    for idx, record in enumerate(result_cursor):
        plain = {}
        for key in record.keys():
            val_plain = _to_plain(record.get(key))
            plain[key] = val_plain
            # Collect node ids for neighbor enrichment
            if isinstance(val_plain, dict) and val_plain.get('_type') == 'node' and 'id' in val_plain:
                node_ids.add(val_plain['id'])
            elif isinstance(val_plain, list):
                for item in val_plain:
                    if isinstance(item, dict) and item.get('_type') == 'node' and 'id' in item:
                        node_ids.add(item['id'])
        # Provide fallback score if none present
        if 'score' not in plain and 'similarity' in plain and isinstance(plain['similarity'], (int, float)):
            plain['score'] = plain['similarity']
        if 'score' not in plain:
            plain['score'] = max(0.0, 1.0 - (idx * 0.01))  # mild decay
        records.append(plain)

    # Enrich each node with its first-degree connected nodes' properties (excluding embeddings)
    neighbors_map = {}
    if node_ids:
        try:
            neighbor_query = """
            MATCH (n) WHERE elementId(n) IN $ids
            OPTIONAL MATCH (n)--(m)
            WITH n, collect(DISTINCT m)[0..5] AS ms
            RETURN elementId(n) AS id, ms
            """
            neighbor_results = session.run(neighbor_query, ids=list(node_ids))
            for nr in neighbor_results:
                raw_list = nr.get('ms') or []
                cleaned = []
                for m in raw_list:
                    if m is None:
                        continue
                    try:
                        props = {k: v for k, v in dict(m).items() if k != 'embedding'}
                        cleaned.append({
                            'id': m.id,
                            'labels': list(m.labels),
                            **props
                        })
                    except Exception:
                        continue
                neighbors_map[nr.get('id')] = cleaned
        except Exception as ne:
            logging.warning(f"Neighbor enrichment failed: {ne}")

    if neighbors_map:
        for rec in records:
            for key, val in list(rec.items()):
                if isinstance(val, dict) and val.get('_type') == 'node' and 'id' in val:
                    val['connected_nodes'] = neighbors_map.get(val['id'], [])
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and item.get('_type') == 'node' and 'id' in item:
                            item['connected_nodes'] = neighbors_map.get(item['id'], [])
  

    logging.info(f"Fetched {len(records)} results (universal)")
    return records

def fuzzy_match_results(results, query_text):
    """Universal fuzzy matching for heterogeneous Neo4j query outputs.

    Strategy:
      1. Identify candidate text fields automatically (string values, excluding ids & embeddings).
      2. Compute fuzzy partial ratios against the user query for each candidate.
      3. Derive a composite similarity (max + small boost if multiple semi-matches).
      4. If nested structures (connected_nodes, neighbors, paths) exist, include their textual fields.
      5. Preserve existing numeric score/similarity if present; keep both (original under 'orig_score').

    Returns list sorted descending by similarity.
    """
    if not query_text or not isinstance(query_text, str) or not results:
        return results

    ql = query_text.lower()

    def iter_texts(obj):
        """Yield textual snippets from arbitrary nested result structures."""
        if obj is None:
            return
        if isinstance(obj, str):
            txt = obj.strip()
            if txt and len(txt) < 2000:  # ignore huge blobs
                yield txt
        elif isinstance(obj, dict):
            for k, v in obj.items():
                lk = k.lower()
                # Skip obvious non-text / embedding / ids
                if lk in {"embedding", "id", "_id"} or lk.endswith("_embedding") or lk.startswith("embedding"):
                    continue
                if isinstance(v, (str, list, dict)):
                    for sub in iter_texts(v):
                        yield sub
        elif isinstance(obj, list):
            for item in obj:
                for sub in iter_texts(item):
                    yield sub

    enriched = []
    for r in results:
        try:
            texts = list(iter_texts(r))
            seen = set()
            dedup_texts = []
            for t in texts:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    dedup_texts.append(t)

            similarities = []
            for t in dedup_texts:
                try:
                    sim = fuzz.partial_ratio(ql, t.lower())
                    similarities.append(sim)
                except Exception:
                    continue

            if similarities:
                max_sim = max(similarities)
                # Boost: count of medium matches (>=50) adds small fraction
                medium = sum(1 for s in similarities if 50 <= s < max_sim)
                composite = max_sim + min(10, medium * 2)
            else:
                composite = 0

            # Preserve original score if present
            if 'score' in r and 'orig_score' not in r:
                r['orig_score'] = r['score']
            r['similarity'] = composite

            enriched.append(r)
        except Exception as e:
            logging.warning(f"fuzzy_match_results skipped a record due to error: {e}")
            continue

    # Sort by similarity (fallback to existing score if similarity ties)
    return sorted(
        enriched,
        key=lambda x: (x.get('similarity', 0), x.get('score', 0)),
        reverse=True
    )


def relax_cypher_query(cypher_query: str, relaxation_level: int, query_text: str) -> str:
    """Schema-agnostic Cypher relaxation with improved syntax handling."""
    import re, math
    if not cypher_query or not isinstance(cypher_query, str):
        return cypher_query

    original = cypher_query
    working = cypher_query

    # Utility: modify LIMIT - fix duplicate LIMIT issues
    def bump_limit(q: str, factor: float = 2.0, floor: int = 20, cap: int = 200) -> str:
        def repl(m):
            num = int(m.group(1))
            new_num = min(cap, max(floor, math.ceil(num * factor)))
            return f"LIMIT {new_num}"
        
        # First replace existing LIMIT
        if re.search(r"LIMIT\s+(\d+)", q, flags=re.IGNORECASE):
            return re.sub(r"LIMIT\s+(\d+)", repl, q, flags=re.IGNORECASE)
        elif re.search(r"LIMIT\s+\$[a-zA-Z_][a-zA-Z0-9_]*", q, flags=re.IGNORECASE):
            # Replace parameter-based LIMIT  
            return re.sub(r"LIMIT\s+\$[a-zA-Z_][a-zA-Z0-9_]*", f"LIMIT {floor}", q, flags=re.IGNORECASE)
        else:
            # If no LIMIT, append one
            return q.rstrip() + f"\nLIMIT {floor}"

    # Utility: lower similarity threshold
    def soften_similarity(q: str, min_floor: float = 0.05) -> str:
        pattern = r"WHERE\s+similarity\s*>\s*(\d+\.\d+|\d+)"
        def repl(m):
            val = float(m.group(1))
            new_val = max(min_floor, round(val * 0.6, 3))
            return f"WHERE similarity > {new_val}"
        return re.sub(pattern, repl, q, flags=re.IGNORECASE)

    # Utility: remove similarity threshold
    def drop_similarity(q: str) -> str:
        return re.sub(r"WHERE\s+similarity\s*>\s*(\d+\.\d+|\d+)", "", q, flags=re.IGNORECASE)

    # Utility: soften strict equality filters of form n.prop = 'literal'
    def soften_equals(q: str) -> str:
        def repl(m):
            left = m.group(1)
            lit = m.group(2)
            return f"toLower({left}) CONTAINS toLower('{lit}')"
        return re.sub(r"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\s*=\s*'([^']+)'", repl, q)

    # Utility: remove label spec e.g. (n:Label) -> (n)
    def drop_labels(q: str) -> str:
        return re.sub(r"\(([a-zA-Z_][A-Za-z0-9_]*):[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)*\)", r"(\1)", q)

    # Utility: prune simple property predicates
    def prune_property_filters(q: str) -> str:
        q = re.sub(r"AND\s+[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*=\s*'[^']*'", "", q)
        q = re.sub(r"AND\s+[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s+IN\s+\[[^\]]*\]", "", q)
        q = re.sub(r"WHERE\s+AND", "WHERE", q)
        q = re.sub(r"\s{2,}", " ", q)
        return q

    # Store evolution for logging if needed
    if relaxation_level <= 0:
        logging.info("Relaxation level 0: returning original query")
        return original

    if relaxation_level >= 1:
        working = soften_similarity(working)
        working = bump_limit(working, factor=1.5, floor=20)

    if relaxation_level >= 2:
        working = drop_similarity(working)
        working = soften_equals(working)

    if relaxation_level >= 3:
        working = drop_labels(working)
        working = prune_property_filters(working)
        working = bump_limit(working, factor=2.0, floor=40, cap=300)

    if relaxation_level >= 4:
        # Simplified neighbor expansion - just boost the limit more aggressively
        working = bump_limit(working, factor=1.5, floor=60, cap=400)

    if relaxation_level >= 5:
        # Final aggressive broadening: remove ORDER BY similarity if present
        working = re.sub(r"ORDER BY\s+similarity\s+DESC", "", working, flags=re.IGNORECASE)

    logging.info(f"Relaxed query (level {relaxation_level}):\n{working}")
    return working



###############################################
def run_semantic_search_for_rdf(session, query_text, query_embedding, n_results=10):
    """
    Run semantic search on RDF data to find relevant entities and their connections.
    Modified to avoid returning copies of the same plant and to capture a diverse set of nodes.
    
    Args:
        session: Neo4j session
        query_text: Original user query
        query_embedding: Vector embedding of the query
        n_results: Number of results to return
        
    Returns:
        List of results with diverse entities and their connections
    """
    # Initial query to find all relevant entities based on embedding similarity
    base_cypher_query = """
    MATCH (e:Entity)
    WHERE e.embedding IS NOT NULL
    WITH e, 
    REDUCE(dot = 0.0, i IN RANGE(0, SIZE(e.embedding)-1) | 
        dot + e.embedding[i] * $embedding[i]
    ) / (
        SQRT(REDUCE(norm1 = 0.0, i IN RANGE(0, SIZE(e.embedding)-1) | 
            norm1 + e.embedding[i] * e.embedding[i]
        )) * 
        SQRT(REDUCE(norm2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
            norm2 + $embedding[i] * $embedding[i]
        ))
    ) AS similarity
    WHERE similarity > 0.3
    
    // Return diverse entities, not just the highest similarity matches
    RETURN e, similarity
    ORDER BY similarity DESC
    LIMIT $n
    """
    
    # Execute the initial query to get diverse entities
    results = session.run(base_cypher_query, embedding=query_embedding, n=n_results*2)
    entities = [(record["e"], record["similarity"]) for record in results]
    
    # Ensure diversity by getting entities with different labels/types
    entity_types = {}
    diverse_entities = []
    
    for entity, similarity in entities:
        # Extract entity type from name or label
        entity_type = None
        if "name" in entity and isinstance(entity["name"], str):
            # Try to extract a type from the name (e.g., "plant1" -> "plant")
            match = re.match(r"([a-zA-Z]+)\d+", entity["name"])
            if match:
                entity_type = match.group(1)
        
        # If we couldn't get a type from the name, use the first label
        if not entity_type and entity.labels:
            entity_type = next(iter(entity.labels))
        
        # Use a default if we still don't have a type
        if not entity_type:
            entity_type = "unknown"
        
        # Limit how many entities of each type we include
        if entity_type not in entity_types:
            entity_types[entity_type] = 0
        
        if entity_types[entity_type] < 3:  # Max 3 entities of each type
            diverse_entities.append((entity, similarity))
            entity_types[entity_type] += 1
            
            # Stop if we have enough diverse entities
            if len(diverse_entities) >= n_results:
                break
    
    results_list = []
    
    # Process each diverse entity to find its connections
    for entity, similarity in diverse_entities:
        # Get the entity's connections
        connections_query = """
        MATCH (e:Entity {name: $entity_name})
        
        // Find outgoing relationships
        OPTIONAL MATCH (e)-[r1]->(e2:Entity)
        
        // Find incoming relationships
        OPTIONAL MATCH (e3:Entity)-[r2]->(e)
        
        // Return all connections in a unified format
        RETURN 
            e.name AS entity1,
            collect(DISTINCT {
                direction: 'outgoing',
                relationship: type(r1),
                target: e2.name
            }) AS outgoing,
            collect(DISTINCT {
                direction: 'incoming',
                relationship: type(r2),
                source: e3.name
            }) AS incoming,
            $similarity AS score
        """
        
        result = session.run(connections_query, entity_name=entity["name"], similarity=similarity)
        
        for record in result:
            entity_result = {
                "entity1": record["entity1"],
                "score": record["score"],
                "connections": []
            }
            
            # Add outgoing connections
            for conn in record["outgoing"]:
                if conn["target"]:  # Skip null values
                    entity_result["connections"].append({
                        "from": record["entity1"],
                        "relationship": conn["relationship"],
                        "to": conn["target"]
                    })
            
            # Add incoming connections
            for conn in record["incoming"]:
                if conn["source"]:  # Skip null values
                    entity_result["connections"].append({
                        "from": conn["source"],
                        "relationship": conn["relationship"],
                        "to": record["entity1"]
                    })
            
            results_list.append(entity_result)
    
    return results_list

def fix_semantic_search_for_rdf(data):
    """
    Main function to fix the semantic search for RDF data.
    Replaces the parts of the semantic search that handle RDF data.
    
    Args:
        data: The request data containing the query
        
    Returns:
        A streaming response with the search results
    """
    query_text = data.get('query', '')
    n_results = data.get('n', 10)
    print(data)
    query_text = query_text["output"]
    # Get the query embedding
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return jsonify({"error": "Failed to get query embedding"}), 400
    
    # Run the improved semantic search
    with driver.session() as session:
        results_list = run_semantic_search_for_rdf(session, query_text, query_embedding, n_results)
        
        # Enhanced preprocessing for relationship detection
        # This avoids hardcoding relationships like "PRODUCES" or "CONSUMES"
        relationship_counts = {}
        for result in results_list:
            for connection in result.get("connections", []):
                rel = connection.get("relationship", "").upper()
                if rel:
                    relationship_counts[rel] = relationship_counts.get(rel, 0) + 1
        
        # Find the most common relationships in the results
        common_relationships = sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare the results for Gemini
        gemini_input = (
            f"Please answer the user query based on the knowledge graph results. "
            f"Each result contains information about entities and their relationships. "
            f"The results show diverse entities and their connections.\n\n"
            f"User query: {query_text}\n\n"
            f"Knowledge Graph Results:\n{json.dumps(results_list, indent=2)}\n\n"
        )
        
        # Add information about the most common relationships if available
        if common_relationships:
            gemini_input += (
                f"Most common relationships in the results:\n"
                f"{json.dumps(dict(common_relationships[:3]), indent=2)}\n\n"
            )
        
        gemini_input += (
            f"Please provide a clear, concise answer focusing specifically on the relationships "
            f"between entities and the direction of those relationships (which entity is the source "
            f"and which is the target of each relationship)."
        )
        
        print(gemini_input)
        # Call Gemini to generate a response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(gemini_input, stream=True)
        print(response)
        # Stream the response
        def generate():
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Content-Type': 'text/event-stream'
            }
        )

def preprocess_results_for_gemini(matched_results, query_text):
    """
    Preprocess results to make them more manageable for Gemini, 
    especially for counting-type operations
    """
    if graph_format == "aas":
        
        # Determine the equipment type from the top-level results to filter instances
        # Only count instances that match the queried equipment type
        equipment_types = set()
        for item in matched_results:
            labels = item.get("labels", [])
            # Check for equipment type labels (Drilling, CircleCutting, Sawing, etc.)
            for label in labels:
                if label in ["Drilling", "CircleCutting", "Sawing", "Milling", "Grinding", "Welding"]:
                    equipment_types.add(label)
        
        # Helper function to recursively count nodes with availability info at all nesting levels
        def count_nodes_recursive(item, visited=None, parent_equipment_type=None):
            """Recursively traverse all connected_nodes to count availability.
            Only count instances that match the parent equipment type."""
            if visited is None:
                visited = set()
            
            # Use node ID to avoid double-counting
            node_id = item.get("props", {}).get("id")
            if node_id and node_id in visited:
                return {"total": 0, "available": 0, "unavailable": 0}
            if node_id:
                visited.add(node_id)
            
            counts = {"total": 0, "available": 0, "unavailable": 0}
            
            # Determine the equipment type for this item
            current_equipment_type = parent_equipment_type
            item_labels = item.get("labels", [])
            for label in item_labels:
                if label in equipment_types:
                    current_equipment_type = label
                    break
            
            # Count current item if it has availability
            availability = item.get("props", {}).get("availability")
            asset_kind = item.get("props", {}).get("assetKind")
            
            # Only count Instance nodes (not Type nodes) that match the queried equipment type
            if asset_kind == "Instance":
                # Check if this instance should be counted based on equipment type context
                should_count = False
                
                if not equipment_types:
                    # If no equipment type filter, count all instances
                    should_count = True
                else:
                    # Check relationship type to determine if this instance belongs to the equipment type
                    rel_type = item.get("rel_type", "")
                    for eq_type in equipment_types:
                        # Match relationship like HAS_DRILLING, HAS_MODEL from Drilling parent
                        if f"HAS_{eq_type.upper()}" in rel_type or current_equipment_type == eq_type:
                            should_count = True
                            break
                
                if should_count:
                    counts["total"] += 1
                    if availability == "True":
                        counts["available"] += 1
                    elif availability == "False":
                        counts["unavailable"] += 1
            
            # Recursively process all connected_nodes, passing the equipment type context
            for node in item.get("connected_nodes", []):
                sub_counts = count_nodes_recursive(node, visited, current_equipment_type)
                counts["total"] += sub_counts["total"]
                counts["available"] += sub_counts["available"]
                counts["unavailable"] += sub_counts["unavailable"]
            
            return counts
        
        # Count all nodes recursively across all results
        all_counts = {"total": 0, "available": 0, "unavailable": 0}
        for item in matched_results:
            # Detect equipment type from this top-level item
            item_equipment_type = None
            for label in item.get("labels", []):
                if label in equipment_types:
                    item_equipment_type = label
                    break
            
            # Pass the equipment type to the recursive function
            item_counts = count_nodes_recursive(item, None, item_equipment_type)
            all_counts["total"] += item_counts["total"]
            all_counts["available"] += item_counts["available"]
            all_counts["unavailable"] += item_counts["unavailable"]

        # Create a summary of the results with essential information
        summary = {
            "total_results": len(matched_results),
            "total_instance_nodes": all_counts["total"],
            "available_nodes": all_counts["available"],
            "unavailable_nodes": all_counts["unavailable"]
        }
        
        # NOTE: For counting queries, the LLM should use the summary above which contains
        # accurate recursive counts. The connected_nodes in results are for reference only.
        # Truncation logic removed to ensure data integrity for counting operations.
        
        # Return both the summary and the truncated results
        return summary, matched_results
    elif graph_format == "rdf":
        summary = {
            "total_results": len(matched_results),
            "total_entities": len(matched_results),
            "entity_types": {result.get("entity1"): 1 for result in matched_results},
            "relationships": {result.get("relationship1"): 1 for result in matched_results},
        }
        
        # Limit the number of results to avoid overloading Gemini
        if len(matched_results) > 5:
            matched_results = matched_results[:5]
            matched_results.append({"note": f"Showing 5 of {len(matched_results)} results"})
        
    return summary, matched_results


# ============================================================================
# CYPHER QUERY ENDPOINT (for MCP Server)
# ============================================================================

@app.route('/cypher', methods=['POST'])
@cross_origin()
def execute_cypher():
    """
    Execute arbitrary Cypher queries on the Neo4j database.
    
    Used by MCP server tools like update_availability.
    
    Request body:
        {
            "query": "MATCH (n:Asset) RETURN n LIMIT 10",
            "params": {"param1": "value1", ...}  // optional
        }
    
    Returns:
        JSON array of results, where each result is a dictionary of the returned values.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        cypher_query = data.get('query', '')
        params = data.get('params', {})
        
        if not cypher_query:
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Executing Cypher query: {cypher_query}")
        logger.info(f"With params: {params}")
        
        # Execute the query using Neo4j driver
        with driver.session() as session:
            result = session.run(cypher_query, **params)
            
            # Convert results to list of dictionaries
            records = []
            for record in result:
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    # Convert Neo4j nodes/relationships to dictionaries
                    if hasattr(value, '__dict__'):
                        record_dict[key] = dict(value)
                    else:
                        record_dict[key] = value
                records.append(record_dict)
        
        logger.info(f"Cypher query returned {len(records)} records")
        return jsonify(records), 200
        
    except Exception as e:
        error_msg = f"Cypher query error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    # Run the Flask app
    # use_reloader=False prevents constant restarts when files change
    app.run(debug=True, port=5001, use_reloader=False)


