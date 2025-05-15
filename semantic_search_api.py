import os
import google.generativeai as genai
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
from sentence_transformers import SentenceTransformer
import tempfile
from create_graph import (
    Neo4jBatchImporter, 
    process_json_in_chunks, 
    _process_batch
)
from gen_embeddings import EnhancedKGEmbeddings
from create_graph_rdf import process_file_in_batches as process_rdf_file
from gen_embeddings_rdf import generate_and_store_embeddings



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Set up environment variables for Google API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"  # Replace with your actual API key

# Configure the genai API with the Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Neo4j connection setup (replace with your connection details)
uri = "neo4j://localhost:7687"
username = "neo4j" # Replace with your Neo4j username
password = "YOUR_PASSWORD" # Replace with your Neo4j password
driver = GraphDatabase.driver(uri, auth=(username, password))
graph = Graph(uri, auth=(username, password))

# Initialize Flask app
app = Flask(__name__)

# Global queue to store progress updates
progress_queue = Queue()


model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_queue = Queue()


global grpah_format
graph_format = "aas"

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

# def run_csv_embedding_generation(queue):
#     """Background task to generate embeddings for CSV data."""
#     def update_queue(total, processed):
#         queue.put({"total": total, "processed": processed})
    
#     batch_size = 100
#     total_query = """
#     MATCH (a:Asset)
#     WHERE a.description_embedding IS NULL
#       AND a.description IS NOT NULL
#     RETURN count(a) AS total
#     """
#     total_result = graph.run(total_query).data()
#     total_to_process = total_result[0]['total'] if total_result else 0
#     total_processed = 0
#     update_queue(total_to_process, total_processed)
    
#     while True:
#         # Fetch batch of assets
#         query = """
#         MATCH (a:Asset)
#         WHERE a.description_embedding IS NULL
#           AND a.description IS NOT NULL
#         RETURN a.id AS id, a.description AS description
#         LIMIT $batch_size
#         """
#         assets = graph.run(query, parameters={'batch_size': batch_size}).data()
#         if not assets:
#             break
        
#         # Process batch embeddings
#         descriptions = [a['description'] for a in assets if a['description']]
#         ids = [a['id'] for a in assets if a['description']]
       
#         try:
#             # Generate embeddings for entire batch at once
#             embeddings = model.encode(descriptions, convert_to_numpy=True)
           
#             # Update database in batch
#             data = [{'id': id, 'embedding': embedding.tolist()}
#                    for id, embedding in zip(ids, embeddings)]
           
#             # Create a new transaction and execute updates
#             tx = graph.begin()
#             try:
#                 for item in tqdm(data, desc="Batches"):
#                     tx.run("""
#                     MATCH (a:Asset {id: $id})
#                     SET a.description_embedding = $embedding
#                     """, parameters=item)
#                 tx.commit()
#             except Exception as e:
#                 tx.rollback()
#                 print(f"Transaction failed: {e}")
#                 continue
           
#             total_processed += len(assets)
#             update_queue(total_to_process, total_processed)
           
#         except Exception as e:
#             print(f"Error processing batch: {e}")
       
#         # Small delay to prevent overwhelming the database
#         time.sleep(0.1)
#         print(str(total_processed)+"/"+str(total_to_process))
        
#     # Clear the queue when done
#     while not queue.empty():
#         queue.get()

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
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(gemini_input)
        enriched_cypher_query = response.text.strip()
        enriched_cypher_query = enriched_cypher_query.replace("```cypher", "").replace("```", "").strip()
        logging.info(f"Gemini fixed query:\n{enriched_cypher_query}")
    except Exception as e:
        logging.error(f"Error generating enriched Cypher query with Gemini: {e}")
        enriched_cypher_query = cypher_query  # Fallback

    return enriched_cypher_query
    




def run_cypher_query(session, cypher_query, query, params):
    """Run a Cypher query and return the results as a list."""
    
    try:
        logging.info(f"Running Cypher query: {cypher_query}")
        results = session.run(cypher_query, **params)
    except Exception as e:
        logging.error(f"Error running Cypher query: {e}")
        
        # Attempt to fix the query if it fails
        logging.info("Attempting to fix the Cypher query...")
        fixed_cypher_query = fix_cypher_query(query, cypher_query)
        logging.info(f"Fixed Cypher query: {fixed_cypher_query}")
        
        try:
            results = session.run(fixed_cypher_query, **params)
        except Exception as retry_error:
            logging.error(f"Failed after retry: {retry_error}")
            return []

    results_list = []
    
    
    for result in results:
        if graph_format == "aas":
            asset = {
                "id": result.get("id"),
                "name": result.get("name", "No name"),
                "description": result.get("description", "No description"),
                "manufacturer": result.get("manufacturer", "Unknown manufacturer"),
                "energy_consumption": result.get("energy_consumption", "No energy consumption data"),
                "drilling": result.get("drilling", "No drilling data"),
                "circle_cutting": result.get("circle_cutting", "No circle cutting data"),
                "sawing": result.get("sawing", "No sawing data"),
                "availability": result.get("availability", "Unknown availability"),
                "score": result.get("score", 0.0),
                "connected_models": result.get("connected_models", [])  
            }
            results_list.append(asset)
        
        elif graph_format == "csv":
            product = {
                "uniq_id": result.get("uniq_id"),
                "name": result.get("name", "No name"),
                "description": result.get("description", "No description"),
                "description_complete": result.get("description_complete", "No detailed description"),
                "price": result.get("price", "N/A"),
                "currency": result.get("currency", "N/A"),
                "review_count": result.get("review_count", 0),
                "review_rating": result.get("review_rating", 0),
                "stock_type": result.get("stock_type", "N/A"),
                "score": result.get("score", 0.0)
            }
            results_list.append(product)
        
        # elif graph_format == "rdf":

        #     # Enhanced RDF entity structure with comprehensive relationship mapping
        #     entity = {
        #         "entity1": result.get("entity1", "No entity1"),
        #         "entity2": result.get("entity2", ""),
        #         "entity3": result.get("entity3", ""),
        #         "relationship1": result.get("relationship1", ""),
        #         "relationship2": result.get("relationship2", ""),
        #         "score": result.get("score", 0.0),
        #         "connections": []  # To store related entities
        #     }
            
        #     # Add direct connections
        #     if result.get("entity2") and result.get("relationship1"):
        #         entity["connections"].append({
        #             "from": result.get("entity1"),
        #             "relationship": result.get("relationship1"),
        #             "to": result.get("entity2")
        #         })
                
        #     if result.get("entity2") and result.get("entity3") and result.get("relationship2"):
        #         entity["connections"].append({
        #             "from": result.get("entity2"),
        #             "relationship": result.get("relationship2"),
        #             "to": result.get("entity3")
        #         })
            
        #     # Add additional connections if present
        #     if result.get("additional_connections"):
        #         for conn in result.get("additional_connections"):
        #             if isinstance(conn, dict) and conn.get("node") and conn.get("relationship"):
        #                 entity["connections"].append({
        #                     "from": result.get("entity1"),
        #                     "relationship": conn.get("relationship"),
        #                     "to": conn.get("node")
        #                 })
            
        #     # Add incoming connections if present
        #     if result.get("incoming_connections"):
        #         for conn in result.get("incoming_connections"):
        #             if isinstance(conn, dict) and conn.get("source") and conn.get("relationship"):
        #                 entity["connections"].append({
        #                     "from": conn.get("source"),
        #                     "relationship": conn.get("relationship"),
        #                     "to": result.get("entity1")
        #                 })
            
        #     # Add context connections if present (for maximum relaxation queries)
        #     if result.get("context_connections"):
        #         for conn in result.get("context_connections"):
        #             if isinstance(conn, dict) and conn.get("source") and conn.get("relationship") and conn.get("target"):
        #                 entity["connections"].append({
        #                     "from": conn.get("source"),
        #                     "relationship": conn.get("relationship"),
        #                     "to": conn.get("target")
        #                 })
            
        #     results_list.append(entity)
            
    logging.info(f"Fetched {len(results_list)} results")
    return results_list

def fuzzy_match_results(results, query_text):
    """Fuzzy match the results to the query text based on various fields."""
    enriched_results = []
    
    for result in results:
        # Calculate similarity scores for different fields
        if graph_format == "aas":
            name_similarity = fuzz.partial_ratio(query_text.lower(), result.get("name", "").lower())
            desc_similarity = fuzz.partial_ratio(query_text.lower(), result.get("description", "").lower())
            manufacturer_similarity = fuzz.partial_ratio(query_text.lower(), result.get("manufacturer", "").lower())
            
            # Calculate similarities for technical specifications
            tech_specs = [
                result.get("energy_consumption", ""),
                result.get("drilling", ""),
                result.get("circle_cutting", ""),
                result.get("sawing", ""),
                result.get("available", "")
            ]
            
            tech_similarities = [
                fuzz.partial_ratio(query_text.lower(), spec.lower())
                for spec in tech_specs if spec
            ]
            
            # Get the maximum similarity across all fields
            all_similarities = [name_similarity, desc_similarity, manufacturer_similarity] + tech_similarities
            max_similarity = max(all_similarities) if all_similarities else 0
            
            # Add similarity score to the result
            result["similarity"] = max_similarity
            enriched_results.append(result)
        
        elif graph_format == "csv":
            name_similarity = fuzz.partial_ratio(query_text.lower(), result.get("name", "").lower())
            description_similarity = fuzz.partial_ratio(query_text.lower(), result.get("description", "").lower())
            max_similarity = max(name_similarity, description_similarity)

            # Retain all fields and add similarity
            result["similarity"] = max_similarity
            enriched_results.append(result)
        
        # elif graph_format == "rdf":
        #     # Enhanced similarity calculation for RDF entities
        #     entity1_similarity = fuzz.partial_ratio(query_text.lower(), result.get("entity1", "").lower())
        #     entity2_similarity = fuzz.partial_ratio(query_text.lower(), result.get("entity2", "").lower()) if result.get("entity2") else 0
        #     entity3_similarity = fuzz.partial_ratio(query_text.lower(), result.get("entity3", "").lower()) if result.get("entity3") else 0
            
        #     # Include relationship similarity for better matching
        #     rel1_similarity = fuzz.partial_ratio(query_text.lower(), result.get("relationship1", "").lower()) if result.get("relationship1") else 0
        #     rel2_similarity = fuzz.partial_ratio(query_text.lower(), result.get("relationship2", "").lower()) if result.get("relationship2") else 0
            
        #     # Add connection similarity checking
        #     connection_similarities = []
        #     for conn in result.get("connections", []):
        #         if isinstance(conn, dict):
        #             conn_from = fuzz.partial_ratio(query_text.lower(), conn.get("from", "").lower())
        #             conn_to = fuzz.partial_ratio(query_text.lower(), conn.get("to", "").lower())
        #             conn_rel = fuzz.partial_ratio(query_text.lower(), conn.get("relationship", "").lower())
        #             connection_similarities.append(max(conn_from, conn_to, conn_rel))
            
        #     # Get maximum similarity across all fields
        #     all_similarities = [entity1_similarity, entity2_similarity, entity3_similarity, rel1_similarity, rel2_similarity]
        #     if connection_similarities:
        #         all_similarities.append(max(connection_similarities))
            
        #     max_similarity = max(all_similarities)
            
        #     # Boost scores for production/consumption queries
        #     if any(word in query_text.lower() for word in ["produce", "produces", "producing", "production"]):
        #         if "PRODUCES" in str(result):
        #             max_similarity += 15
            
        #     if any(word in query_text.lower() for word in ["consume", "consumes", "consuming", "consumption"]):
        #         if "CONSUMES" in str(result):
        #             max_similarity += 15
            
        #     # Add similarity score to the result
        #     result["similarity"] = max_similarity
        #     enriched_results.append(result)

    # Sort by similarity score
    return sorted(enriched_results, key=lambda x: x["similarity"], reverse=True)

def relax_cypher_query(cypher_query, relaxation_level, query_text):
    """Relax the Cypher query progressively to increase recall while maintaining relevance."""
    if graph_format == "aas":
        if relaxation_level == 0:
            # Base query - no relaxation
            return cypher_query

        elif relaxation_level == 1:
            # 1️⃣ Lower similarity slightly + Increase result count
            relaxed_query = re.sub(r"WHERE similarity > \d+\.\d+", "WHERE similarity > 0.3", cypher_query)
            relaxed_query = re.sub(r"LIMIT \$(n|limit)", "LIMIT 20", relaxed_query)

        elif relaxation_level == 2:
            # 2️⃣ Remove strict similarity filter & add keyword-based filtering
            relaxed_query = re.sub(r"WHERE similarity > \d+\.\d+", "", cypher_query)
            relaxed_query = relaxed_query.replace("ORDER BY similarity DESC", "")

        elif relaxation_level == 3:
            # 3️⃣ Remove unnecessary relationships & return more related assets
            relaxed_query = """
            MATCH (a:Asset)
            OPTIONAL MATCH (a)-[:MANUFACTURED_BY]->(m:Manufacturer)
            OPTIONAL MATCH (a)-[:HAS_ENERGY_CONSUMPTION]->(e:EnergyConsumption)
            OPTIONAL MATCH (a)-[:HAS_DRILLING]->(d:Drilling)
            OPTIONAL MATCH (a)-[:HAS_CIRCLE_CUTTING]->(c:CircleCutting)
            OPTIONAL MATCH (a)-[:HAS_SAWING]->(s:Sawing)
            OPTIONAL MATCH (a)-[:HAS_AVAILABILITY]->(v:Availability)
            RETURN 
                a.id AS id,
                a.idShort AS name,
                a.description AS description,
                m.name AS manufacturer,
                e.details AS energy_consumption,
                d.details AS drilling,
                c.details AS circle_cutting,
                s.details AS sawing,
                v.details AS available,
                0.15 AS score, [] AS connected_subassets
            ORDER BY score DESC
            LIMIT 20
            """

        elif relaxation_level == 4:
            # 4️⃣ Include indirectly related assets
            relaxed_query = """
            MATCH (a:Asset)
            OPTIONAL MATCH (a)-[:MANUFACTURED_BY]->(m:Manufacturer)
            OPTIONAL MATCH (a)-[:HAS_ENERGY_CONSUMPTION]->(e:EnergyConsumption)
            OPTIONAL MATCH (a)-[:HAS_DRILLING]->(d:Drilling)
            OPTIONAL MATCH (a)-[:HAS_CIRCLE_CUTTING]->(c:CircleCutting)
            OPTIONAL MATCH (a)-[:HAS_SAWING]->(s:Sawing)
            OPTIONAL MATCH (a)-[:HAS_AVAILABILITY]->(v:Availability)
            OPTIONAL MATCH (a)-[:RELATED_TO]->(r:Asset)
            RETURN 
                a.id AS id,
                a.idShort AS name,
                a.description AS description,
                m.name AS manufacturer,
                e.details AS energy_consumption,
                d.details AS drilling,
                c.details AS circle_cutting,
                s.details AS sawing,
                v.details AS available,
                0.1 AS score, collect(r.idShort) AS connected_subassets
            ORDER BY score DESC
            LIMIT 30
            """

        else:
            # 5️⃣ Maximum relaxation - get everything without filters
            relaxed_query = """
            MATCH (a:Asset)
            RETURN a.id AS id, a.idShort AS name, a.description AS description,
                "" AS manufacturer, "" AS energy_consumption,
                "" AS drilling, "" AS circle_cutting, "" AS sawing,
                0.05 AS score, [] AS connected_subassets
            ORDER BY score DESC
            LIMIT 50
            """
    
    elif graph_format == "csv":
        if relaxation_level == 1:
        # Remove exact name matching, broaden to partial description matching
            relaxed_query = re.sub(
                r"AND toLower\(p\.name\) CONTAINS toLower\('.*?'\)",
                "",
                cypher_query
            )
        elif relaxation_level == 2:
            # Remove all name-based filters; match only on embeddings
            relaxed_query = re.sub(
                r"WHERE similarity > 0.*",
                "WHERE similarity > 0",
                cypher_query,
                flags=re.DOTALL
            )
        elif relaxation_level == 3:
            # Lower the similarity threshold
            relaxed_query = cypher_query.replace("WHERE similarity > 0", "WHERE similarity >= 0.5")
        else:
            # If no relaxation is needed or nothing to relax, return as is
            relaxed_query = cypher_query
    
    # elif graph_format == "rdf":
    #     # Improved RDF format relaxation
    #     if relaxation_level == 1:
    #         # Lower similarity threshold
    #         relaxed_query = re.sub(r"WHERE similarity > 0.4", "WHERE similarity > 0.2", cypher_query)
    #     elif relaxation_level == 2:
    #         # Remove similarity filter entirely
    #         relaxed_query = re.sub(r"WHERE similarity > \d+\.\d+", "", cypher_query)
    #     elif relaxation_level == 3:
    #         # Expand relationship search to include more context
    #         relaxed_query = """
    #         MATCH (e1:Entity)
    #         WHERE toLower(e1.name) CONTAINS toLower($query_text) OR $query_text CONTAINS toLower(e1.name)
            
    #         // Get connected entities and their relations
    #         MATCH paths = (e1)-[r1]->(e2:Entity)
    #         OPTIONAL MATCH extended = (e2)-[r2]->(e3:Entity)
            
    #         // Add reverse direction paths for completeness
    #         OPTIONAL MATCH reverse_paths = (e4:Entity)-[r3]->(e1)
            
    #         RETURN 
    #             e1.name AS entity1,
    #             type(r1) AS relationship1,
    #             e2.name AS entity2,
    #             type(r2) AS relationship2,
    #             e3.name AS entity3,
    #             0.3 AS score,
    #             collect({source: e4.name, relationship: type(r3), direction: 'incoming'}) AS incoming_connections
    #         LIMIT 20
    #         """
    #     else:
    #         # Maximum relaxation - context-aware query focusing on specific relationships
    #         keywords = query_text.lower().split()
    #         production_related = any(word in keywords for word in ['produce', 'produces', 'production', 'manufacturing', 'makes', 'made', 'create'])
    #         consumption_related = any(word in keywords for word in ['consume', 'consumes', 'consumption', 'uses', 'used', 'utilizing', 'requires'])
            
    #         relationship_focus = "PRODUCES" if production_related else "CONSUMES" if consumption_related else ""
            
    #         relaxed_query = f"""
    #         // First find entities mentioned in the query
    #         MATCH (e:Entity)
    #         WHERE toLower(e.name) CONTAINS toLower($query_text) OR any(word IN split(toLower($query_text), ' ') WHERE toLower(e.name) CONTAINS word)
            
    #         // Then find relationships that match the query intent
    #         WITH e
    #         OPTIONAL MATCH (e)-[r1]->(target:Entity)
    #         {f"WHERE type(r1) CONTAINS '{relationship_focus}'" if relationship_focus else ""}
            
    #         // Also check reverse relationships
    #         OPTIONAL MATCH (source:Entity)-[r2]->(e)
    #         {f"WHERE type(r2) CONTAINS '{relationship_focus}'" if relationship_focus else ""}
            
    #         // Return comprehensive connection information
    #         RETURN 
    #             e.name AS entity1,
    #             type(r1) AS relationship1,
    #             target.name AS entity2,
    #             "" AS relationship2,
    #             "" AS entity3,
    #             0.1 AS score,
    #             collect(DISTINCT {{source: source.name, relationship: type(r2), target: e.name}}) AS context_connections
    #         ORDER BY score DESC
    #         LIMIT 25
    #         """

    logging.info(f"Relaxed query (level {relaxation_level}): {relaxed_query}")
    return relaxed_query



from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json



@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    """API endpoint for performing semantic search with streaming response."""
    try:
        data = request.json
        query_text = data.get('query', '')
        n_results = data.get('n', 10)

        query_text = query_text["output"]
        print(query_text)
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        #graph_format = request.form.get('graphFormat') or request.json.get('graphFormat', 'rdf')
        print(graph_format)

        if graph_format == "rdf":
            return fix_semantic_search_for_rdf(data)

        def generate():
            try:
                # Step 1: Get the query embedding
                query_embedding = get_embedding(query_text)
                if not query_embedding:
                    yield "data: " + json.dumps({"error": "Failed to get query embedding"}) + "\n\n"
                    return

                # Step 2: Base Cypher query for asset search
                if graph_format == "aas":
                    base_cypher_query = """
                    MATCH (a:Asset)
                    WHERE a.embedding IS NOT NULL
                    WITH a, 
                    REDUCE(dot = 0.0, i IN RANGE(0, SIZE(a.embedding)-1) | 
                        dot + a.embedding[i] * $embedding[i]
                    ) / (
                        SQRT(REDUCE(norm1 = 0.0, i IN RANGE(0, SIZE(a.embedding)-1) | 
                            norm1 + a.embedding[i] * a.embedding[i]
                        )) * 
                        SQRT(REDUCE(norm2 = 0.0, i IN RANGE(0, SIZE($embedding)-1) | 
                            norm2 + $embedding[i] * $embedding[i]
                        ))
                    ) AS similarity
                    WHERE similarity > 0.6

                    OPTIONAL MATCH (a)-[:MANUFACTURED_BY]->(m:Manufacturer)
                    OPTIONAL MATCH (a)-[:HAS_ENERGY_CONSUMPTION]->(e:EnergyConsumption)
                    OPTIONAL MATCH (a)-[:HAS_DRILLING]->(d:Drilling)
                    OPTIONAL MATCH (a)-[:HAS_CIRCLE_CUTTING]->(c:CircleCutting)
                    OPTIONAL MATCH (a)-[:HAS_SAWING]->(s:Sawing)
                    OPTIONAL MATCH (a)-[:HAS_AVAILABILITY]->(v:Availability)

                    OPTIONAL MATCH (instance)-[:HAS_MODEL]->(a)
                    OPTIONAL MATCH (instance)-[:HAS_ENERGY_CONSUMPTION]->(inst_e:EnergyConsumption)
                    OPTIONAL MATCH (instance)-[:HAS_DRILLING]->(inst_d:Drilling)
                    OPTIONAL MATCH (instance)-[:HAS_CIRCLE_CUTTING]->(inst_c:CircleCutting)
                    OPTIONAL MATCH (instance)-[:HAS_SAWING]->(inst_s:Sawing)
                    OPTIONAL MATCH (instance)-[:HAS_AVAILABILITY]->(inst_v:Availability)

                    RETURN 
                        a.id AS id,
                        a.idShort AS name,
                        a.description AS description,
                        m.name AS manufacturer,
                        e.details AS energy_consumption,
                        d.details AS drilling,
                        c.details AS circle_cutting,
                        s.details AS sawing,
                        v.status AS availability,
                        similarity AS score,
                        COLLECT(DISTINCT {
                            id: instance.id, 
                            name: instance.idShort, 
                            energy_consumption: inst_e.details,
                            drilling: inst_d.details,
                            circle_cutting: inst_c.details,
                            sawing: inst_s.details,
                            availability: inst_v.status
                        }) AS connected_models
                    ORDER BY similarity DESC
                    LIMIT $n
                    """
                elif graph_format == "csv":
                    base_cypher_query = """
                    MATCH (p:Product)
                    WHERE p.description_embedding IS NOT NULL
                    WITH p,
                        reduce(dot = 0.0, i in range(0, size(p.description_embedding)-1) |
                        dot + p.description_embedding[i] * $embedding[i]) / 
                        (sqrt(reduce(a = 0.0, i in range(0, size(p.description_embedding)-1) |
                        a + p.description_embedding[i] * p.description_embedding[i])) *
                        sqrt(reduce(b = 0.0, i in range(0, size($embedding)-1) |
                        b + $embedding[i] * $embedding[i])))
                        AS similarity
                    WHERE similarity > 0
                    RETURN 
                        p.uniq_id AS uniq_id,
                        p.name AS name,
                        p.description AS description,
                        p.description_complete AS description_complete,
                        p.price AS price,
                        p.currency AS currency,
                        p.review_count AS review_count,
                        p.review_rating AS review_rating,
                        p.stock_type AS stock_type,
                        similarity AS score
                    ORDER BY similarity DESC
                    LIMIT $n
                    """
                
                # Step 3: Generate enriched Cypher query using Gemini
                enriched_cypher_query = generate_enriched_cypher(query_text, base_cypher_query)

                # Step 4: Run the query and handle results
                results_list = []
                with driver.session() as session:
                    results_list = run_cypher_query(session, enriched_cypher_query, query_text,
                                                  {"embedding": query_embedding, "n": n_results})
                    logging.info(f"Fetched {len(results_list)} results")
                    if graph_format == "aas":# or graph_format == "rdf":
                        # If no results, try relaxing the query
                        if len(results_list) < 5:
                            
                            relaxation_level = 0
                            max_iterations = 5
                            results_list = []

                            while relaxation_level < max_iterations:
                                logging.info(f"Relaxing search: level {relaxation_level}...")
                                relaxed_cypher_query = relax_cypher_query(base_cypher_query, relaxation_level, query_text)
                                
                                results_list = run_cypher_query(session, relaxed_cypher_query, query_text,
                                                                {"embedding": query_embedding, "n": 10})
                                
                                if len(results_list) > 5:
                                    break
                                relaxation_level += 1

                            if relaxation_level >= max_iterations:
                                logging.warning("Reached maximum iterations while trying to relax the query.")

                    elif graph_format == "csv":
                        if not results_list:
                            relaxation_level = 0
                            max_iterations = 5
                            while relaxation_level < max_iterations:
                                logging.info(f"Attempt {relaxation_level + 1}: Running relaxed query...")
                                relaxed_cypher_query = relax_cypher_query(base_cypher_query, relaxation_level, query_text)
                                results_list = run_cypher_query(session, relaxed_cypher_query, {"embedding": query_embedding, "n": n_results, "query_text": query_text})
                                logging.info(f"Fetched {len(results_list)} results")
                                if results_list:
                                    break
                                relaxation_level += 1

                            if relaxation_level >= max_iterations:
                                logging.warning("Reached maximum iterations while trying to relax the query.")
                    

                # Step 5: Fuzzy match results
                matched_results = fuzzy_match_results(results_list, query_text)
                
                # Debug logging
                print(query_text)
                try:
                    print("relaxed: ", relaxed_cypher_query)
                except:
                    print("enriched: ", enriched_cypher_query)

                # Step 6: Stream the enriched response
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Determine if this is a counting-related query
                if graph_format == "aas":
                    is_counting_query = any(word in query_text.lower() for word in 
                                        ["count", "how many", "number of", "total"])#, "available", "unavailable"])
                
                if graph_format == "aas":
                    # For non-counting queries, use the direct approach from the second version
                    gemini_input = (
                        f"Please answer the user query based on the list of results (each dict in Results list "
                        f"is an item with its attributes, note that connected_models are machines themselves).\n\n"
                        f"User query: {query_text}\nResults:\n{matched_results}\n"
                    )
                    
                    if is_counting_query:
                        # Use the preprocess_results_for_gemini function for counting queries
                        summary, truncated_results = preprocess_results_for_gemini(matched_results, query_text)
                        
                        # Prepare a more structured input for Gemini
                        gemini_input = (
                            f"Please answer the user query accurately based on this data. "
                            f"Pay special attention to counts and availability information in the summary.\n\n"
                            f"User query: {query_text}\n\n"
                            f"Summary:\n{json.dumps(summary, indent=2)}\n\n"
                            f"Sample Results (some lists may be truncated):\n{truncated_results}\n"
                        )
                elif graph_format == "csv":
                    gemini_input = f"Plese answer the user query based on the results or correct the results based on user query (each dict in Results list is an item with its attributes).\n\nUser query: {query_text}\nResults:\n{matched_results}\n"
                
                else:
                    gemini_input = (
                        f"Please answer the user query based on these knowledge graph entities and relationships.\n\n"
                        f"User query: {query_text}\n\n"
                        f"Results (entities and their connections):\n{matched_results}\n"
                        f"Please be specific and direct in your answer based on the graph data."
                    )
                
                print(matched_results)
                print(gemini_input)
                
                response = model.generate_content(gemini_input, stream=True)
                print(response)
                for chunk in response:
                    if chunk.text:
                        yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache, no-store, must-revalidate',  # Disable caching
                'Pragma': 'no-cache',
                'Content-Type': 'text/event-stream'
            }
        )

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



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

        # Create a summary of the results with essential information
        summary = {
            "total_results": len(matched_results),
            "total_models": sum(len(item.get("connected_models", [])) for item in matched_results),
            "available_models": sum(
                sum(1 for model in item.get("connected_models", []) 
                    if model.get("availability") == "True")
                for item in matched_results
            ),
            "unavailable_models": sum(
                sum(1 for model in item.get("connected_models", []) 
                    if model.get("availability") == "False")
                for item in matched_results
            ),
            "model_types": {item.get("name"): len(item.get("connected_models", [])) for item in matched_results},
            "available_by_type": {
                item.get("name"): sum(1 for model in item.get("connected_models", []) 
                                    if model.get("availability") == "True")
                for item in matched_results
            },
            "unavailable_by_type": {
                item.get("name"): sum(1 for model in item.get("connected_models", []) 
                                    if model.get("availability") == "False")
                for item in matched_results
            }
        }
        
        # Limit the number of models in the results to avoid overloading Gemini
        for item in matched_results:
            # Count models before truncating
            total_models = len(item.get("connected_models", []))
            available_models = sum(1 for model in item.get("connected_models", []) if model.get("availability") == "True")
            unavailable_models = total_models - available_models
            
            # Keep only a sample of models (first 5) to avoid overwhelming Gemini
            if "connected_models" in item and len(item["connected_models"]) > 5:
                item["connected_models"] = item["connected_models"][:5]
                # Add a note about truncation
                item["connected_models_note"] = f"Showing 5 of {total_models} models ({available_models} available, {unavailable_models} unavailable)"
        
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

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5001)
