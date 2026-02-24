from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Neo4j connection details
URI = "neo4j://localhost:7687"  # Replace with your Neo4j URI
AUTH = ("neo4jj", "RJCFCs7tsl5fmc9SS0WG-aJyj_PCFrY5GO_lUQSq21c")  # Replace with your Neo4j credentials

# Transformer model for generating embeddings
MODEL_NAME = "all-MiniLM-L6-v2"  # You can use any model from sentence-transformers
BATCH_SIZE = 100  # Number of nodes to process in each batch

# Initialize the transformer model
model = SentenceTransformer(MODEL_NAME)

def fetch_nodes(tx, batch_size, skip):
    """Fetch a batch of nodes from Neo4j."""
    query = """
    MATCH (n:Entity)
    RETURN n.name AS name
    SKIP $skip
    LIMIT $batch_size
    """
    result = tx.run(query, batch_size=batch_size, skip=skip)
    return [record["name"] for record in result]

def update_node_embedding(tx, name, embedding):
    """Update a node with its embedding."""
    query = """
    MATCH (n:Entity {name: $name})
    SET n.embedding = $embedding
    """
    tx.run(query, name=name, embedding=embedding.tolist())

def generate_and_store_embeddings(model, URI, AUTH, batch_size):
    """Generate embeddings for nodes and store them in Neo4j."""
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            skip = 0
            while True:
                # Fetch a batch of nodes
                nodes = session.execute_read(fetch_nodes, batch_size, skip)
                if not nodes:
                    break  # No more nodes to process

                # Generate embeddings for the batch
                print(f"Generating embeddings for batch starting at skip={skip}...")
                embeddings = model.encode(nodes, show_progress_bar=True)

                # Update nodes with their embeddings
                print(f"Updating embeddings for batch starting at skip={skip}...")
                for name, embedding in zip(nodes, embeddings):
                    session.execute_write(update_node_embedding, name, embedding)

                # Move to the next batch
                skip += batch_size

if __name__ == "__main__":
    print("Starting to generate and store embeddings...")
    generate_and_store_embeddings(model, URI, AUTH, BATCH_SIZE)
    print("Embedding generation and storage completed.")