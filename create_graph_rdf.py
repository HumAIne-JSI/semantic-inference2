from neo4j import GraphDatabase
import json

# Neo4j connection details
URI = "neo4j://localhost:7687"  # Replace with your Neo4j URI
AUTH = ("neo4jj", "RJCFCs7tsl5fmc9SS0WG-aJyj_PCFrY5GO_lUQSq21c")  # Replace with your Neo4j credentials

# File path
FILE_PATH = "example_aas_kg_20250822_054856.jsonl"#"sys.knowledge-graph.jsonl"

# Batch size
BATCH_SIZE = 1000  # Adjust based on your system's memory and performance

def create_knowledge_graph(tx, batch):
    """Create nodes and relationships in Neo4j."""
    for record in batch:
        entity1 = record["entity1"]
        entity2 = record["entity2"]
        predicate = record["predicate"].replace("-", "_").replace(" ", "_")  # Replace hyphens and spaces with underscores

        # Create or merge nodes for entity1 and entity2
        tx.run(
            """
            MERGE (e1:Entity {name: $entity1})
            MERGE (e2:Entity {name: $entity2})
            MERGE (e1)-[:%s]->(e2)
            """ % predicate,
            entity1=entity1,
            entity2=entity2,
        )

def process_file_in_batches(file_path, URI, AUTH ,batch_size):
    """Process the JSONL file in batches."""
    batch = []
    with open(file_path, "r") as file:
        for line in file:
            # Parse JSON line
            record = json.loads(line.strip())
            batch.append(record)

            # Process batch when it reaches the specified size
            if len(batch) >= batch_size:
                with GraphDatabase.driver(URI, auth=AUTH) as driver:
                    with driver.session() as session:
                        session.execute_write(create_knowledge_graph, batch)
                batch = []  # Reset the batch

        # Process the remaining records in the last batch
        if batch:
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                with driver.session() as session:
                    session.execute_write(create_knowledge_graph, batch)

if __name__ == "__main__":
    print("Starting to process the knowledge graph file...")
    process_file_in_batches(FILE_PATH, URI, AUTH, BATCH_SIZE)
    print("Knowledge graph creation completed.")