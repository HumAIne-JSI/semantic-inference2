from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class EnhancedKGEmbeddings:
    def __init__(self, uri: str, user: str, password: str, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with Neo4j connection and embedding model."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = model_name#SentenceTransformer(model_name)
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
    def prepare_node_text(self, node_data: Dict[str, Any]) -> str:
        """Prepare text representation of a node including its properties and relationships."""
        texts = []
        
        # Handle Asset nodes
        if node_data['label'] == 'Asset':
            if node_data.get('idShort'):
                texts.append(f"ID: {node_data['idShort']}")
            if node_data.get('description'):
                texts.append(f"Description: {node_data['description']}")
            if node_data.get('assetKind'):
                texts.append(f"Asset Kind: {node_data['assetKind']}")
            
            # Add outgoing relationships
            for conn in node_data.get('connections', []):
                if conn['target']:
                    if conn['type'] == 'HAS_AVAILABILITY':
                        texts.append(f"Availability Status: {conn['target']}")
                    else:
                        texts.append(f"{conn['type']}: {conn['target']}")
        
        # Handle property nodes
        elif node_data['label'] == 'Availability':
            texts.append("Type: Availability")
            if node_data.get('status'):
                texts.append(f"Status: {node_data['status']}")
            # Add incoming relationships for availability
            for conn in node_data.get('incomingConnections', []):
                if conn['source']:
                    texts.append(f"Asset: {conn['source']}")
        
        # Handle other property nodes
        else:
            node_type = node_data['label']
            texts.append(f"Type: {node_type}")
            
            if node_data.get('name'):
                texts.append(f"Name: {node_data['name']}")
            if node_data.get('details'):
                texts.append(f"Details: {node_data['details']}")
            
            # Add incoming relationships
            for conn in node_data.get('incomingConnections', []):
                if conn['source']:
                    texts.append(f"Connected to Asset: {conn['source']} via {conn['type']}")
        return " ".join(filter(None, texts))
    def get_all_node_paths(self) -> List[Dict[str, Any]]:
        """Get all nodes and their relationships with connected nodes."""
        with self.driver.session() as session:
            # First, get all Asset nodes
            assets_query = """
            MATCH (n:Asset)
            OPTIONAL MATCH (n)-[r]->(m)
            WITH n, 
                collect(DISTINCT {
                    type: type(r),
                    target: CASE 
                        WHEN m:Availability THEN m.status
                        WHEN m.details IS NOT NULL THEN m.details
                        WHEN m.name IS NOT NULL THEN m.name
                        ELSE ''
                    END
                }) as connections
            RETURN 
                'asset_' + n.id as node_id,
                labels(n)[0] as label,
                n.idShort as idShort,
                n.description as description,
                n.assetKind as assetKind,
                connections
            """
            
            # Get availability nodes separately
            availability_query = """
            MATCH (n:Availability)
            OPTIONAL MATCH (a:Asset)-[r]->(n)
            WITH n, 
                collect(DISTINCT {
                    type: type(r),
                    source: a.idShort
                }) as incomingConnections
            RETURN 
                'availability_' + coalesce(n.status, id(n)) as node_id,
                'Availability' as label,
                n.status as status,
                incomingConnections
            """
            
            # Get other property nodes
            properties_query = """
            MATCH (n) 
            WHERE n:CircleCutting OR n:Drilling OR n:EnergyConsumption OR n:Sawing OR n:Manufacturer
            OPTIONAL MATCH (a:Asset)-[r]->(n)
            WITH n, 
                labels(n)[0] as label,
                collect(DISTINCT {
                    type: type(r),
                    source: a.idShort
                }) as incomingConnections
            RETURN 
                CASE 
                    WHEN n.name IS NOT NULL THEN 'prop_name_' + n.name 
                    WHEN n.details IS NOT NULL THEN 'prop_details_' + n.details
                    ELSE 'prop_' + id(n)
                END as node_id,
                label,
                n.name as name,
                n.details as details,
                incomingConnections
            """
            
            assets = list(session.run(assets_query))
            availability_nodes = list(session.run(availability_query))
            properties = list(session.run(properties_query))
            
            return [dict(record) for record in assets + availability_nodes + properties]
    def generate_and_store_embeddings(self, batch_size: int = 50):
        """Generate and store embeddings for all nodes with their full context."""
        try:
            # Get all nodes with their relationships
            nodes = self.get_all_node_paths()
            logger.info(f"Found {len(nodes)} nodes to process")
            # Process in batches
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # Prepare text for each node in the batch
                texts = []
                valid_nodes = []
                
                for node in batch:
                    text = self.prepare_node_text(node)
                    if text.strip():  # Only process nodes with actual text content
                        texts.append(text)
                        valid_nodes.append(node)
                
                if not texts:
                    continue
                
                # Generate embeddings
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
                # Store embeddings
                with self.driver.session() as session:
                    for node, embedding in zip(valid_nodes, embeddings):
                        if node['label'] == 'Asset':
                            query = """
                            MATCH (n:Asset {id: $id})
                            SET n.embedding = $embedding
                            """
                            session.run(query, id=node['node_id'].replace('asset_', ''), 
                                    embedding=embedding.tolist())
                        elif node['label'] == 'Availability':
                            query = """
                            MATCH (n:Availability {status: $status})
                            SET n.embedding = $embedding
                            """
                            session.run(query, status=node.get('status', ''),
                                    embedding=embedding.tolist())
                        else:
                            # For other property nodes, match on details or name
                            query = """
                            MATCH (n:{label})
                            WHERE n.details = $details OR n.name = $name
                            SET n.embedding = $embedding
                            """.format(label=node['label'])
                            
                            session.run(query, 
                                    details=node.get('details', ''),
                                    name=node.get('name', ''),
                                    embedding=embedding.tolist())
                
                logger.info(f"Processed batch {i // batch_size + 1} of {(len(nodes) - 1) // batch_size + 1}")
                logger.info(f"Current node types: {', '.join(set(n['label'] for n in valid_nodes))}")
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise
def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4jj"
    NEO4J_PASSWORD = "RJCFCs7tsl5fmc9SS0WG-aJyj_PCFrY5GO_lUQSq21c"
    
    embedding_generator = EnhancedKGEmbeddings(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        embedding_generator.generate_and_store_embeddings(batch_size=50)
        logger.info("Embedding generation completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        embedding_generator.close()
if __name__ == "__main__":
    main()