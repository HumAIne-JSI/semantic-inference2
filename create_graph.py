import json
from neo4j import GraphDatabase
from typing import Iterator, Dict
from contextlib import contextmanager
import logging
from time import sleep
from neo4j.exceptions import ServiceUnavailable
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Neo4jBatchImporter:
    def __init__(self, uri: str, user: str, password: str, batch_size: int = 500):
        self.uri = uri
        self.user = user
        self.password = password
        self.batch_size = batch_size
        self.driver = None
    @contextmanager
    def get_session(self):
        retries = 3
        while retries > 0:
            try:
                if not self.driver:
                    self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                session = self.driver.session()
                yield session
                return
            except ServiceUnavailable:
                retries -= 1
                if retries == 0:
                    raise
                sleep(1)
    def close(self):
        if self.driver:
            self.driver.close()
def process_json_in_chunks(file_path: str, chunk_size: int = 1000) -> Iterator[Dict]:
    """Process large JSON files in chunks."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        shells = data.get("assetAdministrationShells", [])
        submodel_lookup = {submodel["id"]: submodel for submodel in data.get("submodels", [])}
        
        for i in range(0, len(shells), chunk_size):
            chunk = shells[i:i + chunk_size]
            for shell in chunk:
                processed_data = process_shell(shell, submodel_lookup)
                if processed_data:  # Only yield if data is valid
                    yield processed_data
def process_shell(shell: Dict, submodel_lookup: Dict) -> Dict:
    """Process individual shell data."""
    try:
        processed = {
            "id": shell.get("id", ""),
            "idShort": shell.get("idShort", ""),
            "description": shell.get("description", [{}])[0].get("text", ""),
            "assetKind": shell.get("assetInformation", {}).get("assetKind", ""),
            "manufacturer": "",
            "energyConsumption": "",
            "drilling": "",
            "circleCutting": "",
            "sawing": "",
            "availability": "",  # Initialize availability field
            "derivedFrom": shell.get("derivedFrom", {}).get("keys", [{}])[0].get("value", "")  # Add derivedFrom field
        }
        for submodel_ref in shell.get("submodels", []):
            if not submodel_ref.get("keys"):
                continue
            
            submodel_id = submodel_ref["keys"][0]["value"]
            submodel = submodel_lookup.get(submodel_id)
            if not submodel:
                continue
            submodel_name = submodel.get("idShort", "")
            elements = submodel.get("submodelElements", [])
            if not elements:
                continue
            if submodel_name == "Manufacturer":
                processed["manufacturer"] = elements[0].get("value", "")
            elif submodel_name == "Energy Consumption" and len(elements) >= 2:
                processed["energyConsumption"] = f"Voltage: {elements[0].get('value', '')}V, Current: {elements[1].get('value', '')}A"
            elif submodel_name == "Drilling" and len(elements) >= 2:
                processed["drilling"] = f"Diameter: {elements[0].get('value', '')}mm, Depth: {elements[1].get('value', '')}mm"
            elif submodel_name == "Circle Cutting" and len(elements) >= 2:
                processed["circleCutting"] = f"Diameter: {elements[0].get('value', '')}cm, Depth: {elements[1].get('value', '')}cm"
            elif submodel_name == "Sawing" and elements:
                processed["sawing"] = f"Cutting Depth: {elements[0].get('value', '')}mm"
            elif submodel_name == "Availability" and elements:  # Add availability processing
                processed["availability"] = elements[0].get("value", "")
        return processed
    except Exception as e:
        logger.error(f"Error processing shell {shell.get('id', 'unknown')}: {str(e)}")
        return None
    
def _process_batch(importer: Neo4jBatchImporter, batch: list):
    """Process a batch of records with retry logic."""
    with importer.get_session() as session:
        for row in batch:
            try:
                session.run(
                    """
                    MERGE (a:Asset {id: $id})
                    SET a.idShort = $idShort, 
                        a.description = $description, 
                        a.assetKind = $assetKind,
                        a.availability = $availability
                    WITH a
                    MERGE (m:Manufacturer {name: $manufacturer})
                    MERGE (a)-[:MANUFACTURED_BY]->(m)
                    WITH a
                    MERGE (e:EnergyConsumption {details: $energyConsumption})
                    MERGE (a)-[:HAS_ENERGY_CONSUMPTION]->(e)
                    WITH a
                    MERGE (d:Drilling {details: $drilling})
                    MERGE (a)-[:HAS_DRILLING]->(d)
                    WITH a
                    MERGE (c:CircleCutting {details: $circleCutting})
                    MERGE (a)-[:HAS_CIRCLE_CUTTING]->(c)
                    WITH a
                    MERGE (s:Sawing {details: $sawing})
                    MERGE (a)-[:HAS_SAWING]->(s)
                    WITH a
                    MERGE (av:Availability {status: $availability})
                    MERGE (a)-[:HAS_AVAILABILITY]->(av)
                    WITH a
                    // Add has_model relationship if derivedFrom matches an asset id
                    MATCH (model:Asset {id: $derivedFrom})
                    MERGE (a)-[:HAS_MODEL]->(model)
                    """,
                    row
                )
            except Exception as e:
                logger.error(f"Error processing record {row.get('id', 'unknown')}: {str(e)}")
                
def create_neo4j_graph(importer: Neo4jBatchImporter, data_iterator: Iterator[Dict]):
    """Create Neo4j graph with batch processing and error handling."""
    batch = []
    total_processed = 0
    try:
        with importer.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Clear existing data
            
        for row in data_iterator:
            if not row:
                continue
                
            batch.append(row)
            if len(batch) >= importer.batch_size:
                _process_batch(importer, batch)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed} assets")
                batch = []
        # Process remaining records
        if batch:
            _process_batch(importer, batch)
            total_processed += len(batch)
            
        logger.info(f"Import completed. Total assets processed: {total_processed}")
        
    except Exception as e:
        logger.error(f"Error during import: {str(e)}")
        raise
        
def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4jj"
    NEO4J_PASSWORD = "RJCFCs7tsl5fmc9SS0WG-aJyj_PCFrY5GO_lUQSq21c"
    
    try:
        importer = Neo4jBatchImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        data_iterator = process_json_in_chunks("example1.json")
        create_neo4j_graph(importer, data_iterator)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        importer.close()
if __name__ == "__main__":
    main()