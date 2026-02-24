# Semantic-inference2
This is a redesign of the original semantic inference project. The aim of this project is to query vast knowledge graphs using RAG.

## Instructions
### Neo4j
- Install neo4j, create a new project and add a local DBMS, create a password and start the project that will hold your knowledge graph.
- Open neo4j browser for that project (This is where you can look at your knowledge graph)
### Flask API (semantic_search_api.py)
- Enter your neo4j login credentials (the default user is neo4j and the password is your DBMS password)
- Get your gemini google API key at https://ai.google.dev/gemini-api/docs/api-key and replace it in the code
- Install the required dependencies from requirements.txt
- Open a terminal in the local directory and run ```python semantic_search_api.py```
### Chainlit GUI
- Open a terminal in the local directory and run ```pip install chainlit```
- Run ```chainlit run app.py```
- Open the gui on ```http://localhost:8000/```

From there you can upload a ```.jsonl``` or ```.json``` file to create a new knowledge graph inside Neo4j (embeddings will be generated automatically) and preform a semantic search.
