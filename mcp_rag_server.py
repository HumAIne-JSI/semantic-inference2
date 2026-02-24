"""
MCP Server for RAG Semantic Inference System
Exposes Neo4j knowledge graph operations and semantic search capabilities as MCP tools.

Usage:
    python mcp_rag_server.py

Or configure in Claude Desktop config:
    {
      "mcpServers": {
        "rag-semantic-inference": {
          "command": "python",
          "args": ["C:/Users/ailab/Documents/code/rag-semantic-inference/mcp_rag_server.py"]
        }
      }
    }
"""

import asyncio
import json
import logging
from typing import Any
import requests
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-rag-server")

# Configuration
API_BASE_URL = "http://127.0.0.1:5001"

# Initialize MCP Server
app = Server("rag-semantic-inference")

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools for the RAG system."""
    return [
        Tool(
            name="semantic_search",
            description=(
                "Perform semantic search on the Neo4j knowledge graph using natural language queries. "
                "Retrieves relevant entities, relationships, and technical specifications from the AAS (Asset Administration Shell) graph. "
                "Best for: equipment queries, technical specifications, availability status, manufacturer information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query (e.g., 'Name all drilling machines', 'What is the voltage of GreenHarmony EcoPro?')"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="cypher_query",
            description=(
                "Execute a Cypher query directly on the Neo4j knowledge graph. "
                "Use this for precise graph traversals, complex filtering, or when semantic search is insufficient. "
                "Requires knowledge of Cypher query language and graph schema."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "Cypher query to execute (e.g., 'MATCH (n:Asset) RETURN n LIMIT 10')"
                    }
                },
                "required": ["cypher"]
            }
        ),
        Tool(
            name="list_equipment_types",
            description=(
                "List all equipment types available in the knowledge graph. "
                "Returns equipment categories like Drilling, CircleCutting, Sawing, Milling, Grinding, Welding."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="count_machines",
            description=(
                "Count machines by type and optionally by availability status. "
                "Performs accurate recursive counting across all nesting levels in the graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "equipment_type": {
                        "type": "string",
                        "description": "Type of equipment (e.g., 'drilling', 'circular saw')"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["available", "unavailable", "all"],
                        "description": "Availability filter (default: all)",
                        "default": "all"
                    }
                },
                "required": ["equipment_type"]
            }
        ),
        Tool(
            name="get_equipment_specs",
            description=(
                "Get detailed technical specifications for specific equipment. "
                "Returns voltage, current, dimensions, manufacturer, and availability."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "equipment_name": {
                        "type": "string",
                        "description": "Name of equipment (e.g., 'GreenHarmony EcoPro')"
                    }
                },
                "required": ["equipment_name"]
            }
        ),
        Tool(
            name="update_availability",
            description=(
                "Update availability status of machines. Changes 'availability' property in graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "machine_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of machine names"
                    },
                    "status": {
                        "type": "boolean",
                        "description": "true=available, false=unavailable"
                    }
                },
                "required": ["machine_names", "status"]
            }
        )
    ]

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "semantic_search":
            return await semantic_search_tool(arguments)
        elif name == "cypher_query":
            return await cypher_query_tool(arguments)
        elif name == "list_equipment_types":
            return await list_equipment_types_tool()
        elif name == "count_machines":
            return await count_machines_tool(arguments)
        elif name == "get_equipment_specs":
            return await get_equipment_specs_tool(arguments)
        elif name == "update_availability":
            return await update_availability_tool(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def semantic_search_tool(args: dict) -> list[TextContent]:
    """Execute semantic search via the API."""
    query = args.get("query")
    n_results = args.get("n_results", 10)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic-search",
            json={"query": query, "n": n_results},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        # Collect streaming response
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if 'chunk' in data:
                            chunks.append(data['chunk'])
                        elif 'error' in data:
                            return [TextContent(type="text", text=f"Error: {data['error']}")]
                    except json.JSONDecodeError:
                        continue
        
        result = ''.join(chunks)
        return [TextContent(type="text", text=result if result else "No results found.")]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

async def cypher_query_tool(args: dict) -> list[TextContent]:
    """Execute Cypher query via the API."""
    cypher = args.get("cypher")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/cypher",
            json={"query": cypher},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        formatted_result = json.dumps(result, indent=2)
        return [TextContent(type="text", text=formatted_result)]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

async def list_equipment_types_tool() -> list[TextContent]:
    """List all equipment types."""
    query = "List all equipment types"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic-search",
            json={"query": query, "n": 5},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if 'chunk' in data:
                            chunks.append(data['chunk'])
                    except json.JSONDecodeError:
                        continue
        
        result = ''.join(chunks)
        return [TextContent(type="text", text=result if result else "No equipment types found.")]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

async def count_machines_tool(args: dict) -> list[TextContent]:
    """Count machines by type and availability."""
    equipment_type = args.get("equipment_type")
    status = args.get("status", "all")
    
    if status == "available":
        query = f"Count all available {equipment_type} machines"
    elif status == "unavailable":
        query = f"Count all unavailable {equipment_type} machines"
    else:
        query = f"How many {equipment_type} machines are there in total?"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic-search",
            json={"query": query, "n": 5},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if 'chunk' in data:
                            chunks.append(data['chunk'])
                    except json.JSONDecodeError:
                        continue
        
        result = ''.join(chunks)
        return [TextContent(type="text", text=result if result else "Could not determine count.")]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

async def get_equipment_specs_tool(args: dict) -> list[TextContent]:
    """Get equipment specifications."""
    equipment_name = args.get("equipment_name")
    query = f"What are the properties and specifications of {equipment_name}?"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic-search",
            json={"query": query, "n": 5},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if 'chunk' in data:
                            chunks.append(data['chunk'])
                    except json.JSONDecodeError:
                        continue
        
        result = ''.join(chunks)
        return [TextContent(type="text", text=result if result else f"No details found for {equipment_name}.")]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

async def update_availability_tool(args: dict) -> list[TextContent]:
    """Update machine availability status."""
    machine_names = args.get("machine_names", [])
    status = args.get("status")
    
    status_str = "True" if status else "False"
    
    try:
        results = []
        for machine_name in machine_names:
            cypher = """
            MATCH (m:Asset {assetKind: 'Instance'})
            WHERE m.idShort = $machine_name
            SET m.availability = $status
            RETURN m.idShort as name, m.availability as new_status
            """
            
            response = requests.post(
                f"{API_BASE_URL}/cypher",
                json={
                    "query": cypher,
                    "params": {"machine_name": machine_name, "status": status_str}
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if result:
                results.append(f"✓ Updated {machine_name}: availability = {status_str}")
            else:
                results.append(f"✗ Machine not found: {machine_name}")
        
        output = "Availability Update Results:\n" + "\n".join(results)
        return [TextContent(type="text", text=output)]
    
    except requests.exceptions.RequestException as e:
        return [TextContent(type="text", text=f"API Error: {str(e)}")]

# ============================================================================
# RESOURCES
# ============================================================================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="rag://config/api",
            name="API Configuration",
            mimeType="application/json",
            description="API endpoints and configuration"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "rag://config/api":
        config = {
            "api_base_url": API_BASE_URL,
            "endpoints": {
                "semantic_search": f"{API_BASE_URL}/semantic-search",
                "cypher": f"{API_BASE_URL}/cypher"
            },
            "default_results": 10
        }
        return json.dumps(config, indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run the MCP server."""
    logger.info("Starting RAG Semantic Inference MCP Server...")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info("Exposing 6 tools: semantic_search, cypher_query, list_equipment_types, count_machines, get_equipment_specs, update_availability")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
