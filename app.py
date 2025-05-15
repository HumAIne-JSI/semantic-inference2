import os
import time
import asyncio
import json
from typing import List, Dict, Any
import httpx
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput
from chainlit.action import Action
import pandas as pd

# Server configuration
API_HOST = "http://127.0.0.1:5001"

# Global variables for tracking progress
file_upload_task_id = None
embedding_task_id = None

# Helper functions
async def check_progress(task_id, message):
    """Poll progress endpoint and update a message's progress bar"""
    client = httpx.AsyncClient()
    url = f"{API_HOST}/progress"
    
    try:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.replace('data: ', ''))
                    
                    if data.get("status") == "completed":
                        await message.update(
                            content=f"Processing completed! {data.get('processed')} of {data.get('total')} items processed.",
                            progress=1.0
                        )
                        return True
                    
                    if data.get("total", 0) > 0:
                        progress = data.get("processed", 0) / data.get("total", 1)
                        await message.update(
                            content=f"Processing: {data.get('processed')} of {data.get('total')} items...",
                            progress=progress
                        )
                        
                except Exception as e:
                    print(f"Error parsing progress data: {e}")
                
    except Exception as e:
        print(f"Error checking progress: {e}")
        await message.update(content=f"Error checking progress: {e}", progress=None)
    
    return False


# @cl.on_chat_start
# async def setup_avatars():
#     # Set avatars for different roles
#     cl.user_session.set(
#         "avatar",
#         {
#             #"user": "public/user-icon.png",  # Path/URL for user avatar
#             "assistant": "https://humaine-horizon.eu/hmn-uploads/2024/01/humaine_logo.png",  # Path/URL for assistant avatar
#             "system": "https://humaine-horizon.eu/hmn-uploads/2024/01/humaine_logo.png"  # Optional system avatar
#         }
#     )

@cl.on_chat_start
async def start():
    """Initialize the chat interface"""
    # cl.user_session.set(
    #     "logo",
    #     "https://humaine-horizon.eu/hmn-uploads/2024/01/humaine_logo.png",
    # )

    await cl.Message(content="# RAG Knowledge Base System 🧠").send()
    
    # Create chat settings with format selector and action buttons
    settings = await cl.ChatSettings(
        [
            Select(
                id="graph_format",
                label="Knowledge Graph Format",
                values=["csv", "aas", "rdf"],
                initial_value="rdf"
            ),
            Select(
                id="kg_action",
                label="Knowledge Graph Actions",
                values=["none", "delete-kg", "upload-kg"],
                initial_value="none"
            )
        ]
    ).send()
    
    # Store initial settings in session
    cl.user_session.set("graph_format", "rdf")
    cl.user_session.set("kg_action", "none")
    
    # Welcome message with command help
    await cl.Message(content="""👋 Welcome! Type any query to perform a semantic search in the knowledge graph.

**Special Commands:**
- `/delete-kg` - Delete the knowledge graph
- `/upload-kg` - Upload a file to create/update the knowledge graph
- `/graph-format [csv|aas|rdf]` - Change the knowledge graph format
    """).send()

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes including format selection and actions"""
    # Get current format from session
    current_format = cl.user_session.get("graph_format", "rdf")
    
    # Handle graph format change only if it's different
    new_format = settings.get("graph_format")
    if new_format and new_format != current_format:
        await change_graph_format(new_format)
        cl.user_session.set("graph_format", new_format)
    
    # Handle actions
    action = settings.get("kg_action")
    if action == "delete-kg":
        # Clear action before performing operation to prevent retrigger
        await cl.ChatSettings(
            [
                Select(
                    id="graph_format",
                    label="Knowledge Graph Format",
                    values=["csv", "aas", "rdf"],
                    initial_value=cl.user_session.get("graph_format")
                ),
                Select(
                    id="kg_action",
                    label="Knowledge Graph Actions",
                    values=["none", "delete-kg", "upload-kg"],
                    initial_value="none"
                )
            ]
        ).send()
        await delete_knowledge_graph()
        
    elif action == "upload-kg":
        # Clear action before performing operation to prevent retrigger
        await cl.ChatSettings(
            [
                Select(
                    id="graph_format",
                    label="Knowledge Graph Format",
                    values=["csv", "aas", "rdf"],
                    initial_value=cl.user_session.get("graph_format")
                ),
                Select(
                    id="kg_action",
                    label="Knowledge Graph Actions",
                    values=["none", "delete-kg", "upload-kg"],
                    initial_value="none"
                )
            ]
        ).send()
        await upload_knowledge_graph()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages, including special commands"""
    query = message.content.strip()
    
    # Handle special commands
    if query == "/delete-kg":
        await delete_knowledge_graph()
        return
    elif query == "/upload-kg":
        await upload_knowledge_graph()
        return
    elif query.startswith("/graph-format"):
        parts = query.split()
        if len(parts) != 2:
            await cl.Message(content="Invalid format. Usage: `/graph-format [csv|aas|rdf]`").send()
            return
        format_type = parts[1].lower()
        if format_type not in ["csv", "aas", "rdf"]:
            await cl.Message(content="Invalid format type. Choose from: csv, aas, rdf").send()
            return
        await change_graph_format(format_type)
        return
    elif query.startswith("/"):
        await cl.Message(content=f"Unknown command: {query}\n\nAvailable commands:\n- `/delete-kg`\n- `/upload-kg`\n- `/graph-format [csv|aas|rdf]`").send()
        return
        
    # Default behavior - handle as semantic search query
    if not query:
        await cl.Message(content="No query provided. Please type a search query.").send()
        return
    
    # Show a thinking message with progress indicator
    thinking_msg = cl.Message(content="Searching the Knowledge Graph... ")
    await thinking_msg.send()
    
    async with httpx.AsyncClient() as client:
        query_obj = {"output": query}
        
        async with client.stream(
            "POST",
            f"{API_HOST}/semantic-search",
            json={"query": query_obj, "n": 10},
            headers={"Accept": "text/event-stream"},
            timeout=60
        ) as response:
            
            if response.status_code != 200:
                await thinking_msg.update(content=f"Error: HTTP {response.status_code}")
                return
            
            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if data.get("error"):
                            print(f"Error: {data['error']}")
                            return
                        elif data.get("chunk"):
                            full_response += data["chunk"]
                            await thinking_msg.stream_token(data["chunk"])
                    except json.JSONDecodeError:
                        pass
            
            await thinking_msg.update(content=full_response)

async def change_graph_format(format_type):
    """Change the knowledge graph format"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_HOST}/set-graph-format",
                json={"graphFormat": format_type}
            )
            
            if response.status_code == 200:
                await cl.Message(content=f"Graph format successfully changed to {format_type}!").send()
            else:
                await cl.Message(content=f"Error changing graph format: {response.text}").send()
    
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

async def delete_knowledge_graph():
    """Delete the knowledge graph"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_HOST}/delete-graph")
            if response.status_code == 200:
                await cl.Message(content="Knowledge graph deleted successfully!").send()
            else:
                await cl.Message(content=f"Error deleting knowledge graph: {response.text}").send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

async def upload_knowledge_graph():
    """Handle file upload for knowledge graph creation"""
    global file_upload_task_id
    graph_format = cl.user_session.get("graph_format", "rdf")
    
    files = await cl.AskFileMessage(
        content="Please upload your knowledge graph file (CSV, JSONL, or TXT).",
        accept=[".csv", ".jsonl", ".json", ".txt"],
        max_size_mb=10,
        max_files=1
    ).send()
    
    if not files:
        await cl.Message(content="No file was uploaded. Operation cancelled.").send()
        return
    
    file = files[0]
    msg = cl.Message(content="Starting file upload...")
    await msg.send()
    
    try:
        async with httpx.AsyncClient() as client:
            form_data = {"graphFormat": graph_format}
            files_data = {"file": (file.name, open(file.path, "rb"))}
            
            response = await client.post(
                f"{API_HOST}/create-graph",
                data=form_data,
                files=files_data
            )
            
            if response.status_code == 202:
                file_upload_task_id = str(time.time())
                asyncio.create_task(check_progress(file_upload_task_id, msg))
            elif response.status_code == 200:
                await cl.Message(content="File uploaded and processed successfully! Check the server for updates regarding embeddings.").send()
            else:
                await cl.Message(content=f"Error uploading file: {response.text}").send()
    
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()