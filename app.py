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


#MCP_SERVER = "http://localhost:5002"


# import httpx

# async def call_fastapi_mcp(machine: str, status: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.post("http://localhost:9090/update_availability", json={
#             "machine": machine,
#             "status": status
#         })
#         if response.status_code != 200:
#             return f"❌ Error: {response.json().get('detail', 'Unknown error')}"
#         return response.json()["message"]


# import ollama

# async def detect_intent_and_extract(query: str) -> dict:
#     """Use local AI to detect intent and extract machine name and availability status"""
    
#     prompt = f'''
# You classify queries and extract info. Always return a JSON object like this:

# {{"intent": "...", "machines": "[...]", "status": "..."}}

# Rules:
# - If the user wants to change a machine’s availability:
#   → intent = "change_availability"
#   → machine = list of names (like ["Machine #3", "Conveyor A"])
#   → status = "True" (online/up) or "False" (offline/down)

# - If the user asks for a count (like "how many machines..."):
#   → intent = "other"
#   → machine = ""
#   → status = "counting"

# - For anything else:
#   → intent = "other"
#   → machine = ""
#   → status = ""

# Examples:
# "make machine 5 offline" → {{"intent": "change_availability", "machines": "["Machine #5"]", "status": "False"}}
# "make machine 5 and 10 available" → {{"intent": "change_availability", "machines": "["Machine #5", "Machine #10"]", "status": "True"}}
# "turn on conveyor A" → {{"intent": "change_availability", "machines": "["Conveyor A"]", "status": "True"}}
# "how many drilling machines are online?" → {{"intent": "other", "machines": "", "status": "counting"}}
# "name all drilling machines" → {{"intent": "other", "machines": "", "status": ""}}

# Now process this:
# "{query}"
# '''



#     try:
#         response = ollama.chat(
#             model='llama3.2:3b',  # Use a small, fast model
#             messages=[{'role': 'user', 'content': prompt}],
#             options={'temperature': 0.1}  # Low temperature for consistent extraction
#         )
#         print(prompt)
#         print(response)
#         # Parse the JSON response
#         import json
#         result = json.loads(response['message']['content'].strip())
#         return result
        
#     except Exception as e:
#         print(f"Error with local AI: {e}")
#         # Fallback to simple keyword detection
#         content_lower = query.lower()
#         if any(word in content_lower for word in ['machine', 'available', 'unavailable', 'offline', 'online', 'down', 'up']):
#             return {"intent": "change_availability", "machine": "", "status": ""}
#         return {"intent": "other", "machine": "", "status": ""}

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
                initial_value="aas"
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
    cl.user_session.set("graph_format", "aas")
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
    current_format = cl.user_session.get("graph_format", "aas")
    
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
    
    ################    MCP
    # content = message.content.lower()

    # if "make machine" in content:
    #     # Extract machine name and status from the message
    #     parts = content.split()
    #     machine_idx = parts.index("machine") + 1
    #     if machine_idx < len(parts):
    #         machine = f'Machine #{parts[machine_idx].replace("#", " #").title()}'  # Handle "Machine #10" format
    #         print(f"Machine: {machine}")
    #         # Determine status based on available/unavailable keywords
    #         if "unavailable" in content:
    #             status = "False"
    #         elif "available" in content:
    #             status = "True"
    #         else:
    #             await cl.Message(content="Please specify 'available' or 'unavailable'").send()
    #             return
                
    #         result = await call_fastapi_mcp(machine, status)
    #         await cl.Message(content=result).send()
    #     else:
    #         await cl.Message(content="Please specify a machine name").send()
    # else:
    #     await cl.Message(content="🧠 No control intent detected. Try asking a question.").send()

    ##################
    # ai_result = await detect_intent_and_extract(query)
    # print(f"AI Result: {ai_result}")
    # if ai_result["intent"] == "change_availability":
    #     machines = list(ai_result.get("machines", ""))#.strip()
    #     status = ai_result.get("status", "").strip()
        
    #     print(f"Detected machines: {machines}, Status: {status}")
    #     for machine in machines:
    #         machine = machine.strip()  # Normalize machine name
    #         if not machine or not status:
    #             await cl.Message(content="Could not identify the machine name or desired status. Please be more specific.").send()
    #             return
                
    #         #try:
    #         print(f"Machine: {machine}, Status: {status}")
    #         result = await call_fastapi_mcp(machine, status)
    #         await cl.Message(content=result).send()
    #         print(f"Result: {result}")
            #except Exception as e:
            #    print(e)
                #await cl.Message(content=f"Error updating machine availability: {str(e)}").send()
            #return
    ##################


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
            json={"query": query_obj, "n": 11},#, "intent": ai_result},
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