from mcp.server.fastmcp import FastMCP
from langchain_experimental.utilities import PythonREPL
import io
import base64
import matplotlib.pyplot as plt
from openai import OpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import asyncio
from googlesearch import search

@mcp.tool()
async def generate_image(prompt: str) -> str:
    """
    Generate an image using DALL-E based on the given prompt.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Invalid prompt")
    
    try:
        # Since this is an async function, we need to handle the synchronous OpenAI call properly
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        )
        
        # Return both success message and URL
        return f"Successfully generated an image of {prompt}! Here's the URL: {response.data[0].url}"
    except Exception as e:
        return f"Error generating image: {str(e)}"

repl = PythonREPL()

@mcp.tool()
def data_visualization(code: str):
    """Execute Python code. Use matplotlib for visualization."""
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error creating chart: {str(e)}"
        
@mcp.tool()
def python_repl(code: str):
    """Execute Python code."""
    return repl.run(code)

def get_tools(retriever_tool=None):
    # Only include tools that are working
    base_tools = [generate_image, python_repl, data_visualization]
    
    if retriever_tool:
        base_tools.append(retriever_tool)
    
    return base_tools

if __name__ == "__main__":
    mcp.run(transport="sse")