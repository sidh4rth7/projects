from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from nodes import create_chatbot
import asyncio
import os
import dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


async def create_agent(docs_info=None):
    async with MultiServerMCPClient(
        {
            "server":{
                "url":"http://localhost:8000/sse",
                "transport":"sse",
                "timeout": 30
            }
        }
    ) as client:
        # Get MCP tools
        tools = client.get_tools()
        
        # Create the graph builder
        graph_builder = StateGraph(MessagesState)
        
        # Create nodes
        chatbot_node = create_chatbot(docs_info)
        graph_builder.add_node("chatbot", chatbot_node)

# Custom async tool node to handle async MCP tools
        async def async_tool_executor(state):
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check if there are tool calls
            tool_calls = None
            if hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls
            elif hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs["tool_calls"]
                
            if not tool_calls:
                return {"messages": messages}
            
            # Process each tool call
            new_messages = messages.copy()
            
            for tool_call in tool_calls:
                # Handle different formats of tool_call
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "tool-call-id")
                else:
                    tool_name = tool_call.name
                    tool_args = tool_call.args if hasattr(tool_call, "args") else {}
                    tool_id = getattr(tool_call, "id", "tool-call-id")
                
                # Print debug info
                print(f"Executing tool: {tool_name}")
                print(f"Tool args: {tool_args}")
                
                # Find the matching tool
                tool = next((t for t in tools if t.name == tool_name), None)
                
                if not tool:
                    # Tool not found
                    tool_error = f"Error: {tool_name} is not a valid tool, try one of {[t.name for t in tools]}."
                    new_messages.append(AIMessage(content=tool_error))
                else:
                    try:
                        # Execute the async tool
                        if asyncio.iscoroutinefunction(tool.coroutine):
                            result = await tool.coroutine(**tool_args)
                        else:
                            # Fall back to sync execution if needed
                            result = tool.func(**tool_args) if hasattr(tool, 'func') else tool(**tool_args)
                        
                        print(f"Tool result: {result}")
                        
                        # Add tool result
                        new_messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id,
                            name=tool_name
                        ))
                    except Exception as e:
                        # Handle errors
                        error_msg = f"Error: {str(e)}\n Please fix your mistakes."
                        print(f"Tool error: {error_msg}")
                        new_messages.append(AIMessage(content=error_msg))
            
            return {"messages": new_messages}


# Add the async tool executor node
        graph_builder.add_node("tools", async_tool_executor)
        
        # Define router function to handle tool calls
        def router(state):
            messages = state["messages"]
            last_message = messages[-1]
            
            has_tool_calls = False
            if isinstance(last_message, AIMessage):
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    has_tool_calls = True
                elif hasattr(last_message, "additional_kwargs") and last_message.additional_kwargs.get("tool_calls"):
                    has_tool_calls = True
            
            return "tools" if has_tool_calls else "end"
        
        # Add edges
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            router,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph_builder.add_edge("tools", "chatbot")
        
        # Compile the graph
        graph = graph_builder.compile()
        return graph, client  # Return client to keep it alive