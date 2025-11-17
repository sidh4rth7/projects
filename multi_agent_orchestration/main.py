import streamlit as st
import asyncio
from agent import create_agent
from langchain_core.messages import HumanMessage

async def main():
    # Create the agent
    agent, client = await create_agent()
    
    # Get user input from command line
    user_input = input("What would you like to ask? ")
    
    # Create a proper initial message
    initial_message = HumanMessage(content=user_input)
    
    try:
        # Use the agent asynchronously
        print("Processing your request...")
        result = await agent.ainvoke({"messages": [initial_message]})
        
        # Print the results
        for message in result["messages"]:
            if hasattr(message, "type") and message.type == "human":
                print(f"User: {message.content}")
            elif hasattr(message, "type") and message.type == "tool":
                print(f"Tool Result: {message.content}")
                # If it's an image generation result, extract URL
                if "image" in message.content.lower() and "url" in message.content.lower():
                    print("Image Generated Successfully!")
            else:
                print(f"AI: {message.content}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Keep the client alive until all operations are done
    # In a real application, you'd keep the client active as long as needed

if __name__ == "__main__":
    asyncio.run(main())