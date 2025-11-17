# my-python-app/README.md

# Multi-Agent-Orcgestration

The Multi-Agent-Orchestration project is an advanced LLM-based agent system featuring multiple tool calls and cutting-edge techniques. It implements Model-Centric Programming (MCP), treating AI as an autonomous agent with toolkit discovery capabilities, tool combination abilities, and enhanced task handling. Through open protocols, MCP delivers exceptional flexibility and extensibility while maintaining careful design for complexity management and safety. The system also integrates the CodeAct technique, which provides an alternative to standard JSON function-calling by leveraging Turing-complete Python programming. This approach enables more complex tasks to be solved in fewer steps by dynamically combining and transforming outputs from multiple tools, creating a powerful and adaptable orchestration framework.


## Project Structure

- `agent.py`: Contains the `create_agent` function for setting up the asynchronous agent and managing message flow.
- `main.py`: The entry point of the application that processes user input and interacts with the agent.
- `nodes.py`: Defines the chatbot creation and manages the flow of messages, including tool definitions.
- `server.py`: Sets up the server and provides the `get_tools` function for tool management.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd agent_orchestration
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python main.py
```

Follow the prompts to interact with the chatbot. You can use various tools provided by the application for enhanced functionalities.


## Future updates:
- Explore and integrate the "codeact" technique followed by `manusAI`.
- Integration of more tools
