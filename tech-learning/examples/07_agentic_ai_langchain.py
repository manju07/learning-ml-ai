"""
Agentic AI Example with LangChain
Demonstrates building autonomous agents with tools and memory
"""

from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationBufferMemory
import os

# Set up OpenAI API key (set your key)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def setup_basic_agent():
    """Setup a basic LangChain agent with tools"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Define custom tools
    def get_weather(location: str) -> str:
        """Get weather for a location (mock implementation)"""
        # In production, this would call a weather API
        return f"Weather in {location}: Sunny, 72°F"
    
    def calculate(expression: str) -> str:
        """Evaluate mathematical expressions"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create tools
    tools = [
        Tool(
            name="Weather",
            func=get_weather,
            description="Get weather information for a location. Input should be a location name."
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Evaluate mathematical expressions. Input should be a valid Python expression."
        ),
        DuckDuckGoSearchRun(name="Search"),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def setup_agent_with_memory():
    """Setup agent with conversation memory"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Simple tools
    tools = [
        Tool(
            name="Search",
            func=lambda q: f"Search results for: {q}",
            description="Search the web for information"
        )
    ]
    
    # Agent with memory
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return agent

def setup_code_agent():
    """Setup agent that can write and execute Python code"""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Python REPL tool
    python_tool = PythonREPLTool()
    
    agent = initialize_agent(
        tools=[python_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def example_research_agent():
    """Example: Research agent that gathers information"""
    print("=" * 60)
    print("Research Agent Example")
    print("=" * 60)
    
    agent = setup_basic_agent()
    
    # Research query
    query = """
    Research the latest developments in artificial intelligence.
    Find information from multiple sources and summarize:
    1. Key recent breakthroughs
    2. Major companies involved
    3. Future predictions
    """
    
    try:
        result = agent.run(query)
        print(f"\nResearch Result:\n{result}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure OPENAI_API_KEY is set and you have API credits")

def example_calculator_agent():
    """Example: Agent that performs calculations"""
    print("\n" + "=" * 60)
    print("Calculator Agent Example")
    print("=" * 60)
    
    agent = setup_basic_agent()
    
    query = "Calculate 15 * 23 + 45, then find the square root of the result"
    
    try:
        result = agent.run(query)
        print(f"\nCalculation Result:\n{result}")
    except Exception as e:
        print(f"Error: {e}")

def example_code_generation_agent():
    """Example: Agent that writes and executes code"""
    print("\n" + "=" * 60)
    print("Code Generation Agent Example")
    print("=" * 60)
    
    agent = setup_code_agent()
    
    query = """
    Write a Python function to calculate Fibonacci numbers.
    Test it with n=10 and print the result.
    """
    
    try:
        result = agent.run(query)
        print(f"\nCode Execution Result:\n{result}")
    except Exception as e:
        print(f"Error: {e}")

def example_conversational_agent():
    """Example: Agent with memory for conversation"""
    print("\n" + "=" * 60)
    print("Conversational Agent with Memory Example")
    print("=" * 60)
    
    agent = setup_agent_with_memory()
    
    queries = [
        "My name is Alice",
        "What's my name?",
        "I like Python programming",
        "What programming language do I like?"
    ]
    
    try:
        for query in queries:
            print(f"\nUser: {query}")
            result = agent.run(input=query)
            print(f"Agent: {result}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all examples"""
    print("Agentic AI Examples with LangChain")
    print("=" * 60)
    print("\nNote: These examples require:")
    print("1. OPENAI_API_KEY environment variable set")
    print("2. OpenAI API credits")
    print("3. Internet connection for search tools")
    print("\nUncomment the examples below to run them:\n")
    
    # Uncomment to run examples
    # example_research_agent()
    # example_calculator_agent()
    # example_code_generation_agent()
    # example_conversational_agent()
    
    print("\n" + "=" * 60)
    print("To run examples:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Uncomment the example functions in main()")
    print("3. Run: python 07_agentic_ai_langchain.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

