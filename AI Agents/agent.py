# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# # from langchain.tavily import  TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_classic.agents import create_react_agent,AgentExecutor #this legacy version
# from  langchain.agents import create_agent
# from dotenv import load_dotenv

# load_dotenv()

# llm=ChatOpenAI(model='gpt-4o')

# search_tool = TavilySearchResults(
#     max_results=3,
#     search_depth="basic"
# )

# agent=create_agent(
#     model=llm,
#     tools=[search_tool],
#     system_prompt="You are a helpful AI assistant.Use DuckDuckGo search for current information.Respond politely and concisely."
# )

# response=agent.invoke({"messages":[{"role":"user",'content':'What the hot Tech new today'}]})
# result=response["messages"][-1].content
# print(result)


# method 2(old version)
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_classic.agents import create_react_agent, AgentExecutor
# from langchain_classic import hub
# from dotenv import load_dotenv

# load_dotenv()
# llm=ChatOpenAI(model='gpt-4o')

# prompt = hub.pull("hwchase17/structured-chat",
#                 dangerously_pull_public_prompt=True
# )                                                          # Needs {tools}, {tool_names}, {agent_scratchpad}

# searchtool=DuckDuckGoSearchRun()

# agent = create_react_agent(llm, [searchtool], prompt)
# agent_executor = AgentExecutor(agent=agent, tools=[searchtool],verbose=True,handle_parsing_errors=True)

# # Invoked with "input"
# response = agent_executor.invoke({"input": "What is the weather?"})
# print(response["output"])


from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Fetch ReAct prompt using LangSmith
client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)

# Create search tool
searchtool = DuckDuckGoSearchRun()

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[searchtool],
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[searchtool],
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({
    "input": "What is artificial intelligence?"
})

print(response["output"])
