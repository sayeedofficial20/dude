from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  #will use only in case single meassage output is required
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

domain=input("Enter the domain of your application (e.g., healthcare, finance, education): ")
query=input("Enter the specific query or question you want to ask: ")

messages=ChatPromptTemplate.from_messages([('system', "You are a helpful assistant in the domain of {domain}."), ('human', "Help me with explaining {query} in simple terms of less than 25 words.")])

input_mssg=messages.invoke({"domain": domain, "query": query})
# messages=[SystemMessage(content="You are a helpful assistant."), HumanMessage(content="Hello, how can you assist me today?")]

model=ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

model_response=model.invoke(input_mssg)
print("Model Response:")
# print(AIMessage(content=model_response.content))
print(model_response.content)