from langchain_core.messages import HumanMessage,ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(model='gpt-4o')

@tool
def add(a:int,b:int)->int:
    """This tool helps in add the Provided Numbers"""
    return a+b
@tool
def multiply(a:int,b:int)->int:
    """This tool helps in Multiplying the Provided Numbers"""
    return a*b

toolkit=llm.bind_tools([add,multiply])

# response=toolkit.invoke([HumanMessage(content='add 5 and 9')])

# print(response)
# print(response.tool_calls)

#------------------------------------------------------------------------------------

messages=[]

query=HumanMessage("give result of add 5 and 9")

messages.append(query)
# print(messages)

ai_message=toolkit.invoke(messages)
messages.append(ai_message)
# print(messages)

tool_call=ai_message.tool_calls[0]
# print('====================')
# print(tool_call)
# print('====================')

tool_message=ToolMessage(content=str(tool_call),
            name=tool_call['name'],
            tool_call_id=tool_call['id'])

messages.append(tool_message)

# print(messages)

final_result=toolkit.invoke(messages)
print(final_result.content)
