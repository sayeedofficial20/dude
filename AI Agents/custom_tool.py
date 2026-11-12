# from langchain_core.tools import tool

# @tool
# def multiply(a:int,b:int)->int:
#     """This tool helps in Multiplying the Provided Numbers"""
#     return a*b

# result=multiply.invoke({'a':4,'b':7})
# print(result)

# print(multiply.name)
# print('--------------------------------------')
# print(multiply.description)
# print('--------------------------------------')
# print(multiply.args)

#-------------------------------------------------------------------------------------------

# from langchain_core.tools import StructuredTool
# from pydantic import BaseModel,Field


# class MultiplyInput(BaseModel):
#     a:int=Field(required=True,description='The First Number to add')
#     b:int=Field(required=True,description="The Second Number to add")

# def multiplyNumbers(a:int,b:int):
#     return a*b

# multiply_tool=StructuredTool.from_function(
#     func=multiplyNumbers,
#     name="multiply",
#     description='multiply Two Numbers',
#     args_schema=MultiplyInput

# )

# response=multiply_tool.invoke({'a':4,'b':7})

# print(response)
# print(multiply_tool.name)
# print('--------------------------------------')
# print(multiply_tool.description)
# print('--------------------------------------')
# print(multiply_tool.args)

#-------------------------------------------------------------------------------------------

from langchain_core.tools import tool

@tool
def add(a:int,b:int)->int:
    """This tool helps in add the Provided Numbers"""
    return a+b


@tool
def multiply(a:int,b:int)->int:
    """This tool helps in Multiplying the Provided Numbers"""
    return a*b

class MathToolkit:
    def get_tools(self):
        return[add,multiply]

toolkit=MathToolkit()
tools=toolkit.get_tools()

for tool in tools:
    print(tool.name,'->',tool.description)