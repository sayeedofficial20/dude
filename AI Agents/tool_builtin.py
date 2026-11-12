# from langchain_community.tools import DuckDuckGoSearchRun

# search_tool=DuckDuckGoSearchRun()

# result=search_tool.invoke('what is viral crime news today?')
# print(result)

#-----------------------------------------------------------------------------------------------

from langchain_community.tools import ShellTool

shellTool=ShellTool()

results=shellTool.invoke('whoami')
print(results)