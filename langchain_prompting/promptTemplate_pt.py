from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("LangChain Prompting Example") 
role=st.selectbox("Select a role", ["Assistant", "Customer Support", "Salesperson"])
goal=st.text_input("Enter your goal")
context=st.text_area("Enter any additional context")

prompt_template = PromptTemplate(template="""You are a {role}. Your goal is to {goal}. Here is some additional context: {context}. Please provide a response based on this information.""", input_variables=["role", "goal", "context"],validate_template=True)

final_temp=prompt_template.invoke({"role": role, "goal": goal, "context": context})

prompt_template.save("prompt_template_part1.json")

chatmodel=ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

if st.button("Generate Response"):
    response=chatmodel.invoke(final_temp)
    st.write("Response:")
    st.write(response.content)