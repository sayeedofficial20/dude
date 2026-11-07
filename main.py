from langserve import add_routes
from fastapi import FastAPI
from pydantic import BaseModel,Field    
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello, How are you?"}

class TranslationRequest(BaseModel):
    lang :str
    text: str


@app.post("/llm")
def translate(request:TranslationRequest):
    llm = ChatGroq(model='allam-2-7b')
    prompt_v1=PromptTemplate(template="You are a language translation expert, translate the input text : {text}to completely {lang} language only.This is very important:", input_variables=["text", "lang"])
    prompt_v2 = PromptTemplate(template="You are a strict language translation expert. Translate the entire input text '{text}' exclusively into {lang}, using only {lang} vocabulary and script. Never mix languages, retain untranslated words, or include explanations, headings, or additional text. Translate all content accurately while preserving its original meaning and context. Return ONLY the translated text in {lang}.", input_variables=["text", "lang"])
    output_parser=StrOutputParser()
    # chain= prompt_v1 | llm | output_parser
    chain= prompt_v2 | llm | output_parser
    result=chain.invoke({"text":request.text,"lang":request.lang})
    return {"result": result}