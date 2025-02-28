#!/usr/bin/env python

# pip install "langserve[all]"

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes


from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv, find_dotenv
import os 

_ = load_dotenv(find_dotenv())

# 1. Create prompt template
system_template = "Translate the following into {target_lang}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{translate_text}')
])

# 2. Create model
model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
translate_chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    translate_chain,
    path="/chain/translate",
)

if __name__ == "__main__":
    
    # set CORS for all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    
    # from langserve import RemoteRunnable
    # remote_chain = RemoteRunnable("http://localhost:8000/chain/translate")
    # remote_chain.invoke({"target_lang": "italian", "translate_text": "hi"})