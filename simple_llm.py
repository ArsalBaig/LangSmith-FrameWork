from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # StrOutputParser converts raw text into clean text string.
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template('{question}')

llm = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    temperature= 0
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'question' : 'What is the capital of Pakistan? Also write a 5 line sentence on it.'})
print(result)