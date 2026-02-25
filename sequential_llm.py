from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # StrOutputParser converts raw text into clean text string.
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template= 'Generate a detailed report on topic {topic}',
    input_variables= ['topic'] # Explicitly tells langsmith that variable it should look for is in dict named {topic}
)

prompt2 = PromptTemplate(
    template= 'Generate a summary on the following text {text}',
    input_variables= ['text']
)

llm = ChatGroq(
    model = 'llama-3.3-70b-versatile',
    temperature= 0
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic' : 'History related to pak-india war 1965?'})
print(result)