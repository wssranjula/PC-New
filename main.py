from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from prompt import prompt,cp
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-06-05",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=''
    # other params...
)
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./cleaned_document.docx")

data = loader.load()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt,
        ),
        ("human", "here is the {company_policy}"),
    ]
)

chain = prompt | llm | StrOutputParser()

response  = chain.invoke(
    {
        "legislation": data,
        "company_policy": cp,
      
    }
)
print(response)
