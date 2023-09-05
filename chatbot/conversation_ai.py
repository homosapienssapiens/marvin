from utils import DocsJSONLLoader, get_file_path, get_openai_api_key, get_query_from_user
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from rich.console import Console
from dotenv import load_dotenv

console = Console()

recreate_chroma_db = False
chat_type = "memory_chat"


def load_documents(file_path: str):
    loader = DocsJSONLLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1600,
        length_function = len,
        chunk_overlap = 160
    )

    return text_splitter.split_documents(data)

def get_chroma_db(embeddings, documents, path):

    if recreate_chroma_db:
        console.print("RECREANDO CHROMA DB")
        return Chroma.from_documents(
            documents = documents,
            embedding = embeddings,
            persist_directory = path
        )
    else:
        console.print("CARGANDO CHROMA EXISTENTE")
        return Chroma(persist_directory = path, embedding_function = embeddings)
    
def run_conversation(vectorstore, chat_type, llm):

    console.print(
        "\n[purple]IA:[/purple] Hola! Qué quieres preguntarme sobre Transformers e inteligencia artificial?"
    )

    if chat_type=="qa":
        console.print(
            "\n[green]Estás utilizando el chatbot en modo de preguntas y respuestas. Este chatbot genera respuestas basándose puramente en la consulta actual sin considerar el historial de la conversación.[/black]"
        )
    elif chat_type == "memory_chat":
        console.print(
            "\n[green]Estás utilizando el chatbot en modo memoria. Este chatbot genera respuestas basándose en la historia de la conversación y en la consulta actual.[/green]"
        )
    
    retriever = vectorstore.as_retriever(
        search_kwargs = {'k': 3}
    )

    chat_history = []

    while True:
        console.print("\n[red]Tu:[/red]")
        query = get_query_from_user()

        if query.lower() == "salir":
            break

        if chat_type=="qa":
            response = process_qa_query(
                query = query,
                retriever = retriever,
                llm = llm)
        elif chat_type == "memory_chat":
            response = process_memory_query(
                query = query,
                retriever = retriever,
                llm = llm,
                chat_history = chat_history
            )

            console.print(f"[purple]IA: [/purple] {response}")

def process_qa_query(query, retriever, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever
    )
    console.print("[black]La IA está pensando...[/black]")
    return qa_chain.run(query)

def process_memory_query(query, retriever, llm, chat_history):
    conversation = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        verbose = True
    )
    console.print("[black]La IA está pensando...[/black]")
    print(f"La historia antes de esta respuesta es: {chat_history}")
    result = conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

def main():

    
    load_dotenv(".env")
    documents = load_documents(get_file_path())
    get_openai_api_key()
    #embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    embeddings = HuggingFaceInstructEmbeddings(
    model_name = "hkunlp/instructor-base",
    model_kwargs = {"device" : "cpu"}
    )
    
    vectorstore_chroma = get_chroma_db(
        embeddings, documents, "chroma_docs"
    )

    console.print(f"[yellow]Documentos {len(documents)} cargados.[/yellow]")

    llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.2,
        max_tokens = 1000
    )

    run_conversation(vectorstore_chroma, chat_type, llm)


if __name__ == "__main__":
    main()
