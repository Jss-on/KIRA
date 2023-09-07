# Grouping Imports: Standard Libraries, Third-Party Libraries, and Local Modules
import os
import logging
import textwrap
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator

# Third-Party Libraries
import together
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, JSONLoader
from langchain.prompts import PromptTemplate
import fitz

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
import textwrap
import textract



os.environ["TOGETHER_API_KEY"] = "02f9b4bf3afc9eb312604d442f0f51d37de58f58552db6ea892ee21eff593950"
# Constants and Configuration
together.api_key = os.environ["TOGETHER_API_KEY"]
# Configure basic logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class RagChain:
    def __init__(self, user_directory, k=5, api_key=None, model_type="llama2", model_size=70):
        self.user_directory = user_directory
        self.k = k
        self.api_key = os.environ["TOGETHER_API_KEY"]
        self.model_type = model_type
        self.model_size = model_size
        self.llm = None  # To store the LLM model instance
        self.llm_chain = None
        self.embedding = None
        self.retriever = None
        
        logging.info(f"User directory: {self.user_directory}")
        logging.info(f"Retrieval K: {self.k}")
        logging.info(f"Together Api key: {self.api_key}")
        logging.info(f"Model type and size: {self.model_type, model_size}")
        logging.info(f'Current instance of llm: {self.llm}')
        logging.info(f'Current instance of llm_chain: {self.llm}')
        logging.info(f'Current instance of embedding: {self.llm_chain}')
        logging.info(f'Current instance of retriever: {self.retriever}')
        logging.info("Initiating RagChain initiate_llm method")
        self.initiate_llm()
        logging.info("Done initiating RagChain Class")
    
    def set_api_key(self, api_key):
        self.api_key = api_key
        together.api_key = self.api_key

    def save_api_key_to_env(self):
        
        # Determine the path to the .env file (assuming it's in the same directory as the script)
        env_path = os.path.join(os.getcwd(), '.env')

        # Open the .env file in write mode (this will overwrite the file if it already exists)
        with open(env_path, 'w') as f:
            f.write(f"TOGTHER_API_KEY={self.api_key}\n")

        print(f"API key saved to {env_path}")


    def parse_pdf(self, file_path):
        # Your PDF parsing logic here
        logging.info("Start parsing pdf")
        doc = fitz.open(file_path) # open a document
        logging.info("successfully open the file")
        base_file_name = os.path.splitext(file_path)[0]
        logging.info(f"base file name: {base_file_name}")
        for page in doc:  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            page_number = page.number  # zero-based page number

            # save each page as a separate text file
            # page_path = os.path.join(base_file_name, f"page_{page_number}.txt")
            page_path = "_".join([base_file_name, f"page_{page_number}.txt"])

            logging.info(f"Page path: {page_path}")
            with open(page_path, "w") as f:
                f.write(text)
            f.close()
            logging.info("Successfully writing the page as text file.")
    

    def wrap_text_preserve_newlines(self, text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(self, llm_response):
        output_text = ""
    
        # Wrap and add the LLM response result
        output_text += self.wrap_text_preserve_newlines(llm_response['result'])
        
        # Add sources information
        output_text += '\n\nSources:'
        for source in llm_response["source_documents"]:
            output_text += '\n' + str(source.metadata['source'])
            
        return output_text

    def split_list(self, input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    def load_llm_model(self):

        logging.info("Starting to load llm model.")
        together.api_key = self.api_key
        if self.model_type == "llama2":
            if self.model_size == 70:
                # models = together.Models.list()
                together.Models.start("togethercomputer/llama-2-70b-chat")
                self.llm = TogetherLLM(
                    model= "togethercomputer/llama-2-70b-chat",
                    temperature = 0.1,
                    max_tokens = 1024
                )   
                together.Models.stop("togethercomputer/llama-2-13b-chat")
                together.Models.stop("togethercomputer/llama-2-7b-chat")
                logging.info("Loaded 70B llm succesfully")
            elif self.model_size == 13:
                # models = together.Models.list()
                together.Models.start("togethercomputer/llama-2-13b-chat")
                self.llm = TogetherLLM(
                    model= "togethercomputer/llama-2-13b-chat",
                    temperature = 0.1,
                    max_tokens = 1024
                )
                together.Models.stop("togethercomputer/llama-2-70b-chat")
                together.Models.stop("togethercomputer/llama-2-7b-chat")
                logging.info("Loaded 13B llm succesfully")
            else:
                together.Models.start("togethercomputer/llama-2-7b-chat")
                self.llm = TogetherLLM(
                    model= "togethercomputer/llama-2-7b-chat",
                    temperature = 0.1,
                    max_tokens = 1024
                )
                together.Models.stop("togethercomputer/llama-2-13b-chat")
                together.Models.stop("togethercomputer/llama-2-70b-chat")
                logging.info("Loaded 7B llm succesfully")
    
    def parse_file(self, full_file_path,file_extension):
        # Extract text from the file
        extracted_text = textract.process(full_file_path).decode()
        # print(extracted_text)
        
        # Split the full file path into name and extension
        file_name_without_extension, existing_extension = os.path.splitext(full_file_path)
        
        # Generate a new file path, appending an underscore if it's already a .txt file
        new_file_path = f"{file_name_without_extension}{'_.txt' if existing_extension == '.txt' else '.txt'}"
    
        try:
        # Save the extracted text to a .txt file
            with open(new_file_path, 'w') as f:
                f.write(extracted_text)
        except TypeError as e:
            print(f"An error occurred while writing to the file: {e}")
            # Additional error handling can go here

    def load_retriever(self):
        logging.info("Starting load_retriever.")
        logging.info(f"User directory: {self.user_directory}")
        # self.user_directory = os.path.join(self.user_directory, '')
        #Iterate through all files in the user_directory and its subdirectories
        supported_extensions = [".csv", ".doc", ".docx", ".eml", ".epub", ".gif", ".txt", ".pdf", ".htm", ".html", ".jpeg", ".jpg",".log", ".mp3", ".msg", ".odt", ".ogg", ".png", ".pptx", ".ps", ".psv", ".rtf", ".tab", ".tff", ".tif", ".tiff", ".tsv", ".wav", ".xls", ".xlsx"]
        for dirpath, dirnames, filenames in os.walk(self.user_directory):
            for filename in filenames:
                # Check if the file is a PDF
                # if filename.endswith('.pdf'):
                #     # Construct the full file path
                #     full_file_path = os.path.join(dirpath, filename)
                #     # Pass it to your PDF parser
                #     self.parse_pdf(full_file_path)
                
                logging.info(f"Filename: {filename}")
                if filename.endswith(('.c', '.h', '.py', '.cpp', '.md', '.json')):
                    full_file_path = os.path.join(dirpath, filename)
                    logging.info(f"Full file path: {full_file_path}")
                    file_root, file_ext = os.path.splitext(filename)
                    new_filename = f"{file_root}.txt"
                    new_full_file_path = os.path.join(dirpath, new_filename)
                    os.rename(full_file_path, new_full_file_path)
                    logging.info(f"Renamed: {full_file_path} -> {new_full_file_path}")
                    filename = new_filename[::]
                

                file_extension = os.path.splitext(filename)[1]
                if file_extension in supported_extensions:
                    full_file_path = os.path.join(dirpath, filename)
                    logging.info(f"Textract full file path: {full_file_path}")
                     # Pass it to your file parser
                    self.parse_file(full_file_path, file_extension)

                logging.info("Done parsing the file.")

                
                    
        logging.info("Starting document loader.")
        text_loader_kwargs={'autodetect_encoding': True}
        # loader = DirectoryLoader(show_progress=True,use_multithreading=True)
        # root_directory = os.path.join(self.user_directory, '')
        # logging.info(f"Root Directory : {root_directory}")
        loader = DirectoryLoader(self.user_directory, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True, loader_kwargs=text_loader_kwargs, show_progress=True,use_multithreading=True)
        documents = loader.load()

        logging.info(f"Number of documents: {len(documents)}")
        #splitting the text into
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Sample text: {texts[0]}")

        split_docs_chunked = self.split_list(texts, 100)
        self.load_bge_embeddings()
        logging.info("Succesfull loaded embeddings.")

        split_docs_chunked_list = list(split_docs_chunked)

        persist_directory = os.path.join(self.user_directory,"db")
        logging.info(f"User persist directory: {persist_directory}")
        for split_docs_chunk in tqdm(split_docs_chunked_list, desc="Processing chunks"):
            vectordb = Chroma.from_documents(documents=split_docs_chunk,
                                            embedding=self.embedding,
                                            persist_directory=persist_directory)
            vectordb.persist()
        
        self.retriever = vectordb.as_retriever(search_kwargs={"k": self.k})



    def load_bge_embeddings(self):
        logging.info("Loading bge embeddings")
        model_name = "BAAI/bge-base-en"
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

        self.embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs=encode_kwargs
        )
    # ... Include other utility methods
    def initiate_llm(self):
        
        self.load_retriever()
        logging.info("Successfully loaded the retriever.")
        logging.info("Starting system prompt setup.")

        ## Default LLaMA-2 prompt style
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        DEFAULT_SYSTEM_PROMPT = """\
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
            SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
            prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
            return prompt_template


        sys_prompt = """You are a helpful, respectful and honest assistant. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

        instruction = """CONTEXT:/n/n {context}/n

        Question: {question}"""
        get_prompt(instruction, sys_prompt)

        self.load_llm_model()

        logging.info("Done loading llm model.")

        prompt_template = get_prompt(instruction, sys_prompt)

        llama_prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": llama_prompt}
        self.llm_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        chain_type="stuff",
                                        retriever=self.retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True)
        
        logging.info('Succesfully created llm_chain')


    def update_llm_chain(self,user_directory, k=5, api_key=None, model_type="llama2", model_size=70):
        self.user_directory = user_directory
        self.k = k
        self.api_key = api_key# or os.environ.get("TOGETHER_API_KEY", "02f9b4bf3afc9eb312604d442f0f51d37de58f58552db6ea892ee21eff593950")
        os.environ["TOGETHER_API_KEY"] = self.api_key
        self.model_type = model_type
        self.model_size = model_size
        self.initiate_llm()
        logging.info("Successfully updated llm chain")



    def query(self, query_string):
        # This method will contain the core logic of your `llm_response` function
        # It will use the utility methods defined above and the instance variables for configurations

        # Initialize API ke y
        # self.set_api_key(self.api_key)

        llm_response = self.llm_chain(query_string)
        response = self.process_llm_response(llm_response)
        return response


# Class Definitions
class TogetherLLM(LLM):
    
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""    
    model: str = "togethercomputer/llama-2-70b-chat"
    temperature: float = 0.7
    max_tokens: int = 512
    class Config:
        extra = Extra.forbid
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        api_key = get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        values["together_api_key"] = api_key
        return values
    @property
    def _llm_type(self) -> str:
        return "together"
    def _call(self, prompt: str, **kwargs: Any) -> str:
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt, model=self.model, max_tokens=self.max_tokens, temperature=self.temperature)
        text = output['output']['choices'][0]['text']
        return text
    def update_api(self, api_key):
        self.together_api_key = api_key

def save_api_key_to_env(api_key):
        
    # Determine the path to the .env file (assuming it's in the same directory as the script)
    env_path = os.path.join(os.getcwd(), '.env')

    # Open the .env file in write mode (this will overwrite the file if it already exists)
    with open(env_path, 'w') as f:
        f.write(f"TOGETHER_API_KEY={api_key}\n")

    print(f"API key saved to {env_path}")

# Main Logic
if __name__ == "__main__":
    # Your original main logic here
    pass  # Placeholder, replace with your original code
