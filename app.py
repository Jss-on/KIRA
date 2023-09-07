import gradio as gr
import inspect
import uuid
import os
import shutil
from logic import RagChain, save_api_key_to_env
from copy import deepcopy
import logging
import hashlib

# Initialize RagChain once (use a placeholder directory)
# llm = RagChain("placeholder_folder", k=1, model_size=7)
COUNT = 0
history = []

# Initialize the record
# history_record = {"query": "", "response": "", "question_counter": 0}

# File path
json_file_path = 'history.json'

class AppManager:

    def __init__(self) -> None:
        self.llm_instances = {}  # Dictionary to hold llm instances keyed by user_id
        self.userid = None
        self.user_session_count = {}
        self.COUNT = 0
        self.history = []

    def get_llm_instance(self, user_data_path):
        if self.userid not in self.llm_instances:
            self.llm_instances[self.userid] = RagChain(user_data_path, k=3, model_size=13)
        return self.llm_instances[self.userid]

    def process_file_and_text(self, file_obj, text, user_id):
        logging.info(f"-----------------------------------------SESSION NO.{self.COUNT} START------------------------------------------- ")
        self.COUNT += 1
         # Incrementing the session counter for the specific userid
        if self.userid in self.user_session_count:
            self.user_session_count[self.userid] += 1
        else:
            self.user_session_count[self.userid] = 1

        try:
            for file in file_obj:
                path_components = file.name.split(os.path.sep)
                path_components.insert(3, self.userid)
                dst_path =  os.path.join(*path_components)
                dst_path = os.path.join('/', dst_path)
                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                # Move the file from its source path to its destination path
                shutil.move(file.name, dst_path)
                print(f"Moved {file.name} -> {dst_path}")
        except FileNotFoundError:
            print("File not found.")
            
        
        user_data_path = os.path.join(*path_components[:4])
        user_data_path = os.path.join('/',user_data_path)
        logging.info(f"User Data Path: {user_data_path}")
        llm = self.get_llm_instance(user_data_path)
        logging.info(f"User ID: {self.userid}")
        # Check if it's the first time the llm instance is being used for this userid
        if self.user_session_count[self.userid] == 1:
            logging.info("Not initiating llm anymore")
            response = "llm is initiated. Go and ask your question."
        else:
            response = llm.query(text)
            

        
        history_record = {"query": text, "response": response, "question_counter": COUNT}
        self.history.append(history_record)

        logging.info(f"Query: {text}")
        logging.info(f"Response: {response}")
        logging.info(f"User : {self.userid} Count: {self.user_session_count[self.userid]}")
        logging.info(f"Session Count: {self.COUNT}")
        logging.info(f"-----------------------------------------SESSION NO.{self.COUNT} END------------------------------------------- ")
        
        return response, user_id

    def authenticate(self, username, password):
        
        """
        Authenticate a user based on the username and password.
        
        Parameters:
        - username (str): The username to authenticate
        - password (str): The password to authenticate with
        
        Returns:
        - bool: True if authentication is successful, False otherwise
        """
        user_db = {
            "0000": hashlib.sha256("11qqQQ@@".encode()).hexdigest(),
            "0001": hashlib.sha256("pass".encode()).hexdigest(),
            "0002": hashlib.sha256("pass".encode()).hexdigest()
        }
        # Hash the input password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Check if the username exists in the "database"
        if username in user_db:
            # Compare the hashed passwords
            if user_db[username] == hashed_password:
                self.userid = username
                return True
        return False

am = AppManager()


iface = gr.Interface(
    fn=am.process_file_and_text,
    inputs=[
        gr.Files(file_count='directory', label="Upload your document here"),
        gr.Textbox(lines = 2, label="Query", placeholder="Enter your query here"),
        gr.State(label="User ID")
    ],
    
    outputs=[gr.Markdown(), gr.State(label="User ID")]
)

# iface.launch(auth=("admin", "1234"))
iface.queue(max_size=5, api_open=False)

iface.launch(share=True,auth=am.authenticate)
