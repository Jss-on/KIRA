import gradio as gr
import inspect
import uuid
import os
import shutil
from logic import RagChain, save_api_key_to_env
from copy import deepcopy
import logging
import hashlib
import threading




class UserSession:
    _instances = {}  # Class-level dictionary to hold instances
    
    def __new__(cls, userid, *args, **kwargs):
        # If an instance already exists for this userid, return it
        if userid in cls._instances:
            return cls._instances[userid]
        
        # Otherwise, create a new instance and store it in the dictionary
        instance = super(UserSession, cls).__new__(cls)
        cls._instances[userid] = instance
        return instance
    
    def __init__(self, userid, llm_instance=None, file=None, state=None, user_count=0, user_data_path=None):
        self.userid = userid
        self.llm_instance = llm_instance
        self.file = file
        self.state = state
        self.user_count = user_count
        self.user_data_path = user_data_path
        
    def add_count(self):
        self.user_count += 1



class AppManager:
    def __init__(self):
        self.sessions = {}  # Dictionary to hold UserSession instances keyed by user_id
        self.llm_instances = {}
        self.global_count = 0  # Global session count
        self.history = []
        self.state = None
        logging.info(f"State value: %s", self.state)
        

    
    def get_state(self):
        return self.state
    
    def get_llm_instance(self, user_data_path):
            logging.info(f"Value of : {self.state not in self.sessions}")
            logging.info(f"Sessions : {self.sessions.keys()}")
            session = self.sessions.get(self.state, None)
            if session.llm_instance is None:
                logging.info("Creating session llm instance")
                self.sessions[self.state] = UserSession(self.state, llm_instance=RagChain(user_data_path, k=3, model_size=13))
            return self.sessions[self.state].llm_instance

    def set_file_object(self, file):
            session = self.sessions.get(self.state, None)
            if session:
                self.sessions[self.state].file = file
            
            logging.info("file upload complete")

    def reset_user_session_count(self):
            session = self.sessions.get(self.state, None)
            logging.info(f"reset user session: {session.userid}")
            if session:
                logging.info(f"session persisted")
                self.sessions[self.state].user_count = 0
                self.sessions[self.state].llm_instance = None
                self.delete_user_session_logs()

    def process_file_and_text(self, text, session_state):
            self.global_count += 1
            logging.info(f"State_process_file_and_text: %s" % self.state)
            session = self.sessions.get(self.state, None)
            logging.info(f"LLM instance: {session.llm_instance}")
            logging.info(f"Session: %s" % session)
            if session:
                logging.info(f"User : {session.userid} Count: {session.user_count}")
                if session.user_count == 0:
                    # File processing logic
                    logging.info("File processing")
                    try:
                        for file in session.file:
                            path_components = file.name.split(os.path.sep)
                            path_components.insert(3, session.userid)
                            dst_path = os.path.join('/', *path_components)
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            shutil.move(file.name, dst_path)
                            print(f"Moved {file.name} -> {dst_path}")
                    except FileNotFoundError:
                        print("File not found.")
                    
                    user_data_path = os.path.join('/', *path_components[:4])
                    logging.info(f"user_data_path: %s" % user_data_path)
                    logging.info("before get_llm_instance")
                    session.user_data_path = user_data_path
                    llm = self.get_llm_instance(user_data_path)
                else:
                    llm = self.get_llm_instance(session.user_data_path)

                logging.info("After llm_instance")
                response = llm.query(text)
                
                
            history_record = {"query": text, "response": response, "question_counter": self.global_count}
            self.history.append(history_record)
            
            logging.info(f"Query: {text}")
            logging.info(f"Response: {response}")
            logging.info(f"User : {session.userid} Count: {session.user_count}")
            logging.info(f"Session Count: {self.global_count}")
            session.user_count += 1
                
            return response, session_state
    def delete_user_session_logs(self):
            logging.info("Delete User Session Logs")
            try:
                session = self.sessions.get(self.state, None)
                if os.path.exists(session.user_data_path):
                    print("The path exists.")
                    shutil.rmtree(session.user_data_path)
                else:
                    print("The path does not exist.")
        # os.mkdir(session.user_data_path)
            except TypeError:
                dir_path = os.path.join('/','tmp','gradio',self.state)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print("The path exists.")
                else:
                    print("The path does not exist.")

    def authenticate(self, username, password):
            user_db = {
                "0000": hashlib.sha256("11qqQQ@@".encode()).hexdigest(),
                "0001": hashlib.sha256("pass".encode()).hexdigest(),
                "0002": hashlib.sha256("pass".encode()).hexdigest()
            }
            
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            logging.info("authentication")
            if username in user_db and user_db[username] == hashed_password:
                if username not in self.sessions:
                    self.sessions[username] = UserSession(username)
                    self.state = username
                    logging.info(f"State value: %s", self.state)
                    logging.info(f"State type : {type(self.state)}")
                return True
            return False


# Initialize AppManager
am= AppManager()

with gr.Blocks(title = "KIRA") as iface:
    # Add a title and an icon placeholder
    gr.HTML('<h1 style="text-align: center;"><b>KIRA: <i>Knowledge Intensive Retrieval Assistant</i></b></h1>')
    
    with gr.Accordion("Read here!"):
        gr.Markdown("""
        

        ## What is KIRA?

        KIRA stands for **Knowledge Intensive Retrieval Assistant**. It's a tool that helps you get answers based on text you upload. Here's what you should know about KIRA:

        ### What Can KIRA Do?

        - KIRA is really smart. It uses a state of the art open source large language model to understand text.
        - It can help you get better answers if your questions are related to the text you upload.

        ### What Should You Keep in Mind?

        - KIRA's answers depend on how good your questions are and the quality of the uploaded text.
        - It's better to ask questions in a way that matches the style of the uploaded text.

        ### How to Get the Best Out of KIRA?

        To get the most accurate answers:
        - Ask questions that are closely related to the uploaded document.
        - Make sure the uploaded text is clear and easy to understand.

        ### Want to Explore More?

        KIRA has a lot of untapped potential. Feel free to try different things to see how you can get the most accurate answers.

                
        """)


    # gr.HTML('<img src="static/logo.png" alt="App Icon" width="100" height="100">')
    logging.info("Inside init_gr")
    with gr.Tab("QA"):
        file = gr.File(file_count='multiple', label="Upload your document here")
        chatbot = gr.Chatbot()
        state = gr.State()
        prompt = gr.Textbox(label="Prompt", placeholder="Ask here...")
        
        clear = gr.ClearButton([prompt, chatbot, file])
    
        # display = gr.Textbox(label="Display", placeholder="You can see here your documents")



    
    def reset_session():
            userid = am.get_state()
            if userid:
                am.reset_user_session_count()

    clear.click(fn=reset_session)

    def upload_file(file):
            userid = am.get_state()
            if userid:
                am.reset_user_session_count()
                am.set_file_object(file)

    file.upload(fn=upload_file, inputs=[file])

    def respond(prompt, chat_history, session_state):
            userid = am.get_state()
            if userid:
                bot_response, user_state = am.process_file_and_text(prompt, session_state)
                chat_history.append((prompt, bot_response))
                
                return "", chat_history

    prompt.submit(respond, [prompt, chatbot, state], [prompt, chatbot])
logging.info("Inside before launce")
logging.info(f"State value: %s", am.get_state)

iface.queue(max_size=20, api_open=False)

iface.launch(share=True, auth=am.authenticate)
logging.info("Done running App")



