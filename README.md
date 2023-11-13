# KIRA - Knowledge Intensive Retrieval Assistant

## Description
KIRA is a Gradio-based chatbot designed to provide Knowledge-Intensive Retrieval Assistance. Utilizing Retrieval-Augmented Generation (RAG) for document queries, it's powered by the open-source Language Model, Llama 2, running on its default 13B variant.

## Prerequisites
- TOGETHER API KEY

## Installation

### Clone the repository:
```bash
git clone https://github.com/Jss-on/KIRA.git
```

### Navigate into the project directory:
```bash
cd rag_gradio_app/
```

### Install the required dependencies:
```
pip install -q -r requirements.txt
```

## Configuration
Before running KIRA, you must supply a TOGETHER API KEY. Insert the API key in the logic.py file where indicated.

## How to Run
After the prerequisites are installed and the API key is configured, run KIRA using the following command:

```
python app.py
```

## Technology Stack
- Gradio
- Llama 2 (13B Variant)
- RAG (Retrieval-Augmented Generation)

## Contributing
If you wish to contribute to the development of KIRA, feel free to fork the project, make changes, and submit pull requests. All contributions are welcome!

## License
KIRA is open-source and available under the MIT License.
