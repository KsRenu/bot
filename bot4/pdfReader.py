#/Users/Renu/Documents/chat-with-pdf-doc/ChatPDF/Software-Testing-A-Craftsman-s-Approach-Fourth-Edition-Paul-C-Jorgensen-2.pdf


import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmkCUuORIyV00QkWWMdzUbR3DQqv358bM"

# Configure file upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('pdf.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    # Receive file from the user
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()

        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS index from PDF documents
        db = FAISS.from_documents(pages, embeddings)

        # Receive input query from the user
        query = request.form['query']

        # Query for text
        docs = db.similarity_search(query)

        # Get page content from similar documents
        content = "\n".join([x.page_content for x in docs])

        # Prepare input text for generative AI
        qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
        input_text = f"{qa_prompt}\nContext:{content}\nUser question:\n{query}"

        # Initialize Google Generative AI model
        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        # Generate response using Google Generative AI
        result = llm.invoke(input_text)

        return jsonify({'result': result.content})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
