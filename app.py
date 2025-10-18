# app.py
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import easyocr
import tempfile
import os
from ultralytics import YOLO
from gtts import gTTS
import io

# LangChain and DB components
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM

# Translation
from deep_translator import GoogleTranslator

# --- CONFIGURATION (from config.py) ---
# ⚠️ CRITICAL: The order of this list MUST EXACTLY MATCH the order of classes
# used during the YOLOv8 model training. Check the 'data.yaml' file from training.
CLASS_NAMES = ['Aspirin', 'Crocin', 'Meftal', 'Paracetamol', 'Saridon']
YOLO_MODEL_PATH = "yolo_finetuned.pt"
MEDICINE_CSV = "medicines.csv"
DB_LOCATION = "./chroma_medicine_db"
SUPPORTED_LANGS = {'en': 'English', 'te': 'Telugu'}

# --- MODEL LOADING (Cached for performance) ---

@st.cache_resource
def load_models():
    """Loads all AI models and the vector DB into memory and caches them."""
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_reader = easyocr.Reader(['en', 'te'], gpu=False)
    llm = OllamaLLM(model="llama3.2")
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Setup Vector DB from db.py logic
    add_documents = not os.path.exists(DB_LOCATION)
    vector_store = Chroma(
        collection_name="medicine_information",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    if add_documents:
        print("First time setup: Creating and persisting vector DB...")
        try:
            df = pd.read_csv(MEDICINE_CSV)
            documents = [
                Document(
                    page_content=f"Name: {row['Name']}\nGeneric: {row['Generic Name']}\nUses: {row['Uses']}\nDosage: {row['Typical Dosage']}\nSide Effects: {row['Side Effects']}",
                    metadata={"Name": row["Name"], "Generic": row["Generic Name"]},
                ) for _, row in df.iterrows()
            ]
            vector_store.add_documents(documents=documents)
            vector_store.persist()
        except FileNotFoundError:
            st.error(f"FATAL ERROR: Medicine data file not found at '{MEDICINE_CSV}'. Please create it.")
            st.stop()
            
    retriever = vector_store.as_retriever(search_kargs={"k": 5})
    print("Models loaded successfully!")
    return yolo_model, ocr_reader, llm, retriever

# --- CORE FUNCTIONS (from ocr_yolo.py & agent.py) ---

def process_image(image_bytes, yolo_model, ocr_reader):
    """Takes image bytes and returns detected medicine name and OCR text."""
    # Convert bytes to a NumPy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save to a temporary file for model processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as fp:
        cv2.imwrite(fp.name, img)
        image_path = fp.name

    # OCR
    ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
    extracted_text = " ".join(ocr_result)

    # YOLO detection
    results = yolo_model.predict(image_path, verbose=False)
    medicine_name = "Unknown Medicine"
    
    if results and len(results[0].boxes) > 0:
        best_conf_idx = results[0].boxes.conf.cpu().numpy().argmax()
        best_class_id = int(results[0].boxes.cls.cpu().numpy()[best_conf_idx])
        if best_class_id < len(CLASS_NAMES):
             medicine_name = CLASS_NAMES[best_class_id]

    os.remove(image_path) # Clean up temp file
    return medicine_name, extracted_text

def get_agent_response(llm, retriever, user_query, yolo_output, ocr_output):
    """Runs the RAG chain to get a response from the LLM."""
    docs = retriever.get_relevant_documents(user_query)
    info_text = "\n".join([doc.page_content for doc in docs])

    prompt_text = f"""
    You are a helpful AI assistant for identifying medicines. Prioritize the 'YOLO detected' name if it is available. Use the 'OCR text' and the 'Retrieved Info' to provide details like uses, dosage, and side effects.

    Retrieved Info:
    {info_text}
    ---
    Context from Image:
    YOLO detected: {yolo_output}
    OCR text: {ocr_output}
    ---

    User question: {user_query}
    Answer concisely based ONLY on the information provided. If the information is not present, say so.
    """
    
    agent_response = llm.invoke(prompt_text)
    return agent_response

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI Medicine Assistant", layout="wide")

st.title("💊 AI Medicine Assistant")
st.markdown("An intelligent assistant for Professor Karthik, powered by AI.")

# Load models once
yolo_model, ocr_reader, llm, retriever = load_models()

# Initialize session state for chat history and image data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "yolo_result" not in st.session_state:
    st.session_state.yolo_result = ""
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# --- UI LAYOUT (2 Columns) ---
col1, col2 = st.columns(2)

with col1:
    st.header("Visual Input")
    
    # Camera Input
    img_file_buffer = st.camera_input("Take a picture of the medicine")

    if img_file_buffer:
        st.session_state.uploaded_image = img_file_buffer.getvalue()
        with st.spinner("Analyzing image..."):
            st.session_state.yolo_result, st.session_state.ocr_result = process_image(
                st.session_state.uploaded_image, yolo_model, ocr_reader
            )
    
    # Display results from image processing
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Captured Image", use_column_width=True)
        st.success(f"**Detected Medicine:** {st.session_state.yolo_result}")
        with st.expander("See Extracted Text (from OCR)"):
            st.write(st.session_state.ocr_result or "No text found.")


with col2:
    st.header("Conversational Agent")

    # Language Selection
    lang_options = list(SUPPORTED_LANGS.values())
    selected_lang_name = st.selectbox("Choose your language", options=lang_options)
    preferred_lang_code = [code for code, name in SUPPORTED_LANGS.items() if name == selected_lang_name][0]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Translate user query to English for the agent
                english_query = GoogleTranslator(source='auto', target='en').translate(prompt)
                
                response = get_agent_response(
                    llm, 
                    retriever, 
                    english_query, 
                    st.session_state.yolo_result,
                    st.session_state.ocr_result
                )
                
                # Translate response back to user's language
                translated_response = GoogleTranslator(source='en', target=preferred_lang_code).translate(response)
                
                st.markdown(translated_response)

                # Add text-to-speech audio
                try:
                    tts = gTTS(text=translated_response, lang=preferred_lang_code, slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3', start_time=0)
                except Exception as e:
                    st.warning(f"Could not generate audio: {e}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": translated_response})
