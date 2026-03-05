import streamlit as st
import speech_recognition as sr
import pyttsx3
import nltk
import re
import numpy as np
import datetime
import webbrowser
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG
st.set_page_config(page_title="VoxAI Assistant", page_icon="🤖", layout="wide")


# HEADER
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=130)
st.title("VoxAI - Intelligent Voice Assistant")
st.caption("Real-time Speech Recognition + NLP Intent Detection")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    show_confidence = st.toggle("Show Intent Confidence", value=True)

    st.markdown("---")
    st.markdown("### 🧠 Features")
    st.write("✔ Speech Recognition")
    st.write("✔ NLP Intent Detection")
    st.write("✔ Wikipedia Search")
    st.write("✔ Web Automation")
    st.write("✔ Text-to-Speech")


# NLTK SETUP
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

recognizer = sr.Recognizer()


# INTENTS
intents = {
    "greeting": ["hello", "hi", "hey assistant"],
    "open_google": ["open google", "launch google"],
    "open_youtube": ["open youtube"],
    "time": ["what is the time", "tell me time"],
    "date": ["what is the date", "today date"],
    "calculate": ["calculate", "what is"],
    "wikipedia": ["who is", "tell me about"],
    "exit": ["stop", "exit", "quit"]
}

intent_labels = []
intent_sentences = []

for key, values in intents.items():
    for sentence in values:
        intent_sentences.append(sentence)
        intent_labels.append(key)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(intent_sentences)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\-\*/]', '', text)
    return text

def detect_intent(user_input):
    user_input = clean_text(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    index = np.argmax(similarity)
    confidence = similarity[0][index]
    return intent_labels[index], confidence

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []

if "context_image" not in st.session_state:
    st.session_state.context_image = None


# BUTTONS
col1, col2, col3 = st.columns(3)

with col1:
    start = st.button("🎤 Start Listening")

with col2:
    stop = st.button("🛑 Stop")

with col3:
    clear = st.button("🚪 Clear Chat")

if clear:
    st.session_state.history = []
    st.session_state.context_image = None
    st.success("Conversation cleared.")

# LISTENING
if start:
    with st.spinner("Listening..."):
        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            intent, confidence = detect_intent(text)

            # Default image
            image_url = None

            if intent == "greeting":
                response = "Hello Tanya! How can I assist you today?"
                image_url = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"

            elif intent == "open_google":
                webbrowser.open("https://www.google.com")
                response = "Opening Google."
                image_url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

            elif intent == "open_youtube":
                webbrowser.open("https://www.youtube.com")
                response = "Opening YouTube."
                image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg"

            elif intent == "time":
                now = datetime.datetime.now().strftime("%H:%M:%S")
                response = f"The current time is {now}"
                image_url = "https://cdn-icons-png.flaticon.com/512/992/992700.png"

            elif intent == "date":
                today = datetime.datetime.now().strftime("%B %d, %Y")
                response = f"Today's date is {today}"
                image_url = "https://cdn-icons-png.flaticon.com/512/747/747310.png"

            elif intent == "calculate":
                try:
                    expression = clean_text(text)
                    result = eval(expression)
                    response = f"The result is {result}"
                    image_url = "https://cdn-icons-png.flaticon.com/512/992/992651.png"
                except:
                    response = "Sorry, I could not calculate that."

            elif intent == "wikipedia":
                try:
                    topic = re.sub(r"(who is|tell me about)", "", text.lower()).strip()
                    summary = wikipedia.summary(topic, sentences=2)
                    response = summary
                    image_url = f"https://source.unsplash.com/600x400/?{topic}"
                except:
                    response = "Sorry, I could not find information."

            else:
                response = "I am still learning. Please try again."

            st.session_state.history.append(("You", text))
            st.session_state.history.append(("VoxAI", response))
            st.session_state.context_image = image_url

            speak(response)

            if show_confidence:
                st.sidebar.metric("Intent Confidence", f"{round(confidence*100,2)} %")

        except:
            st.error("Sorry, could not understand audio.")


# CONTEXT IMAGE DISPLAY
if st.session_state.context_image:
    st.image(st.session_state.context_image, use_container_width=True)

# CHAT DISPLAY
st.subheader("💬 Conversation")

for speaker, message in st.session_state.history:
    st.markdown(f"**{speaker}:** {message}")