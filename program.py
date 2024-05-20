import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
#from transformers import AutoModelForCausalLM
#import torch
#from quanto import quantize, freeze
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
st.title("Application de génération de titre")
st.markdown("Cette application utilise un large modèle de langage (LLM) de la famille StableLM (StableL2-Zephyr-1.6B) pour générer un titre à partir d'un texte donné.")


st.subheader("Analyse exploratoire")

def load_data(path):
    return pd.read_csv(path)

data = load_data("selected_df.csv")
data = data[["Title", "Text", 'Len_text', 'Len_title']]
st.write(data.head(5))

# nombre de mots dans le texte/titre

fig, ax = plt.subplots()
n_bins = st.number_input(
	label = "Choisir un nombre de bins",
	min_value=25,
	value=50
	)
ax.hist(data['Len_text'], bins=n_bins, alpha=0.5, color='g')
plt.title("Distribution de la longueur des Textes")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(data['Len_title'], bins=50, alpha=0.5, color='b')
plt.title("Distribution de la longueur des Titres")
st.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Wordcloud
st.subheader("word cloud")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['Text']))
st.image(wordcloud.to_array(), use_column_width=True)

st.subheader("Comparaison des perfomances")
st.subheader("Llama 13B - Mistral 7B - StableLM Zephyr 1.6B")

df_results = pd.read_csv("results_df.csv", index_col=["Unnamed: 0"])
st.write(df_results)

st.title("Génération de titre avec StableLM2 Zephyr 1.6B ")
# Saisie de données et prédiction
gen_config = {
    "temperature": 0.1,
    "top_p": 1,
    "repetition_penalty": 0.75,
    "top_k": 8,
    "do_sample": True,
    "max_new_tokens": 30,
    }

modelpath="stabilityai/stablelm-2-zephyr-1_6b"
tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True, use_fast=False,) 

my_token = st.secrets["TOKEN"]
client = InferenceClient(model=modelpath, token=my_token)  
text_input = st.text_area("Entrer le texte:", "")


if st.button("Générer un titre"):
    question = f"Generate an appropriate title to the following text. Do not provide justification or explanation: {text_input}.  Make sure the answer contains only the title"
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    output = client.text_generation(prompt, 
				    temperature = 0.1, 
				    top_p = 1,
				    repetition_penalty = 0.75,
				    top_k = 8,
				    do_sample = True,
				    max_new_tokens=30)
    st.subheader("Titre généré :")
    st.write(output)
