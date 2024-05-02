import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from quanto import quantize, freeze

st.title("Application de génération de titre")
st.markdown("Cette application utilise un large modèle de langage (LLM) de la famille StableLM (StableL2-Zephyr-1.6B) pour générer un titre à partir d'un texte donné.")


st.subheader("Analyse exploratoire")

def load_data():
    return pd.read_csv("selected_df.csv")

data = load_data()
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
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.title("Nuage de mots du texte")
#st.pyplot()

st.title("StableLM2 Zephyr 1.6B Title Prediction")
# Saisie de données et prédiction
gen_config = {
    "temperature": 0.1,
    "top_p": 1,
    "repetition_penalty": 0.75,
    "top_k": 8,
    "do_sample": True,
    "max_new_tokens": 30,
    }
    
text_input = st.text_area("Entrer le texte:", "")

if st.button("Generate Title"):
	modelpath="stabilityai/stablelm-2-zephyr-1_6b"

	tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True, use_fast=False,) 
	model = AutoModelForCausalLM.from_pretrained(
	    modelpath,    
	    torch_dtype=torch.bfloat16,
	    device_map="cpu",
	    #attn_implementation="flash_attention_2",
	    trust_remote_code=True,       # needed for Stable LM 2 based models
	)
	quantize(model, weights=torch.int8, activations=None)
	freeze(model)



	question = f"Generate an appropriate title to the following text in a maximum of 10 words. Do not provide explanations or justifications. Make sure the answer contains only the title in this format 'title' : {text_input}"
	messages = [{"role": "user", "content": question}]
	input_tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cpu")
	with torch.no_grad():
		output_tokens = model.generate(input_tokens, **gen_config, pad_token_id=tokenizer.eos_token_id,)
	output_tokens = output_tokens[0][len(input_tokens[0]):]
	output = tokenizer.decode(output_tokens, skip_special_tokens=True)
	st.subheader("Titre généré :")
	st.write(output)
