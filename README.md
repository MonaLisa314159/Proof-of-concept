Preuve de concept réalisée sur les larges modèle du langage: Llama 2 13B, Mistral 7B et Stable LM 2 1.6B
La preuve de concept est constituée d'une analyse comparative de ces trois modèles et d'un déploiement sur le Cloud du modèle retenu.

L'analyse est axée sur la génération de texte, en particulier, la génération d'un titre pou un texte donné.
Les modèles sont utilisés sans fine-tuning. 

Les données de test des mmodèles sont isssues de Kaggle Dataset «1300+ Towards DataScience Medium Articles Dataset ».
Ce dataset constitue une compilation d'articles de blog provenant de la plateforme Medium, publiés sous la rubrique "Towards Data Science".

Des tests seront effectués via différents prompts (simples ou avec des instructions précises) pour évaluer la pertinence des titres générés.
Pour l’évaluation des performances des modèles, les métriques utilisées sont :
• Similarité Cosinus
• Similarité de Levenshtein
• BLEU (Biligual Evaluation Understudy)
• ROUGE score (Recall-Oriented Understudy for Gisting Evaluation) :
o Rouge-1
o Rouge-L

Les modèles Llama 2 et Mistral sont chargés via Ollama, Stable LM 2 via HuggingFace.
Réalisation d'un Dashboard développé à l'aide de la bibliothèque Streamlit. Il inclut une analyse exploratoire des données utilisées pour tester différents modèles de génération de titres.
Le dashboard permet d’interroger le modèle retenu via Hugging Face Inference API.
