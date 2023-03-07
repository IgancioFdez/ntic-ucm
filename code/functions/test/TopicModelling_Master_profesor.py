#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import sys  
get_ipython().system('{sys.executable} -m pip install contractions')
get_ipython().system('pip install pyLDAvis')


# # Imports
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import pandas as pd
import re
import numpy as np
from nltk import word_tokenize, pos_tag


# # Cargar y preparar los datos
# 

# En primer lugar, cargamos los datos. Esta vez vamos a descargarlos de un usuario de github que ha compartido el típico dataset de "20newsGroups" en su cuenta

# In[ ]:


dataset =pd.read_json("https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json")


# In[ ]:


dataset


# Se observa un  texto muy "sucio" que vamos a tener que "limpiar"

# Vamos a comprobar el tamaño del dataset:

# In[ ]:


print("Tenemos un conjunto de {} documentos".format(dataset.shape[0]))


# Despues, quitaremos las filsa con algún valor vacío (NA) y quitaremos los duplicados.

# In[ ]:


print("Existen {} reviews duplicadas".format(np.sum(dataset.duplicated(subset=["content"]))))
# Quitaremos esos duplicados
dataset = dataset.drop_duplicates()
print("Despues de quitar duplicados tenemos un conjunto de {} noticias".format(dataset.shape[0]))


# Observamos que el corpus está sin preprocesar. Hay correos electrónicos, símbolos de retorno de carro ("\n")... 

# # Preprocesado

# ## Normalización

# Vamos a transformar los datos. Comenzaremos trabajando con los componentes que "ensucian" el corpus: los correos electrónicos, presencia de "\n", comillas...

# In[ ]:


# Load the regular expression library
import re
import string
# Quita los emails presententes en un string (toma todos los string desde un espacio hasta la @ y todo lo que hay desde la @ hasta un espacio y lo sustituye por un "")
dataset['text_processed'] = dataset['content'].apply(lambda x: re.sub('\S*@\S*\s?', '', x))
# sustituir el \n
dataset['text_processed'] = dataset['text_processed'].apply(lambda x: re.sub('\n', ' ', x))
# Quitar/eliminar las comillas
dataset['text_processed'] = dataset['text_processed'].apply(lambda x: re.sub('\'', ' ', x))
# Convertir a lowercase
dataset['text_processed'] = dataset['text_processed'].apply(lambda x: x.lower())
# Eliminar puntuación
dataset['text_processed'] = dataset['text_processed'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
dataset['text_processed']


# El dataset es muy grande como para hacer las operaciones que queremos hacer durante la clase de forma rápida. Vamos a quedarnos con sólo una muestra.

# In[ ]:


dataset2 = dataset.sample(700)


# ¿De qué ira el dataset? Una opción es hacer una wordcloud a partir de los datos anteriores. Esto se hace con la librería WordCloud.
# 
# Toma como entrada una frase que agrupa todos las frases del corpus, y genera el gráfico.

# In[ ]:


from wordcloud import WordCloud
# Une las frases
long_string = ','.join(list(dataset2['text_processed'].values))
# Genera un objecto WordCloud 
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=0, contour_color='steelblue')
# Genera el wordcloud
wordcloud.generate(long_string)
# Visualizalo en una imagen
wordcloud.to_image()


# Se observan muchas stopwords como "re" o el auxiliar "will". También las palabras well, even... Que no proporcionan información. Vemos muchas frases a las que se les podría aplicar lematización... Es el momento de usar spacy!

# In[ ]:


#!python -m spacy download en_core_web_sm
#!python -m spacy link en_core_web_sm en
#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
import string
import spacy
punctuations = string.punctuation
nlp = spacy.load("en_core_web_sm")
stop_words = stopwords.words('english')

def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens


# Procesemos las frases con esa función:

# In[ ]:


data_words = list()
for i in dataset2["text_processed"].tolist():
  data_words.append(spacy_tokenizer(i))


# In[ ]:


data_words[1]


# ## Transformación

# Para preprocesar datos de gensim utilizamos las funciones específicas con las que cuenta. 
# Podemos utilizar el método `Phrases` 

# In[ ]:


import gensim
# Construimos los modelos de bigramas y trigramas con gensim
# No devuelve trigramas o bigtamas en si mismos, si no que asocia palabras
# que aparecen juntas en mltitud de ocasiones
bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100) # higher threshold fewer phrases.

# Con esto todo va más rápido, pero ya no pueden modificarse lo anterior
# https://www.kite.com/python/docs/gensim.models.phrases.Phraser
# The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
# not strictly needed for the bigram detection task.
# Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.
bigram_mod = gensim.models.phrases.Phraser(bigram)


# In[ ]:


bigram_mod


# *Si* miramos los resultados de acontinuación, lo que hacen estas librerías es generar bigramas a partir de la similitud que tienen las palabras en un espacio vectorial (utiliza word2vec por detrás). 
# 
# Por ejemplo, los tokens "New" "York" son unidos en el proceso anterior a "New_York".  Algo similar ocurre con "computer_science". Basicamente la idea de los brigramas en gensim es la de agrupar palabras que generalmente se usen juntas, y no incorporar más variables al sistema.

# In[ ]:


[bigram_mod[i] for i in data_words]


# In[ ]:


bigram_mod[data_words[1]]


# In[ ]:


type(bigram_mod)


# Ahora hay que aplicar los modelos a los datos. Para eso utilizamos las funciones que aparecen más abajo. 
# 
# ADemás, se ha incorporado la función filtra_tags, que permite coger tokens de una categoría específica. Muy util apra el topic_modeling

# In[ ]:


# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#stop_words.extend(['day', 'hotel', 'room', 'great', 'night','staff','service'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
# Antes "entrenamos" el generador de bigrams, ahora se generan realmente en nuestros textos
def make_bigrams(textos):
    return [bigram_mod[doc] for doc in textos]

# Función para eliminar cierto tipo de tags
def filtra_tags(textos, tags_permitidas=['NOUN', 'ADJ', 'VERB', 'ADV']):
    textos_out = []
    for sent in textos:
        # Juntal los "trigrams"
        doc = nlp(" ".join(sent)) 
        # Filtra por etiqueta y coge lemma (es algo redundante)
        textos_out.append([token.text for token in doc if token.pos_ in tags_permitidas])
    return textos_out


# Vamos a aplicar esas funciones:

# In[ ]:


import spacy
# Remove Stop Words
#data_words_nostops = remove_stopwords(data_words)
# Crear Bigrams
data_words_bigrams = make_bigrams(data_words)

# Desabilitamos el "NER" y el "Parser" que no lo vamos a usar
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
#  Filtrar por categoría gramatical 
data_pos = filtra_tags(data_words_bigrams, tags_permitidas=['NOUN','ADJ','ADV'])


# In[ ]:


print("Numero tokens antes de filtrar: {} tokens".format(len(data_words_bigrams[2])))
print("Numero tokens despues de filtrar: {} tokens".format(len(data_pos[2])))


# In[ ]:


len(data_pos)


# Despues de ese procesado tenemos un conjunto de 1000 documentos procesados y preparados para introudcir a un modelo de topic modeling

# # Entramiento y validación

# En primer lugar generamos un diccionario utilizando los objetos de Gensim. Ese diccionario contiene un método "doc2bow" que transforma el texto a vectores comprensibes por el modelo LDA.

# In[ ]:


import gensim.corpora as corpora
# Creamos diccionario de términos 
id2word = corpora.Dictionary(data_pos)
print(id2word)


# In[ ]:


# Asignamos a la variable texts nuestro corpus
texts = data_pos
# Transformamos nuestro corpus limpio a Bag of Words. 
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus)


# Los modelos LDA necesitaban además de los vectores de entrada, un número de "topics" para ser entrenados. 
# Como no sabemos a priori cuantos hay, creamos la función "calculo_valor_coherencia" que a partir dle corpus, el diccionario y otros valores calcula un modelo lda, calcula la coherencia entre sus topics y la devuelve

# In[ ]:


# supporting function
def calculo_valor_coherencia(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_pos, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()


# In[ ]:


from tqdm.autonotebook import tqdm
tqdm.pandas()


# Vamos a calcular la coherencia suponiendo distintos números de topics. Por ejemplo de 1 a 22 en saltos de 2.

# In[ ]:


import numpy as np
import tqdm
grid = {}
# Topics range
min_topics = 2
max_topics = 22
step_size = 2
topics_range = range(min_topics, max_topics, step_size)
# Alpha
alpha = 0.01
# Beta
beta = 0.9
# Validation sets
num_of_docs = len(corpus)
corpus_sets = corpus
corpus_title = '100% Corpus'
model_results = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []}

# Vamos a esperar mucho....
if 1 == 1:
    pbar = tqdm.tqdm()
    # Itera a lo largo del range de los topics
    for k in topics_range:
      # Calculamos coherencia para esos topics
      cv = calculo_valor_coherencia(corpus=corpus, dictionary=id2word,
                                    k=k, a=alpha, b=beta)
      # Guardamos los datos
      #model_results['Validation_Set'].append(corpus_title[i])
      model_results['Topics'].append(k)
      model_results['Alpha'].append(alpha)
      model_results['Beta'].append(beta)
      model_results['Coherence'].append(cv)

      pbar.update(1)
    pbar.close()


# In[ ]:


model_results


# Ploteamos el resultado y observamos que el máximo de cogerencia está en torno a 13 topics.

# In[ ]:


# Show graph
import matplotlib.pyplot as plt
plt.plot(topics_range, model_results["Coherence"])
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# Entrenamos el modelo con el número de topics anteriormente calculado:

# In[ ]:


lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                      id2word=id2word,
                                      num_topics=12, 
                                      random_state=100,
                                      chunksize=100,
                                      passes=10,
                                      alpha=0.01,
                                      eta=0.9)


# In[ ]:


model_topics = lda_model.show_topics(formatted=False)
print(lda_model.print_topics(num_words=3))


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim 
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# Una vez lo hemos validado, podemos asociar topics a cada uno de los documentos.

# Utilizando el método get_document_topics() sobre el texto del corpus que queramos podemos detectar la composición de temas que tiene cada documento:

# In[ ]:


lda_model.get_document_topics(corpus[0])


# Podríamos ver la contribución de una palabra específica a cada uno de los topics con get_term_topics:

# In[ ]:


lda_model.get_term_topics("people", minimum_probability=0.0001) #Get the most relevant topics to the given word.

