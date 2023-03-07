#!/usr/bin/env python
# coding: utf-8

# 
# # Representación numérica de textos
# En la mayoría de técnicas comúnmente utilizadas en Text Mining y NLP es necesario representar los textos en forma de vectores numéricos para que puedan ser procesados por modelos estadísticos o modelos de aprendizaje automático. Existen diferentes formas de representar el texto de esta manera, todas de ellas consisten en mapear la información a un espacio vectorial, numérico, aunque difieren en la manera de modelar los datos.
# 
# Por una parte, nos encontramos estrategias para representar documentos completos en el espacio vectorial. Estas estrategias se basan en construir una matriz de términos y documentos, que representan el contenido del texto en vectores de manera que puedan ser utilizados como fuente de datos de entrada para tareas de clasificación, topic modelling o búsqueda de información. 
# 
# 
# Por otra parte, existen estrategias para representar los documentos a nivel de palabra. Es decir, que cada token del vocabulario tenga asociada un vector específico. Estas estrategias, comúnmente denominadas como embeddings, permiten capturar la semántica de los términos y realizar de forma muy competente tareas más complejas como la extracción y linkeo de entidades, aunque también pueden ser utilizadas para otro tipo de tareas. 
# 
# 
# 
# 

# 
# ## 1. Representación numérica de documentos
# 
# En este notebook vamos a aprender a representar numéricamente documentos a través de los métodos de representación Bag Of Words (BoW) y TF-IDF.
# 
# 
# Para este ejercicio trabajaremos con un corpus muy sencillo, para visualizar bien los resultados:

# In[ ]:


corpus = ["yo quiero agua",
          "yo quiero cocacola",
          "yo quiero agua y un agua",
          "yo no quiero vino",
          "yo quiero un entrecot"]


# Vamos a tokenizar cada una de las frases, para trabajar comodamente más adelante:

# In[ ]:


corpus_tok = [i.split() for i in corpus]  #Corpus conocido y sencillo, podemos dividir palabras con split()
corpus_tok


# ### Bag of Words
# Como hemos visto anterioremente, Bag Of Words es uno de los modelos de representación de texto más intuitivos. Para implementarlo, se construye un vocabulario de tamaño N con los tokens únicos del corpus de trabajo, para despues representar cada documento con un vector del mismo tamaño en el que cada elemento será el nº de veces que aparece el token en cuestión en el documento.
# 
# 

# En primer lugar, construimos el vocabulario que se utilizará para representar nuestro corpus:

# In[ ]:


# El vocabulario de nuestro corpus son los tokens únicos utilizados en nuestro conjunto de documentos
import itertools
import numpy as np

all_tokens = itertools.chain(*corpus_tok)
vocab = sorted(set(all_tokens))

print("Nuestro vocabulario contiene {} tokens. Que son: {}".format(len(vocab),vocab))


# Cada uno de los documentos del corpus estará representado por un vector de 9 elementos. El elemento 0 representará la presencia del token agua en el documento, el elemento 1 representará la presencia de "cocacola" en el documento... etc
# 
# A continuación, vamos a construir esos vectores:

# In[ ]:


import collections
# Creamos un vector de salida
doc_vectors = list()
# Iteramos en cada uno de los documentos del corpus
for doc in corpus_tok:
  # Contamos las ocurrencias de cada uno de los tokens. Counter devuelve un diccionario
  counter_doc = collections.Counter(doc)
  # Creamos el vector que representará al documento
  doc_vec = list()
  # Iteramos por los términos del voabulario
  for word in vocab:
    # Si el término está en el diccionario de conteo, introducimos el valor en
    # esa posición del vector. Si no está, introducimos un 0
    if word in counter_doc:
      doc_vec.append(counter_doc[word])
    else:
      doc_vec.append(0)
  # Añadimos al vector de salida el vector del documento
  doc_vectors.append(doc_vec)


# In[ ]:


for frase, vector in zip(corpus, doc_vectors):
  print("La frase '{}' está representada por el vector {}".format(frase, vector))


# En Bag Of Words: 
# 
# - Los términos que aparecen varias veces en el documento tienen un número mayor, lo que se puede traducir en que esos términos (caracteristicas) tienen un mayor peso en el documento.
# - Si un término ocurre muchas veces en todos los documentos, tendrá gran importancia en cada uno de ellos, a pesar de que no será útil para clasificar o agrupar textos.
# 
# 
# El método de representación TF-IDF intenta precisamente  compensar este efecto mediante la aplicación de una penalización a palabras comunes en muchos documentos.

# ### TF-IDF  (intuición)
# 

# Para realizar los cáculos de forma más eficiente generaremos un diccionario que asocie a cada una de las palabras de nuestro corpus un índice numérico.
# 

# Para calcular TF-IDF realizaremos los siguientes pasos:
# 
# 1. Calcular el Term Frequency (TF). Que se calcula como:
#   $$TF(w) = n_{w-in-d} / N_d$$
# 
# Anteriormente hemos calculado para cada documento el número de veces que aparecía cada token del vocabulario. Para calcular el TF bastaría con dividir cada uno de los elementos del vector con la suma total de tokens de cada documento.
# 

# 2. Calcular el componente Inverse-Document Frequency (IDF). Que se calcula como:
# $$ IDF(w) = ln(N/n_w)$$
# 
# donde n_w  es el número de documentos que contiene el token w y N es el número total de documentos. 
# En el código incluimos una versión simplificada, para simplificar el código, en el que n_w es el número de veces que el token aparece 
# 

# En primer lugar generamos una función que permita calcular el número de vees que un token aparece en un documento d enuestro corpus.

# Depsues, unimos ambas expresiones para calcular el TF-IDF de una frase:

# In[ ]:


def tf_idf_calc(doc, N_corpus):
  # Generamos vector salida
  tf_idf_vec = np.zeros((len(vocab),))
  for word in doc:
    # Calculamoes el valor tf para el token en cuestion
    tf = TF_calc(doc, word)
    # Calculamos el valor idf para el token en cuestion
    idf = idf_calc(doc, N_corpus)

    tf_idf = tf * idf
    tf_idf_vec[index_dict[word]] = tf_idf 
  return tf_idf_vec


# Por ultimo lo calculamos para todo el corpus

# In[ ]:


#TF-IDF Encoded text corpus
tf_idf_result = []
for sent in corpus_tok:
    vec = tf_idf_calc(sent,len(corpus_tok))
    tf_idf_result.append(vec)


# ### Cálculo con scikit-learn
# 
# 
# 

# Aunque hemos mostrado una manera de aplicar el BoW y Tf-IDF a un corpus de documentos. Existen varias modificaciones que pueden mejorar los resultados (modificaciones en el cálculo de la componente IDF, por ejemplo). Además, las librerías de ML están preapradas para transformar documentos de forma más inmedianta, eficiente e incorporando funciones que pueden ser de utilidad
# 
# 
# En scikit-learn podemos utilizar distintas funciones para obtener el vocabulario de un corpus de documentos. Ambas están presentes dentro del módulo feature_extraction.text y son [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) y  [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer). 
# Vamos a crear un objeto con cada una de esas clases para introducir nuestro corpus y extraer el vocabulario.

# 
# **Tanto las funciones CountVectorizer como TfidfVectorizer cuentan con muchos parámetros personalizables. Algunos de los más relevantes son:**
# 
# 
# *   *strip_accents*: Elimina los acentos en codificación ascii o unicode. Por defecto es None. Es preferible hacer una gestión de acentos previas.
# *   *lowercase*: Transforma todos los caracteres a minúsculas antes de hacer la tokenización.
# *   *tokenizer*: Utiliza un tokenizador específico. Se puede utilizar una de NLTK o de Spacy(computacionalmente menos eficiente).
# *   *stop_words*: Si se pone el valor "english" eliminara la lista de stop_words definida en scikit-learn. Se puede utilizar la lista de stopwords de otras librerías o definir unas propias.
# *   *ngram_range*: Cálculo de n-gramas en el proceso. Mediante la tubla (min_n, max_n) se pueden incorporar n-gramas al cálculo de la matriz tfidf.
# *   *max_df*: Valor por defecto 1.  Ignora los tokens (o n-gramas) que aparecen en más del X % de documentos cuando es menor de 1. Si max_df es mayor que uno se ignorarán los términos que aparecen en más de X documentos.
# *   *min_df*: Valor por defecto 1. Ignora los tokens que aparecen  en menos del X % de los documentos. Siendo X el valor de X. (0.01 = 1%, por ejemplo)
# *   *max_features*: Máximas características que devuelve la función TfidfVectorizer. Valor mayor que 1. Representa las caracaterísticas más importantes (las más repetidas o comunes) Esto es muy interesante para no sobreentrenar el sistema.
# *   *norm*: Valores "l1" y "l2", por defecto "l2". Normaliza los valores entre 0 y 1.
# *   *use_idf*: Habilita el uso del inverse-document frequency en la función. Por defecto es True.
# *   *smooth_idf*: Suaviza los pesos de IDF sumando una unidad a cada frecuencia. Es muy importante para evitar divisiones por cero.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import text

# Creamos los objetos
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False) # Re-escribimos los valores por defecto para tener el tf-idf básico

# Hacemos Fit con nuestro corpus
count_data = count_vectorizer.fit(corpus)
tfidf_data = tfidf_vectorizer.fit(corpus)


# In[ ]:


# Obtenemos los vocabularios de dos formas:
print("COUNT VECTORIZER")
print("Obtenemos un dictionario que mapea las palabras con su posición en el vector del vocabulario")
print(count_data.vocabulary_)
print("Obtenemos el vocabulario en si mismo como una lista")
print(count_data.get_feature_names())
print("Numero de características:")
print(len(count_data.get_feature_names()))


# In[ ]:


# Obviamente, dado que hemos utilizado el mismo corpus, obtenemos el mismo resultado con el tfidf_vectorizer.
print("\n\n TF-IDF VECTORIZER")
print("Obtenemos un dictionario que mapea las palabras con su posición en el vector del vocabulario")
print(tfidf_data.vocabulary_)
print("Obtenemos el vocabulario en si mismo como una lista")
print(tfidf_data.get_feature_names())
print("Numero de características:")
print(len(tfidf_data.get_feature_names()))


# Si imprimimos el objeto tfidf_data podemos ver la configuración de los parámetros de TfidfVectorizer:

# In[ ]:


tfidf_data


# A continuación se muestra el resultado de transformar nuestro corpus con lso métodos de scikit-learn. El `CountVectorizer` mostrará el conteo de veces que una palabra del vocabulario está presente dentro del documento, el `TfidfVectorizer` mostrará el resultado con la métrica TF-IDF mostrada en los apuntes.

# In[ ]:


# Resultado del CountVectorizer
count_data_result = count_data.transform(corpus).toarray()
print(count_data_result)


# In[ ]:


# Resultado del TfidfVectorizer
tfidf_data_result = tfidf_data.transform(corpus).toarray()
print(tfidf_data_result)


# Vamos a mostrar los resultados con seaborn para que se vean mejor. Importante mencionar que esto se puede hacer cuando el vocabulario es muy reducido, si no podría ocasionar problemas en la memoria.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16, 6))
# Figura CountVectorizer
sns.heatmap(count_data_result, annot=True,cbar=False,
            xticklabels=count_data.get_feature_names(),
            yticklabels = ["Frase 1", "Frase 2", "Frase 3","Frase 4", "Frase 5"])


# In[ ]:


plt.figure(figsize=(16, 6))
sns.heatmap(tfidf_data_result, annot=True,cbar=False,
            xticklabels=tfidf_data.get_feature_names(),
            yticklabels = ["Frase 1", "Frase 2", "Frase 3","Frase 4", "Frase 5"])


# ### Uso de preprocesadores externos a scikit-learn
# 
# 
# 

# En primer lugar, dado que vamos a utilizar spacy, instalaremos la librería y el modelo:

# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download es_core_news_sm')


# Importamos las librerías y creamos el objeto "nlp" para procesar los textos
# 
# 
# 

# In[ ]:


import spacy
import es_core_news_sm
nlp = es_core_news_sm.load()


# Guardamos las stop words de Spacy en una variable llamada "stop_words". También cogemos los tokens considerados símbolos de punctuación en la variable "punctuations":

# In[ ]:


import string
spacy_stopwords = spacy.lang.es.stop_words.STOP_WORDS
stop_words = spacy_stopwords
punctuations=string.punctuation


# Generamos una función "spacy_tokenizer" que:

# In[ ]:


def spacy_tokenizer(sentence):
    # Pasamos la frase por el objeto nlp para procesarla
    mytokens = nlp(sentence)

    # Lematizamos los tokens y los convertimos  a minusculas
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Quitamos las stopwords y los signos de puntuacion
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # devolver una lsita de tokens
    return mytokens


# Utilizamos esa función como tokenizador en TfidfVectorizer:
# 

# In[ ]:


tfidf_vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, tokenizer = spacy_tokenizer) # Re-escribimos los valores por defecto para tener el tf-idf básico
tfidf_data = tfidf_vectorizer.fit(corpus)


# In[ ]:


print("\n\n TF-IDF VECTORIZER")
print("Obtenemos un dictionario que mapea las palabras con su posición en el vector del vocabulario")
print(tfidf_data.vocabulary_)
print("Obtenemos el vocabulario en si mismo como una lista")
print(tfidf_data.get_feature_names())
print("Numero de características:")
print(len(tfidf_data.get_feature_names()))


# **Ejercicio**: 
# 
# Queremos transformar nuestro dataset de noticias con las siguientes especificaciones:
#  - Transformar los primeros 3000 documentos 
#  - Se utilice idf y el valor de norm por defecto.
#  - Se utilice la función de preprocesado de spacy utilizada anteriormente
#  - Se consideren unigramas, bigramas, trigramas
#  - Se consideren un máximo de 250 características (max_features)
#  - El vectorizador no debe considerar los elementos que aparezcan en menos del 5% de documentos.

# In[ ]:


# Dataset de noticias
get_ipython().system('wget "https://github.com/luisgasco/ntic_master_datos/raw/main/datasets/news_summary.csv"')


# In[ ]:


import pandas as pd
news_summary = pd.read_csv('../content/news_summary.csv', encoding='latin-1')
news_subset = news_summary["text"].to_list()[0:3000]


# In[ ]:


def spacy_tokenizer(sentence):
    # Pasamos la frase por el objeto nlp para procesarla
    mytokens = nlp(sentence)

    # Lematizamos los tokens y los convertimos  a minusculas
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Quitamos las stopwords y los signos de puntuacion
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Quitamos números:
    mytokens = ['' if word.isdigit() else word for word in mytokens  ]
    # devolver una lsita de tokens
    return mytokens


# In[ ]:


tfidf_vect = TfidfVectorizer(lowercase = False, stop_words = None, 
                             ngram_range = (1,3), use_idf = True, smooth_idf = True,
                             norm = "l2",max_features=250, min_df=0.05, tokenizer=spacy_tokenizer)
tfidf_data = tfidf_vectorizer.fit(news_subset)


# In[ ]:


tfidf_data.get_feature_names()


# ### Diferencias entre BoW y TF-IDF
# 

# Hasta el momento hemos visto diferente maneras de realizar la representación de documentos utilizando el método BoW y TF-IDF. 
# Sin embargo, ¿Qué implica realmente utilizar un método u otro? 
# 
# La clave de TF-IDF es su penalización de palabras comunes en todos los documentos. Para ver su efecto, en este apartado vamos a:
# 1. Descargar un conjunto de documentos de wikipedia
# 2. Preprocesarlos
# 3. Mostrar los 10 términos más importantes para BoW y Tf-IDF, y ver las diferencias

# En primer lugar importanmos y descargamos las librerías que utilizaremos
# 

# In[ ]:


get_ipython().system('pip install wikipedia')
import wikipedia
import spacy
import collections
import math
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import text
import string
import numpy as np
import re

nlp = spacy.load('en')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
stop_words = spacy_stopwords
punctuations=string.punctuation
RE_WHITESPACE = re.compile(r"\s+")


# Definimos una función de procesado similar a la anteriormente definida. En este caso eliminarmos los números (no los sustitumos por un caracter vacío). Además, eliminamos posibles espacios extras despues del proceso de tokenización

# In[ ]:


def spacy_tokenizer(sentence):
    # Pasamos la frase por el objeto nlp para procesarla
    mytokens = nlp(sentence)

    mytokens = [ word.text.lower()   if word.pos_ == 'PRON' or word.lemma_ == '-PRON-' else word.lemma_.lower() for word in mytokens]
  
    # Lematizamos los tokens y los convertimos  a minusculas
 #   mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Quitamos las stopwords y los signos de puntuacion
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # Quitamos números:
    mytokens = [word  for word in mytokens if not word.isdigit()]
    # Remove extra spaces
    mytokens = [token.strip() for token in mytokens]
    # devolver una lsita de tokens
    return mytokens


# Vamos a descargar el texto de descripción de varios guitarristas históricos. Para ello definiremos una lista de strings con sus nombres y los obtendremos utilizando la función "page" de la librería wikipedia. 
# 

# In[ ]:


pages =  [
  "Brian May","jimmy page","eddie van halen","david gilmour",
  "Jeff Beck","mark knopfler","Billy Gibbons","Carlos Santana", 
  "Stevie Ray Vaughan","BB king","Buddy Guy","Albert King","Rory Gallagher",
  "Joe Satriani", "jimi hendrix","George Harrison"
    ]
documentes = [RE_WHITESPACE.sub(" ",wikipedia.page(page, auto_suggest=False).content).strip() for page in pages]


# Calculamos y entrenamos vectorizadores TFIDF y BoW (CountVectorizer) para el corpus descargado y preprocesado:

# In[ ]:


tfidf_vect = TfidfVectorizer(lowercase = False, stop_words = None, use_idf = True, smooth_idf = False,
                             norm = "l2", tokenizer=spacy_tokenizer)
count_vect = CountVectorizer(lowercase = False, stop_words = None, tokenizer=spacy_tokenizer)

tfidf_data = tfidf_vect.fit(documentes)
count_data = count_vect.fit(documentes)


# Generamos funciones para obtener las palabras más importantes del vectorizador:
# 

# In[ ]:


tfidf_feature_names = np.array(tfidf_data.get_feature_names())
count_feature_names = np.array(count_data.get_feature_names())

def get_top_vect_words(response, top_n=2,feature_name_array=tfidf_feature_names):
  # De la respuesta del vectorizador, cogemos los datos en bruto del array, los ordenamos
  # de mayor a menor y cogemos los índices de los top_n terminos. Que se seleccionarán del 
  # los nombres de las caracteristicas (tokens) del vectorizador.
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_name_array[response.indices[sorted_nzs]]
  


# Generamos está función para mostrar los 10 términos más importantes de un documento vectorizado con TF-IDF y BoW para ver sus diferencias. 

# In[ ]:


def show_doc_results(doc,top_n):
  responses_tfidf = tfidf_vect.transform([doc])
  responses_count = count_vect.transform([doc])
  most_frequent = list(get_top_vect_words(responses_count,top_n,count_feature_names))
  tfidf = list(get_top_vect_words(responses_tfidf,top_n,tfidf_feature_names))
  print("Tokens más frecuentes: {}".format(most_frequent))
  print("Tokens con mayor TF-IDF: {}".format(tfidf))


# In[ ]:


for i in range(0, len(pages)):
  print(pages[i])
  show_doc_results(documentes[i],10)


# Observar diferencias de palabras muy comunes en documentos de guitarristas como "guitar", "release","play" o "album". En BoW aparecen como términos muy importantes porque son comunes en este tipo de documentos. Sin embargo, al calcular el TF-IDF, como estos términos aparecen en muchos documentos su importancia se ve disminuida.
# 

# # Representación numérica de palabras
# 

# También se puede hacer una representación de un documento obteniendo una representación de los vectores de las palabras que lo componen, y haciendo algún tipo de combinación sobre éstas (suma, concatenación..).
# 
# En este caso presentaremos el One Hot Encoding y los word embeddings.

# ## One-hot encoder
# 

# Probablemente sea la técnica más elemental para representar datos textuales numéricamente. En este modelo de representación cada palabra del vocabulario se representa mediante un vector único en el que cada posición del vector representa una palabra del vocabulario, siendo 0 todos los valores menos el índice del token al que representa el vector. 
# 
# En este caso, trabajaremos con el método [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) de Scikit-learn.

# En primer lugar, es necesario obtener un índice único para cada una de las palabras de nuestro vocabulario. 
# 
# Podemos hacerlo de forma manual o utilizando un LabelEncoder de scikit-learn:

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(vocab)

for elem, index in zip(vocab, integer_encoded): 
  print("Al token `{}` del vocabulario se le ha asignado el índice {}".format(elem,  index))


# Una vez asignados estos índices, los utilizaremos para crear un vector único de tamaño del vocabulario. 
# 
# Para ello, utilizarmeos el método OneHotEncoder:

# In[ ]:


onehot_encoder = OneHotEncoder(sparse=False)


# Cambiamos las dimensiones del vector de índices, y los transformamos utilizando el onehot_encoder:

# In[ ]:


integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

for elem, index in zip(vocab, onehot_encoded): 
  print("Al token `{}` del vocabulario se le ha asignado el onehot vector: {}".format(elem,  index))


# Después de definir esto podemos generar funciones para transformar un conjunto de tokens a su versión one-hot. Así como el paso inverso. 

# In[ ]:


def using_to_categorical(doc,label_encoder, onehot_encoder):
    # Transformamos los tokens a índices
    data = label_encoder.transform(doc)
    # Adaptamos los indices al formato adecuado
    data = data.reshape(len(data), 1)
    # Les aplicamos one hot encoder
    encoded = onehot_encoder.transform(data)
    return encoded


def invert_encoding(vectores_onehot,label_encoder):
    inverted = label_encoder.inverse_transform(list(np.argmax(vectores_onehot,axis=1)))
    return list(inverted)


# Utilicemos las funciones:

# In[ ]:


documento_ejemplo = "yo quiero agua".split()
documento_ejemplo


# In[ ]:


a_hot_enc=using_to_categorical(documento_ejemplo,label_encoder, onehot_encoder)
a_hot_enc


# In[ ]:


invert_encoding(a_hot_enc,label_encoder)


# ## Embeddings

# Los embeddings son una técnica de modelización de lenguaje que permite representar palabras mediante vectores densos (*dense vectors*), de menor tamaño que los obtenidos tras _one-hot enconding_, y que además tienen capacidad de representar ontenido semántico. 
# 
# A lo largo del tiempo han aparecido diferentes formas de obtener estos embeddings. Pero en este curso aprendermeos a utilizar y a entrenar embeddings estáticos, especificamente word2vec, utilizando la librería [gensim](https://radimrehurek.com/gensim/models/word2vec.html). Hay que mencionar que los actuales embeddings contextuales, como Elmo o BERT, serán utilizados en el futuro para poder mostrar su uso, que en esencia es similar al del embedding estático. 
# 
# ![image.png](data:image/png;base64,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kTCeSSRaLQwBo1gcYvJBgYrJNImCsUwiYWGIRMFEmkTCeCaRaLkwBIxScYjJBwUqJtMkCsYyiYSFIRIFE2kSCeOZRKL1whAAjLPb7XZfHwQREREREREREfnGmN3KnoiIiIiIiIiI/A+LQ0REREREREREGsbiEBERERERERGRhrE4RERERERERESkYSwOERERERERERFpGItDREREREREREQaxuIQEREREREREZGGsThERERERERERKRhLA4REREREREREWkYi0NERERERERERBrG4hARERERERERkYaxOEREREREREREpGEsDhERERERERERaRiLQ0REREREREREGsbiEBERERERERGRhrE4NIZ27dqF5cuXY9y4cRg3bhzMZjM2bdoEm82m+vyqqiqsXLkSRqMR48aNg9FoxMqVK1FVVTXo+xiNRhiNRrfPqaqqwvr162E2m+XjWb58OY4cOTKsz0jaMJT4OXLkiFfxDwA2mw1Lly7FuHHjsH37do+O7ciRI/J3Zvny5V5/NtIeb8dmm82GTZs2eT1+ejI2S/qP/+PGjcPSpUs5RtOAbDYbtm/fjnnz5slxM2/ePOzatcvta8ZibN61a5fimMxmM1auXDmkz0jaceTIEa/Og4c6NpeVlcmvKSsrc3l8165d8v4G+kPkzlDG5qHmgevXrx/0HFjtuzJv3jxs2rRpyJ+RBGGnMbFu3To7ADsAe0FBgT0nJ0fx3/2VlpbaDQaDHYC9sLDQXlRUZC8oKLADsBsMBrvVanV5jdVqtVssFnm/7v7vraysdLtvAPbi4uIR//wkDuf4SUtLsxcUFMj/7S5+du7cKcfuunXr7EVFRfa0tDS38W+32+379u1T7LeoqMij4xvsu0XkzNux2Wq1ys8pKCiwFxUV2QsLCweMf0/HZolz7Etj9Lp16+w5OTkefw9Im6TYNBgM9oKCAnmcdTeGjsXYLJ1fpKWl2YuKiuxFRUV2i8ViNxgMI/KZSUz79u2TYywnJ0dxnqp2HjyUsdlut9uLiooUY7Pa84qLi+0FBQVu/3gyrpO2eTs2DyUPLC0t9fgcuP93Zd26dfIxrVu3bkQ/OwUWjmRjxGKx2IuKihRfZumkDIC9tLRU8XxpANi3b59iu5TIFBYWKraXlpYqvtQD/VCVlpbaCwoK7JWVlar7tlgsw/moJDjpx8f5BMr5pKx/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2+3DywMHOweWCq5qsS5ddGV8a9c4u91u977fiLxlNBrR0NCAtLQ0mM1m1eccOHBA/vv+/ftx3333wWAwYPny5ZgyZQree+89HDx4EDk5OTh+/LjitWVlZfJtaQHH3UEAoKioCIBjoT+pzXH79u146qmnkJaWBovFgpiYGFy6dAlbt24F4Fg80tuFWEk7pPgBoLpOEQB861vfUsTQ+vXr5YUkpZh79dVXUVlZiXXr1uGnP/2py3tIbeBS3BcUFOCuu+4C4Fi3YKB5/keOHEF+fj4KCgoU3yui/rwdm6uqqpCTk4OGhgZYLBZkZ2fLYPxR3QAAAnxJREFU46e0ZptzbHozNvfff2FhIaZMmTLg/okkZWVlmD17NgDHdDC1aTpz585VjLejPTY77/+JJ54AgAH3TyRZvnw5du/eDYPB4PbOeUVFRfI0M2/HZpvNplgQXYpLadwFHNOGnFVVVclTiKxWK+9SRh4ZytjsbR64f/9+nDlzBgDkuHced2+//Xb5jr7SNHoA8ncFAP70pz/hxIkTyMnJGXCqJwnO19UprZCqtAP96a+4uFi+koEbV9+KiopUK87SVWV3f/q3H+7cuVOxb4PBYLdYLLyKR4MaLNbg5krytm3b5KuAuHH1xN3VwMG+L4NdRZS64DidjAYzlLG5srLSbrFY5NcaDAZ7YWGhaselt2Oz3e6IX+cpk9L+OZ2MBuLc/evuj9qV5NEem4uKiuSr4NL+9+3bN+Kfn8TiybTx/ues3ozNnnxf+r9OGs85HZK8MdSx2Zs8cLDvS/9O/+LiYsV3RRqb3e2ftIOdQ0REREREREREGsa7lRERERERERERaRiLQ0REREREREREGsbiEBERERERERGRhrE4RERERERERESkYSwOERERERERERFpGItDREREREREREQaxuIQEREREREREZGGsThERERERERERKRhLA4REREREREREWkYi0NERERERERERBrG4hARERERERERkYaxOEREREREREREpGEsDhERERERERERaRiLQ0REREREREREGsbiEBERERERERGRhrE4RERERERERESkYSwOERERERERERFpGItDREREREREREQaxuIQEREREREREZGGsThERERERERERKRhLA4REREREREREWkYi0NERERERERERBr2/wBLYK2u7zHq/gAAAABJRU5ErkJggg==)

# ### Carga de embeddings pre-entrenados con ``gensim``
# 
# 
# 
# 
# 

# Los embeddings suelen ser entrenados con grandes conjuntos de datos y suele ser costoso su entrenamiento a nivel temporal. Por ese motivo, una de las posibilidades cuando se trabaja con embeddings, es la búsqueda de recursos ya entrenados con textos similares a los que vayamos a trabajar (como contenido de noticias o redes sociales), y utilizarlo para representar nuestros documentos. 
# 
# 
# Cuando se busca este tipo de recurso es evidente la influencia anglosajona en el desarrollo de las tecnologías del lenguaje, ya que la mayoría de los recursos se encuentran en este idioma y son fácilmente aplicables desde las librerías de NLP más comunes, en este caso Gensim. Si se quieren utilizar recursos en otros idiomas, hay que buscarlos en repositorios externos para descargarlos manualmente y cargarlos en la librería en cuestión. 
# 
# Aunque en este notebook trabajaremos exclusivamente con Gensim, este problema de acceso a recursos se está viendo parcialmente solventado con el incremento de popularidad de librerías como [Hugginface](https://huggingface.co/models) que favorecen el intercambio de recursos en múltiples lenguas
# 

# #### Carga de pre-trained embeddings de gensim
# 

# La librería Gensim cuenta con un conjunto de embeddings pre-entrenados accesibles desde la propia librería mediante una API.
# 
# Para acceder a estos modelos basta con utilizar el comando ```gensim.downloader.info()```, que nos permitirá ver la lista de modelos disponibles:

# In[ ]:


import gensim.downloader as api
print("Este es el listado de modelos disponibles en Gensim: \n {}".format(list(api.info()["models"].keys())))


# Se puede observar que no sólo hay embeddings word2vec. También los hay glove, fasttext y conceptnet, todos estáticos. En este caso vamos a cargar un embedding Glove por ocupar menos espacio en memoria. 

# In[ ]:


glove_emb = api.load('glove-twitter-25')


# Una vez hayamos cargado el embedding, podemos aplicar diferentes funciones de gensim para obtener información útil:
# 
# - Listado de palabras más similares a una dada
# 

# In[ ]:


glove_emb.most_similar("noise", topn=3)


# -  Operación matemática con vectores para obtener resultados

# In[ ]:


glove_emb.most_similar(positive=["france","italy"],negative=["rome"])


# -  Detectar término "intruso"

# In[ ]:


glove_emb.doesnt_match(['spain', 'italy', 'paris'])


# -  Obtener el vector que representa a una palabra para utilizarlo posteriormente en otras tareas

# In[ ]:


glove_emb.get_vector("madrid")


# #### Carga de pre-trained embedding de otras fuentes
# 

# Como hemos visto anteriormente, no siempre se encuentran embeddings en el idioma en el que estamos trabajando. Aunque hoy en día existen mejores opciones para representar palabras en diferentes idiomas  que el uso de Word2Vec (como el [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/1)]), en este apartado vamos a cargar un [Word2Vec embedding en español](https://github.com/aitoralmeida/spanish_word2vec)l creado con datos de noticias, Wilipedia y el BOE entre otros y no disponible de forma nativa en la librería de Gensim.
# 

# En primer lugar descargamos y hacemos unzip del modelo:

# In[ ]:


get_ipython().system('wget https://zenodo.org/record/1410403/files/keyed_vectors.zip?download=1')
get_ipython().system('unzip /content/keyed_vectors.zip?download=1')


# Una vez descargado y descomprimido, cargamos los ``KeyedVectors``, una manera de almacenar un embedding como un diccionario que permite ahorrar espacio de almacenamiento y memoria.
# 

# In[ ]:


from gensim.models import KeyedVectors


# In[ ]:


word_vectors = KeyedVectors.load('complete.kv', mmap='r')


# Una vez cargado, podemos aplicar las funciones del embedding disponibles en Gensim, además de obtener la representación vectorial de cada palabra.

# In[ ]:


word_vectors.most_similar(positive=["mujer","rey"],negative=["hombre"])


# In[ ]:


word_vectors.similar_by_word("virus")


# In[ ]:


word_vectors.get_vector("hola")


# ### Entrenamiento de nuestro propio embedding

# Los embeddings generalistas son populares para trabajar con cualquier tipo de datos. Sin embargo, es posible que en aplicaciones específicas no funcionen de forma óptima. 
# 
# Hoy en día se pueden construir embeddings contextuales que tienen un mejor funcionamiento, pero requieren una mayor capacidad de cómputo y recursos. 
# 
# En este subapartado vamos a ver como entrenar un embedding con un conjunto de datos público y vamos a comprobar su funcionamiento. 

# En primer lugar descargamos y descomprimimos el conjunto de datos de entrenamiento de la [Tarea MESINESP (MEdical Semantic Indexing in Spanish)](https://temu.bsc.es/mesinesp2/). Este conjunto de datos está compuesto por un conjunto de abstracts de artículos científicos extraidos de repositorios en español. 

# In[ ]:


import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
from gensim.test.utils import datapath
import re
import unicodedata
from tqdm import tqdm
import gensim
import multiprocessing
import random
from gensim.models.phrases import Phrases, Phraser

nltk.download('stopwords')

get_ipython().system('wget https://zenodo.org/record/5602914/files/Subtrack1-Scientific_Literature.zip?download=1')
get_ipython().system('unzip /content/Subtrack1-Scientific_Literature.zip?download=1')


# Después cargamos el archivo _*.json_ en un diccionario, e iteramos por cada elemento de la lista cogiendo el campo "abstractText", que contiene el abstract de cada artículo.
# 
# Para acelerar el proceso de cálculo, solo trabajaremos con los primeros 2500 artículos (el corpus tiene más de 200000)

# In[ ]:


import json
with open("/content/Subtrack1-Scientific_Literature/Train/training_set_subtrack1_only_articles.json", 'r') as f:
  biomed_dict = json.load(f)

documentos = [doc["abstractText"] for doc in biomed_dict["articles"]][:2500]


# Creamos la función ``clean_data`` que realizará un preprocesado a un documento de entrada: pasará a minúsculas, sustituirá dígitos por espacios, tokenizará, eliminará stopwords y quitará tokens menores o iguales de 2 caracteres.

# In[ ]:


stopwords_list=stopwords.words('spanish')
def clean_data(w):
    w = w.lower()
    w=re.sub(r'[^\w\s]','',w)
    w=re.sub(r"([0-9])", r" ",w)
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)


# Creamos una función generator para trabajar más eficientemente en memoria

# In[ ]:


def get_inp(documentos):
    sent=list(map(clean_data,documentos))
    for lines in tqdm(sent):
        yield lines.split()


# Aplicamos función sobre documentos:

# In[ ]:


sent = [row for row in get_inp(documentos)]


# Entrenamos el modelo [Phrases](https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.FrozenPhrases) de gensim para detectar bigramas presentes en nuestros datos. En este caso se considerarán conjutos de tokens que al menos aparezcan 20 veces en nuestro corpus. 
# 
# Además, utilizamos el método Phraser para mejorar el rendimiento de Gensim en pasos posteriores. 

# In[ ]:


phrases = Phraser(Phrases(sent, min_count=20, progress_per=10000))


# Aplicamos el modelo a nuestro conjunto de datos:

# In[ ]:


sentences = phrases[sent]


# En el documento del índice 3, se puede observar que se ha aplicado con éxito. El término "atención_primaria" es la unión de los tokens "atención" y "primaria" que aparecía de forma continua en al menos 20 ocasiones dentro del corpus

# In[ ]:


sentences[3]


# Una vez preparado el corpus, vamos a proceder con el entrenamiento del modelo [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec). Para ello utilizaremos las capacidades multiprocesador de Gensim y utilizaremos unos parámetros típicos en este tipo de modelo.
# 
# En la estrategia de entrenamiento CBOW especificada en el paper de Word2Vec, se toman ventanas de tamaño N de cada documento. Entonces, se extrae la palabra central del enventanado y se entrena un modelo capaz de predecir esa palabra central a partir del resto (el contexto).
# 
# El término "window" indica el tamaño de la ventana elegido para subdividir cada uno de nuestros documentos. La función se encarga de transformar todo el corpus (train/test, transformación de las palabras en vectores, backprogataion...), pero al final internamente se tendrá el siguiente formato de dato:
# 
# 
# Palabras de contexto | Word vector dle contexto | Palabra central | vector de la palabra central|
# -----|----------|----------|----------------------|
# yo quiero y un agua | media de los one-hot vectors |agua |[ 0 0 0 .... 1.... 0 |

# In[ ]:


import multiprocessing
from gensim.models import Word2Vec
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=20,
                     window=5,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20, 
                     workers=cores-1)


# Una vez creado el objeto Word2Vec, procedemos a generar un vocabulario para el embedding, que será extraido a partir de nuestro corpus.

# In[ ]:


w2v_model.build_vocab(sentences, progress_per=10000)


# Por último se procede a entrenar el embedding, para posteriormente guardarlo (y cargarlo para verificar que se ha aplicado correctamete)

# In[ ]:


w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)


# In[ ]:


w2v_model.wv.save_word2vec_format("guardar.bin",binary=True)
saved_model_path='/content/guardar.bin'
trained_model = KeyedVectors.load_word2vec_format(saved_model_path, binary=True)


# Procedemos a ver el funcionamiento del embedding buscando sinónimos a algunas palabras de ejemplo presentes en el corpus.

# In[ ]:


trained_model.wv.most_similar(positive=["virus"], topn=3)


# In[ ]:


trained_model.wv.most_similar(positive=["vph"], topn=10)


# In[ ]:


print(trained_model.similarity('virus', 'covid'))


# Otra opción es visualizar el embedding, para ver si palabras similares están cercanas entre si. Para ello, vamos a extraer el vocabulario del modelo e introducir en una matriz X los vectores de todas las palabras

# In[ ]:


vocab = list(trained_model.wv.vocab.keys())
X = trained_model.wv[vocab]


# Hacemos una reducciónn de dimensionalidad con t-sne a dos dimensiones, para poder mostrar cada posición en un espacio 2D.

# In[ ]:


from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
# Transformamos a dataframe para facilitar su plotting
import pandas as pd
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])


# Cogemos una muestra, para visualizarlo de forma más clara.

# In[ ]:


df_2 = df.sample(200)


# In[ ]:


fig = plt.figure(figsize=(24,16))

ax = fig.add_subplot(1, 1, 1)

ax.scatter(df_2['x'], df_2['y'])

for word, pos in df_2.iterrows():
    ax.annotate(word, pos)

