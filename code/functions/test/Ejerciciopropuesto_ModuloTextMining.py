#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Descarga de archivos de ProfNER
get_ipython().system('wget https://zenodo.org/record/4563995/files/profner.zip?download=1')
# Si el unzip no funciona, revisar cual es el nombre de descarga del archivo
get_ipython().system('unzip  profner.zip?download=1')


# Funciones de carga y preparación de datos

# In[1]:


import sys
import os
import pandas as pd
def get_tweet_content(list_paths):
  """
    Función para guardar en un diccionario el contenido de archivos txt 
    que se introduce en su entrada. 
    Devuelve un diccionario en el que las claves son el id del tweet, y
    el valor el texto del tweet.
  """
  output_dict = dict()
  #quitar el print
  print(list_paths)
  for i in list_paths:
    tweet_id = i.split("/")[-1].split(".txt")[0]
    #quitar el print
    print(tweet_id)
    with open(i) as f:
      output_dict[int(tweet_id)] = f.read()
  
  return output_dict

def get_profner_data(profner_path_data):
    # Obtenemos el path a los txt de los tweets.
    path_to_txt = profner_path_data+"subtask-1/train-valid-txt-files/"
    tweets_train_files = [path_to_txt+"train/"+i for i in os.listdir(path_to_txt+"train/")]
    tweets_valid_files = [path_to_txt+"valid/"+i for i in os.listdir(path_to_txt+"valid/")]
    # Obtenemos diccionarios en los que el key es el tweet_id y el value el texto del tweet.
    train_txt_content = get_tweet_content(tweets_train_files)
    valid_txt_content = get_tweet_content(tweets_valid_files)

    # Cargamos dos dataframes con los tweet_id y la categoría de los tweets
    path_to_labeled = profner_path_data+"subtask-1/"
    train_tweets = pd.read_csv(path_to_labeled+"train.tsv",sep="\t")
    valid_tweets = pd.read_csv(path_to_labeled+"valid.tsv",sep="\t")

    # Introducimos a los df el campo de texto mapeando los diccionarios con tweet_id
    train_tweets["tweet_text"] = train_tweets['tweet_id'].map(train_txt_content)
    train_tweets["set"] = "train"
    valid_tweets["tweet_text"] = valid_tweets['tweet_id'].map(valid_txt_content)
    valid_tweets["set"] = "valid"

    # Concatenamos el resultado
    output_df = pd.concat([train_tweets,valid_tweets],axis=0)
    # Eliminamos retorno de carro
    output_df["tweet_text"] = output_df.tweet_text.apply(lambda x: x.replace('\n', ' '))
    return output_df[["tweet_id","tweet_text","label","set"]].reset_index(drop=True)


# In[2]:


#my_dict = get_tweet_content('profner/subtask-1/test-background-txt-files/1263117343498080257.txt')
#my_dict
os.getcwd()


# In[3]:


my_dict = get_tweet_content(['/content/profner/subtask-1/test-background-txt-files/1263117343498080257.txt'])
my_dict


# In[4]:


get_profner_data('/content/profner/')


# # Ejercicio
# 

# En este ejercicio se trabajará con un conjunto de datos reales publicados para la shared-task [ProfNER](https://temu.bsc.es/smm4h-spanish/), celebrada en el año 2021. Específicamente, se utilizarán los datos textuales de la subtarea 1, centrada en la clasificación de textos. Este conjunto de datos son tweets en español que tienen asignada una etiqueta numérica, que representa la presencia (valor 1) o no (valor 0) de menciones de profesiones en el tweet. Por si fuera de tu interés, el proceso de obtención, selección y anotación de datos está descrita en [este enlace.](https://temu.bsc.es/smm4h-spanish/?p=4003).
# 
# Para el ejercicio debéis entrenar diferentes modelos de clasificación que permitan clasificar correctamente los tweets. Para ello será necesario crear y utilizar funciones de preprocesado de datos similares a las vistas en clase, aplicar estrategias de vectorización de trextos como TF-IDF o embeddings, y entrenar/evaluar modelos de clasificación. Para que os sirva de orientación, los criterios de evaluación del ejercicio serán los siguientes:
# 
# -	**Análisis exploratorio, pre-procesado y normalización de los datos (30%)**:
#         -	El ejercicio deberá contener un análisis exploratorio de los datos como número de documentos, gráficas de distribución de longitudes y/o wordclouds, entre otros análisis que se os pudieran ocurrir. Vuestros ejercicios deberán incorporar al menos los análisis exploratorios vistos en clase.
# 
#     -	También tendréis que tener funciones para normalizar textos que permitan eliminar palabras vacías, quitar símbolos de puntuación y lematizar o hacer stemming.  
# 
# -	**Vectorización de textos (40%)**
# 
#     En clase hemos visto diferentes estrategias de vectorización como TF-IDF y Word Embeddings. También hemos visto como incorporar características adicionales utilizando el sentimiento de los documentos. Para este ejercicio sois libres de utilizar la estrategia de vectorización que queráis, pero:
#   -	Si decidís utilizar TF-IDF será necesarios que incorporéis a modelo características adicionales de sentimiento utilizando recursos adicionales (como por ejemplo la librería TextBlob). 
#   -	Si optáis por representar el texto mediante embeddings, dado que en clase no se profundizado sobre el tema no será necesario incorporar esas características adicionales. Si decidís esta segunda opción, podéis utilizar los embeddings en español que vimos en clase
# 
# -	**Entrenamiento y validación del sistema (30%)**
#   -	En el proceso de entrenamiento del modelo tendréis que testear al menos 3 modelos de clasificación. El procedimiento debe ser similar al visto en clase, en el que primero estimábamos el rendimiento de varios algoritmos de forma general, para posteriormente seleccionar el mejor para ajustar los hiperparámetros.
# 

# ## 0. Imports
# 

# In[5]:


get_ipython().system('python -m spacy download es_core_news_sm')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import es_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report


# ## 1. Obtención del corpus
# Para la obtención de los datos teneis disponible la función `get_profner_data()`. Esta función prepara los datos del ejercicio en formato Pandas dataframe para que podais realizarlo.

# In[6]:


profner_path = "./profner/"
datos_ejercicio = get_profner_data(profner_path)


# In[7]:


datos_ejercicio.head(4)


# ## 2. Análisis exploratorio

# En este apartado se realiza un análisis de los datos previo a la normalización de los mismos para extraer información relevante del dataset.

# * **Número de documentos y columnas:**
# 
# Comenzamos mostrando el número de documentos del data frame:

# In[8]:


print("Tenemos un conjunto de {} documentos".format(len(datos_ejercicio)))
print("El dataframe tiene {} columnas".format(datos_ejercicio.shape[1]))


# * **Número de documentos duplicados:**

# Comprobamos y eliminamos las filas con algún valor vacío (NA) y quitaremos los duplicados. En este caso, no hay duplicados que eliminar.

# In[9]:


print("Existen {} tweets duplicados".format(np.sum(datos_ejercicio.duplicated(subset=["tweet_text"]))))
# Quitaremos esos duplicados
#datos_ejercicio = datos_ejercicio.drop_duplicates()
#print("Despues de quitar duplicados tenemos un conjunto de {} tweets".format(datos_ejercicio.shape[0]))


# Comprobaramos que no hayan quedado Nulls en ningunas de las dos columnas del dataset.

# In[10]:


print("Hay {} valores vacíos en los tweets y {} valores vacíos en las etiquetas en los datos".format(np.sum(datos_ejercicio.isnull())[1],
                                                                                                       np.sum(datos_ejercicio.isnull())[2]))


# * **Quitamos columnas no necesarias:**
# 
# Eliminamos la columna "set" del dataframe porque no aporta información relevante.

# In[11]:


datos_ejercicio = datos_ejercicio.drop(['set'], axis=1)
datos_ejercicio.columns


# * **Número de documentos por cada clase:**
# 
# Contamos el número de elementos de cada clase que están presentes en la columna "label". En este caso nos encontramos dos tipos de documentos (tweets):
# 
# - "Label = 1": Tweets contienen referencia a una profesión
# - "Label = 0": Tweets no contienen referencia a una profesión 

# Comprobemos la distribución de las clases.

# In[12]:


datos_ejercicio["label"].value_counts()


# El dataset no está balanceado. Posteriormente, en la sección de los modelos, balancearemos para obtener resultados más acordes.
# 

# * **Distribución de la longitud de los tweets en caracteres:**
# 
# Obtenemos la longitud de cada uno de los textos de los documentos para despues dibujar su histograma. 

# In[13]:


datos_ejercicio["char_len"] = datos_ejercicio["tweet_text"].apply(lambda x: len(x)) #tweet_text
datos_ejercicio.head(4)


# In[14]:


# Importamos las librerías matplotlib y seaborn:
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(14,12))
sns.set_style("darkgrid")
# añadimos series para cada categoría (eligiendo la seríe de char_len
plt1 = sns.distplot(datos_ejercicio[datos_ejercicio["label"]==0].char_len, hist=True, label="no_profesion")
plt2 = sns.distplot(datos_ejercicio[datos_ejercicio["label"]==1].char_len, hist=True, label="profesion")
fig.legend(labels=['no profesión','profesión'], loc = 5)


# Definimos el título de los ejes:
plt.xlabel('Caracteres', fontsize=16)
plt.ylabel('Densidad', fontsize=16)

# Finalmente mostramos el gráfico:
plt.show()


# En la figura se ve que no existen diferencias significativas entre las dos clases. Por ello, no somos capaces de diferenciar los tweets por clases en base a la longitud. Necesitamos de algún mecanismo de IA para ello.

# ## 3. Preprocesado y Normalización
# Como hemos visto, está dividido en dos pasos Normalización o Preprocesado y Transformación
# 
# ## Normalización
# Vamos a proceder a normalizar los datos. Para ello vamos a generar pequeñas funciones que nos permitan:
# - Eliminar saltos de línea y espacios extra.
# - Eliminar urls
# - Transformar a minúsculas.
# - Tokenizar.
# - Eliminar stopwords.
# - Eliminar sígnos de puntuación.
# - Lematizar tokens.

# * **Eliminar saltos de línea:**
# 
# Eliminamos los saltos de línea.

# In[15]:


# Eliminar saltos de línea
def eliminar_salto_linea(text): 
    return  re.sub('\n','',text) 


# In[16]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text"].apply(lambda x: eliminar_salto_linea(x))


# * **Eliminar espacios extra:**
# 
# Eliminamos espacios extra.

# In[17]:


# Eliminar espacios extra
def eliminar_espacios(text): 
    return  " ".join(text.split()) 


# In[18]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: eliminar_espacios(x))


# * **Quitamos las urls:**
# 
# Eliminamos las urls del texto. En este caso, procedemos a eliminar las urls que aparecen el contenido de los tweets porque no se considera que aporten información.

# In[19]:


# Eliminar url
def eliminar_url(text): 
    return  re.sub('http[s]*\S+','',text)


# In[20]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: eliminar_url(x))


# * **Transformar a minúsculas:**
# 
# Pasamos el texto a minúsculas.

# In[21]:


# Pasar a minúsculas
def texto_to_lower(text):
  return text.lower()


# In[22]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: texto_to_lower(x))


# * **Tokenizar:**
# 
# Obtenemos los tokens.

# In[23]:


# Tokenizador
from nltk.tokenize import TweetTokenizer
def tokenize(texto):
    tweet_tokenizer = TweetTokenizer()
    tokens_list = tweet_tokenizer.tokenize(texto)
    return tokens_list


# In[24]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: tokenize(x))


# * **Eliminar stopwords:**
# 
# Eliminamos stopwords.

# In[25]:


# Quitar stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def quitar_stopwords(tokens):
    stop_words = set(stopwords.words('spanish')) 
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence


# In[26]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: quitar_stopwords(x))


# * **Eliminar signos de puntuación:**
# 
# Eliminamos la signos de puntuación.

# In[27]:


# Eliminar signos de puntuación (nos quedamos sólo lo alfanumérico en este caso)
def quitar_puntuacion(tokens):
    #words=[word for word in tokens if (word.isalnum() or word.startswith("@") or word.startswith("#"))]
    words=[word for word in tokens if word.isalnum()]
    return words


# In[28]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: quitar_puntuacion(x))


# * **Lematizaremos:**
# 
# Lematizaremos.

# In[29]:


# Lemmatization de los tokens. Devuelve una string entera para hacer la tokenización
# con NLTK
nlp = es_core_news_sm.load(disable=['parser', 'ner'])
def lematizar(tokens):
    sentence = " ".join(tokens)
    mytokens = nlp(sentence)
    # Lematizamos los tokens y los convertimos  a minusculas
    mytokens = [ word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Extraemos el text en una string
    return " ".join(mytokens)


# In[30]:


datos_ejercicio["tweet_text_processed"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: lematizar(x))


# Mostramos un gráfico de nube de palabras, una vez tenemos el contenido de los tweets tratado.

# In[31]:


from wordcloud import WordCloud

long_string=str(','.join(list(datos_ejercicio['tweet_text_processed'].values)))
                          
wordcloud = WordCloud(background_color="white",max_words=5000,contour_width=0,contour_color="steelblue")

# Generate a word cloud image
wordcloud.generate(long_string)

wordcloud.to_image()


# * **Extracción de sentimiento:**
# 
# Una vez lematizados, extraemos sentimiento del texto. Al no ser en inglés el contenido de los tweets, no podemos hacer uso de la librería TextBlob.

# In[32]:


url_path = "https://www.cic.ipn.mx/~sidorov/SEL.txt"
sel_lexicon = pd.read_csv(url_path,sep="\t", encoding="latin-1")

dicc_mapping = {"Alegría":"positive",
                "Sorpresa":"positive",
                "Tristeza":"negative",
                "Enojo":"negative",
                "Miedo":"negative",
                "Repulsión":"negative"}
                
sel_lexicon["Categoría"] = sel_lexicon["Categoría"].map(dicc_mapping)

positive_words = dict(zip(sel_lexicon[sel_lexicon["Categoría"]=="positive"].Palabra, sel_lexicon[sel_lexicon["Categoría"]=="positive"][' PFA']))
negative_words = dict(zip(sel_lexicon[sel_lexicon["Categoría"]=="negative"].Palabra, sel_lexicon[sel_lexicon["Categoría"]=="negative"][' PFA']))

def calculate_sentiment(frase, positive_words, negative_words):
  """
  Función para calcular el score de sentimiento de una frase

  Args:
    frase [str]: Frase pre-preprocesada en español. Debe venir lematizada.
    positive_words [dict]: Diccionario de palabras positivas extraidas de SEL
    negative_words [dict]: Diccionario de palabras negativas extraídas de SEL

  Out:
    Sentiment score  
  """
  score = 0
  for i in frase.split():
    if i in positive_words:
      score = score + float(positive_words[i])
    elif i in negative_words:
      score = score - float(negative_words[i])
    else:
      score = score + 0
  
  return score


# In[33]:


datos_ejercicio["tweet_text_sentiment"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: calculate_sentiment(x, positive_words, negative_words))


# In[34]:


datos_ejercicio["tweet_text_sentiment"]


# In[35]:


#datos_ejercicio["tweet_text"].to_csv("tweets.csv", sep='\t', encoding='utf-8')
datos_ejercicio.columns


# * **Distribución de la longitud de los tweets finales en caracteres:**
# 
# Obtenemos la longitud de cada uno de los textos tras haber pasado los procesamientos anteriores para después dibujar su histograma. 

# In[36]:


datos_ejercicio["token_len"] = datos_ejercicio["tweet_text_processed"].apply(lambda x: len(x))

fig = plt.figure(figsize=(14,12))
sns.set_style("darkgrid")
plt1 = sns.distplot(datos_ejercicio[datos_ejercicio["label"]==0].token_len, hist=True, label="no_profesion")
plt2 = sns.distplot(datos_ejercicio[datos_ejercicio["label"]==1].token_len, hist=True, label="profesion")
fig.legend(labels=['no profesión','profesión'],loc=5)

# Definimos el título de los ejes:
plt.xlabel('Caracteres', fontsize=16)
plt.ylabel('Densidad', fontsize=16)

plt.show()


# Vemos que el gráfico sigue sin permitirnos diferenciar en función del número de caracteres. Sin embargo, se puede apreciar que se ha reducido el rango del número de caracteres.

# ## 4. Vectorización

# Con los textos ya tratados, nos disponemos a vectorizar usando TfidfVectorizer.
# 
# Queremos que la funcionalidad 'TfidfVectorizer' tenga en cuenta los siguientes aspectos:
# - Considerar unigramas, bigramas y trigramas
# - Que el sistema no considere los elementos que salgan en menos del 1% de los documentos.
# - No considerar procesados ya hechos (lowercase y stopwords) 

# In[37]:


# BoW Features
vectorizador = TfidfVectorizer(min_df=0.01,ngram_range=(1,3),lowercase=False,stop_words=None)
vector_data = vectorizador.fit_transform(datos_ejercicio["tweet_text_processed"])


# In[38]:


vector_data


# ## 5. Entrenamiento y evaluación de modelos
# 

# Antes de empezar, debemos seleccionar el conjunto de variables que queremos considerar en el entrenamiento. 

# In[39]:


extra_features = datos_ejercicio[['tweet_text_sentiment']]


# In[40]:


extra_features


# Utilizamos la librería scipy (función sparse.hstack) para unir las características TFIDF (contenidas en ´vector_data´) con las que acabamos de seleccionar (´extra_features´). 
# 
# Especificaremos cual es la variable con las clases de cada documento.

# In[41]:


import scipy as sp

# Definimos la variable de categorías:
y = datos_ejercicio["label"].values.astype(np.float32) 

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.fit_transform
X = sp.sparse.hstack((vector_data,extra_features.values),format='csr')


# También vamos a extraer el nombre de las caracteríticas por si quisieramos utilizarlos con posterioridad.

# In[42]:


X_columns=vectorizador.get_feature_names()+extra_features.columns.tolist()


# In[43]:


X_columns


# In[44]:


print(y.shape)


# Tenemos 198 características para 8k documentos.

# Dividimos el dataset en Train/Test:

# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=0)
print(X_train.shape)
print(X_test.shape)


# * **Decision de modelo de ML a utilizar**
# 
# Se ha generado una función para medir la calidad de varios modelos estándar de forma fácil y ver sus resultados. 
# 
# La función hace un KFold y evalua diferentes modelos con una métrica de evaluación.

# In[46]:


# Definimos las funcionalidades pertinentes de sklearn:
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings 
# Definimos la función encargada de evaluar los modelos:
def model_evaluation(models, score, X, y):
      results = []
      names = []
      #PAra cada modelo
      for name, model in models:
          warnings.filterwarnings('ignore') 
          # Generamos un Kfold
          KF = KFold(n_splits = 10, shuffle = True, random_state = 98)

          # hacemos croos_val
          cv_results = cross_val_score(model, X, y, cv = KF, scoring = score, verbose = False)
          
          # Guardamos los resultados:
          results.append(cv_results)
          names.append(name)
          
          # Mostramos los resultados numéricamente:
          print('Metric: {} , KFold '.format(str(score)))
          print("%s: %f (%f) " % (name, cv_results.mean(), cv_results.std()))

      return results, names


# Una vez definida la función, podemos definir los modelos con los que hacer la evaluación. En este caso hemos incorporado la regresión logística, una naive bayes, un random forest y KNeighbors.

# In[47]:


# Cargamos los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Definimos los modelos y generamos una lista con cada uno de ellos:
models = [
         ("Logistic",LogisticRegression(random_state=30)), 
         ("GaussianNB",GaussianNB()),
         ("RandomForestClassifier",RandomForestClassifier(random_state=0)),
         ("KNeighborsClassifier",KNeighborsClassifier(n_neighbors=3))
]

evaluation_score = "accuracy"

model_evaluation(models,  evaluation_score, X.toarray(), y)   


# Observamos un mejor funcionamiento con el regresor logístico, que será con el que intentaremos afinar los hiperparámetros.

# Vamos a generar un diccionario para la búsqueda Grid y también generaremos el objeto GridSearchCV.

# In[48]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = KFold(n_splits=8)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)


# Entrenamos el modelo y hacemos un "print" del mejor resultado.

# In[49]:


grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Mejor accuracy: %f usando los parámetros %s" % (grid_result.best_score_, grid_result.best_params_))


# Los resultados mejores deben ser introducidos a un modelo específico para ser entrenado.

# In[50]:


from sklearn.model_selection import (KFold, cross_val_score,cross_validate)
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

model=LogisticRegression(C=100,penalty="l2",solver="newton-cg")
model.fit(X_train,y_train)


# Predecimos el set de test.

# In[51]:


y_pred = model.predict(X_test)


# Creamos una matriz de confusión y un "classification report".

# In[52]:


from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Por los resultados, vemos que el modelo funciona de manera óptima.
# 
# Pero los datos no estaban balanceados, así que, vamos a balancear y probar de nuevo.

# **Balancear**

# Sólo debemos hacer el rebalanceo en el conjunto de datos de entrenamiento.

# In[53]:


from imblearn.over_sampling import SMOTE

# Creamos el objeto
sm=SMOTE(random_state=50)

# Retransformamos
X_res,y_res = sm.fit_resample(X_train,y_train)


# In[54]:


print(len(y_res[y_res==0]),len(y_res[y_res==1]))
y_res.shape


# Vemos que hay el mismo número de valores para ambas clases y se han añadido registros.
# 
# Con los datos balanceados, volvemos a probar los modelos.

# In[55]:


# Cargamos los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Definimos los modelos y generamos una lista con cada uno de ellos:
models = [
         ("Logistic",LogisticRegression(random_state=30)), 
         ("GaussianNB",GaussianNB()),
         ("RandomForestClassifier",RandomForestClassifier(random_state=0)),
         ("KNeighborsClassifier",KNeighborsClassifier(n_neighbors=3))
]

evaluation_score = "accuracy"

model_evaluation(models,  evaluation_score, X_res.toarray(), y_res)   


# In[56]:


X_res.shape


# Observamos un mejor funcionamiento con el RandoForest, que será con el que intentaremos afinar los hiperparámetros.

# Vamos a generar un diccionario para la búsqueda Grid y también generaremos el objeto GridSearchCV.

# In[57]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestClassifier()

param_grid = { 
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10],
    'criterion' :['gini', 'entropy']
}

cv_technique = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv_technique, scoring='accuracy',error_score=0)


# Entrenamos el modelo y hacemos un "print" del mejor resultado.

# In[58]:


grid_result = grid_search.fit(X_res, y_res)
# summarize results
print("Mejor accuracy: %f usando los parámetros %s" % (grid_result.best_score_, grid_result.best_params_))


# Tenemos modelo! Los resultados mejores deben ser introducidos a un modelo específico para ser entrenado.

# In[59]:


from sklearn.model_selection import (KFold, cross_val_score,cross_validate)
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

model = RandomForestClassifier(n_estimators= 100,max_features='log2',max_depth=10,criterion='gini')

model.fit(X_res.toarray(),y_res)


# Predecimos el set de test.

# In[60]:


y_pred = model.predict(X_test)


# Creamos una matriz de confusión y un "classification report".

# In[61]:


from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Mostramos esa matriz de confusión de una manera gráfica.

# In[62]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test.toarray(), y_test, normalize = None)


# El modelo funciona bastante bien para predecir los tweets de clase 0, pero no da buenos resultados para los de clase 1, que son los que indican alguna profesión. 
# 
# Se tendría que revisar el tema del balanceo porque se han creado registros auxiliares.
# 
# Hay posibilidad de mejora del modelo para que de mejores resultados.
