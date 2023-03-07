#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('wget https://github.com/luisgasco/ntic_master_datos/raw/main/datasets/datos_twitter_master.tsv')
get_ipython().system('pip install emoji_extractor')
get_ipython().system('pip install emoji')
import sys  
get_ipython().system('{sys.executable} -m pip install contractions')
get_ipython().system('pip install spacy==3.2.1')
get_ipython().system('python -m spacy download en_core_web_sm')
import nltk
nltk.download('stopwords')
# Download emoji sentiment
get_ipython().system('wget https://www.clarin.si/repository/xmlui/handle/11356/1048/allzip')
get_ipython().system('unzip allzip')


# # Imports

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import contractions
import re
from emoji_extractor.extract import Extractor
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report


# # Funciones que se utilizarán

# In[ ]:


# Preparar diccionario de emojis
def load_emoji_sentiment(path):
  # Cargamos el csv de emoji_sentiment
  emoji_sent_df = pd.read_csv(path,sep=",")
  # Calculamos los scores dividiendo el número de emojis negativos y entre el total
  emoji_sent_df["Negative"] = emoji_sent_df["Negative"]/emoji_sent_df["Occurrences"]
  emoji_sent_df["Neutral"] = emoji_sent_df["Neutral"]/emoji_sent_df["Occurrences"]
  emoji_sent_df["Positive"] = emoji_sent_df["Positive"]/emoji_sent_df["Occurrences"]
  # Transformamos a dict
  emoji_sent_df = emoji_sent_df.set_index('Emoji')
  emoji_dict = emoji_sent_df.to_dict(orient="index")
  return emoji_dict


# In[ ]:


# Reemplazar contracciones y slang en inglés usando la librería "contractions" https://github.com/kootenpv/contractions
def replace_contraction(text):
    expanded_words = []
    # Divide el texto
    for t in text.split():
        # Aplica la función fix en cada sección o token del texto buscando contracciones y slang
        expanded_words.append(contractions.fix(t, slang = True))
    expanded_text = ' '.join(expanded_words) 
    return expanded_text

# Hay un tokenizador guay para twitter https://github.com/jaredks/tweetokenize


# In[ ]:


# Función para extraer emojis del texto en formato lista
def extract_emojis(text):
  extract = Extractor()
  emojis = extract.count_emoji(text, check_first=False)
  emojis_list = [key for key, _ in emojis.most_common()]
  return emojis_list


# In[ ]:


# Calcula el sentimiento de los emojis de una lista utilizando el diccionario
# de emoji sentiment score generado previamente con la función load_emoji_sentiment()
# Se puede extraer el valor de positividad de los emojis con la option "positive"
# Se puede extraer el valor de neutralidad de los emojis con la option "neutral""  
# Se puede extraer el valor de e negatividad de los emojis con la option "negative""  

def get_emoji_sentiment(lista, option = "positive"):
  output = 0
  for emoji in lista:
    try:
      if option == "positive":
        output = output + emoji_sent_dict[emoji]["Positive"]
      elif option =="negative":
        output = output + emoji_sent_dict[emoji]["Negative"]
      elif option =="neutral":
        output = output + emoji_sent_dict[emoji]["Neutral"]
    except Exception as e: 
      continue
  return output


# In[ ]:


# Eliminar los emojis de un texto. Esto es útil porque una vez extraido los emojis
# puede interesarnos tener un texto sin presencia de emojis para mejor análisis.
def clean_emoji(text):
    # Poner todos los comandos de http://www.unicode.org/Public/emoji/1.0/emoji-data.txt
    emoji_pattern = re.compile("["
        u"\U0001F300-\U0001F6FF"  # symbols & pictographs
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u'\u2600-\u26FF\u2700-\u27BF'
        u'\u2934' u'\u2935' u'\u2B05' u'\u2B06' u'\u2B07' u'\u2B1B' u'\u2B1C' 
        u'\u2B50' u'\u2B55' u'\u3030' u'\u303D' u'\u3297' u'\u3299' u'\u00A9'
        u'\u00AE' u'\u203C' u'\u2049' u'\u2122' u'\u2139' u'\u2194-\u2199' 
        u'\u21A9' u'\u21AA' u'\u231A' u'\u231B' u'\u2328' u'\u23CF'
        u'\u23E9-\u23F3' u'\u23F8' u'\u23F9' u'\u23FA' u'\u24C2' u'\u25AA'
        u'\u25AB' u'\u25B6' u'\u25C0' u'\u25FB' u'\u25FD' u'\u25FC' u'\u25FE'
        ']+', flags=re.UNICODE)
    string2 = re.sub(emoji_pattern,r' ',text)
    return string2


# In[ ]:


# Tokenizar los tweets con el tokenizador "TweetTokenizer" de NLTK
def tokenize(texto):
  tweet_tokenizer = TweetTokenizer()
  tokens_list = tweet_tokenizer.tokenize(texto)
  return tokens_list

# Quitar stop words de una lista de tokens
def quitar_stopwords(tokens):
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence


# Eliminar signos de puntuación de una lista de tokens
# (nos quedamos sólo lo alfanumérico en este caso)
def quitar_puntuacion(tokens):
    words=[word for word in tokens if word.isalnum()]
    return words


# Lemmatization de los tokens. Devuelve una string entera para hacer la tokenización
# con NLTK
nlp = en_core_web_sm.load(disable=['parser', 'ner'])
def lematizar(tokens):
    sentence = " ".join(tokens)
    mytokens = nlp(sentence)
    # Lematizamos los tokens y los convertimos  a minusculas
    mytokens = [ word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Extraemos el text en una string
    return " ".join(mytokens)


# # Cargar y preparar los datos
# En primer lugar cargamos los datos que vamos a utilizar en este notebook (emoji_sentiment_data y datos de twitter).

# In[ ]:


emoji_sent_dict = load_emoji_sentiment("Emoji_Sentiment_Data_v1.0.csv")


# Podemos textear que hemos obtenido un diccionario cuyas claves son los emojis presentes dentro de emoji sentiment score. Cada emoji tiene un score de negatividad, neutralidad, positividad y otros campos.

# In[ ]:


emoji_sent_dict["😭"]


# Cargamos el fichero .tsv con los datos de Twitter:

# In[ ]:


dataset = pd.read_csv("datos_twitter_master.tsv", sep="\t")
dataset


# # Análisis preliminar 

# # Análisis exploratorio de los datos (EDA)
# 
# 
# 

# En este apartado pretendemos realizar un análisis de los datos previo a la normalización de los mismos. Este análisis nos va a permitir extraer información relevante del dataset, así como posibles inconvenientes que serán solucionados llegado el caso.
# 

# * **Número de documentos y columnas:**
# 
# Comenzamos mostrando el número de documentos, o lo que es lo mismo, el número de filas del data frame:

# In[ ]:


print("Tenemos un conjunto de {} documentos".format(len(dataset)))
print("El dataframe tiene {} columnas".format(dataset.shape[1]))


# * **Número de documentos duplicados:**

# Despues, comprobamos y eliminamos las filas con algún valor vacío (NA) y quitaremos los duplicados.

# In[ ]:


print("Existen {} noticias duplicadas".format(np.sum(dataset.duplicated(subset=["tweet_text"]))))
# Quitaremos esos duplicados
dataset = dataset.drop_duplicates()
print("Despues de quitar duplicados tenemos un conjunto de {} noticias".format(dataset.shape[0]))


# Comprobaramos que no hayan quedado Nulls en ningunas de las dos columnas del dataset

# In[ ]:


print("Hay {} valores vacíos en las noticias y {} valores vacíos en las etiquetas en los datos".format(np.sum(dataset.isnull())[0],
                                                                                                        np.sum(dataset.isnull())[1]))


# * **Número de documentos por cada clase:**
# 
# Contamos el número de elementos de cada clase. Vemos que en la columna "molestia" nos encontramos las etiquetas del dataset. En este caso nos encontramos dos tipos de documentos (tweets):
# 
# - "Molestia = 1": Tweets con la palabra ruido que hacen referencia a molestias sufridas por ruido acústico proveniente de distintas fuentes (coches, vecinos, mascotas,...)
# - "Molestia = 0": Tweets que contienen la palabra ruido perso no expresan una molestia sufrida por el usuario que lo escribió (otras acpciones de ruido, noticias que hablan sobre ruido o uso de ruido como algo positivo) 

# Comprobemos la distribución de las clases:

# In[ ]:


dataset["molestia"].value_counts()


# ¡¡Tenemos un dataset balanceado!! Esto nos evitará problemas en el entrenamiento de los modelos😀. 
# 
# Disponemos 509 noticias verdaderas (valor 0) y 29571 noticias falsas (valor 1).
# 
# Vamos a dibujar un histograma con las clases así practicamos:

# In[ ]:


ax, fig = plt.subplots()
etiquetas = dataset.molestia.value_counts()
etiquetas.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()


# * **Distribución de la longitud de los tweet en caracteres:**
# 
# Para seguir con el análisis exploratorio, vamos a hacer un cálculo típico: la longitud de cada uno de los textos de los documentos para despues dibujar su histograma. 
# 
# Comenzamos creando las columnas que van a almacenar las longitud en caracteres y en tokens de los documentos del corpus:

# In[ ]:


dataset["char_len"] = dataset["tweet_text"].apply(lambda x: len(x))


# In[ ]:


# Importamos las librerías matplotlib y seaborn:
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(14,12))
sns.set_style("darkgrid")
# añadimos series para cada categoría (eligiendo la seríe de char_len
plt1 = sns.distplot(dataset[dataset["molestia"]==0].char_len, hist=True, label="no_molestia")
plt2 = sns.distplot(dataset[dataset["molestia"]==1].char_len, hist=True, label="molestia")
fig.legend(labels=['no molestia','molestia'], loc = 5)


# Definimos el título de los ejes:
plt.xlabel('Caracteres', fontsize=16)
plt.ylabel('Densidad', fontsize=16)

# Finalmente mostramos el gráfico:
plt.show()


# En la figura se ve que no existen diferencias significativas entre las dos clases. Quizá los tweets en los que el usuario se queja sobre el ruido (molestia ==1)tienen una tendencia a ser más cortos, pero no se observa nada destacable.

# # Transformación 
# 
# Como hemos visto, está dividido en dos pasos Normalización o Preprocesado y Transformación
# 

# 
# ## Normalización
# Vamos a proceder a normalizar los datos. Para ello vamos a utilizar las funciones anteriormente definidas:
# 
# - Por una parte vamos a extraer los emojis de los tweets, los vamos a guardar en una lista dentro de una nueva columna del dataframe y por último calcularemos un valor de sentimiento de emojis de positividad, negatividad y neutralidad.
# 
# - Preprocesar los textos:
#     - Primero expanderemos las contracciones de los tweets
#     - Despues quitaremos los emojis, ya que antes habremos calculado los scores necesarios.
#     - Tokenizaremos
#     - Quitaremos stop words
#     - Quitaremos puntuación
#     - Lematizaremos

# ### Emojis
# En primer luigar vamos a trabajar con los emojis. 
# 
# Vamos a extraerlos con una función lambda aplicando la función extract_emojis() definida anteriormente en el dataframe

# In[ ]:


dataset["emoji_list"] = dataset["tweet_text"].apply(lambda x: extract_emojis(x))


# Vemos que nos ha guardado los emojis en la columna "emoji_list":

# In[ ]:


dataset["emoji_list"]


# A continuación, se calcula un score de sentimiento a los emojis asociados a cada tweet. Si no hay emojis, estos scores serán cero.
# Para calcular esto lo haremos de nuevo con funciones lambda aplicando la función get_emoji_sentiment() anteriormente generada:

# In[ ]:


dataset["sent_emoji_pos"] = dataset["emoji_list"].apply(lambda x: get_emoji_sentiment(x, "positive"))
dataset["sent_emoji_neu"] = dataset["emoji_list"].apply(lambda x: get_emoji_sentiment(x, "neutral"))
dataset["sent_emoji_neg"] = dataset["emoji_list"].apply(lambda x: get_emoji_sentiment(x, "negative"))


# In[ ]:





# ### Preprocesar textos
# Vamos a realizar los preprocesados indicados antes.

#  En primer lugar expandimos las contracciones. Además, despues del proceso de extracción de emojis, los quitaremos de nuestros textos porque no nos serán útiles.

# In[ ]:


# Reemplazar contracciones
dataset["tweet_text_processed"] = dataset["tweet_text"].apply(lambda x: replace_contraction(x))
# Quitar emojis de los textos
dataset["tweet_text_processed"] = dataset["tweet_text_processed"].apply(lambda x: clean_emoji(x))


# Despues tokenizamos el texto, y trabajaremos en limpiar los tokens que no son útiles en este problema para reducir dimensionalidad

# In[ ]:


dataset["tokenized"] = dataset["tweet_text_processed"].apply(lambda x: tokenize(x))


# Procesamos los tokens:

# In[ ]:


# Quitar stopwords
dataset["tokenized_clean"] = dataset["tokenized"].apply(lambda x: quitar_stopwords(x))
# Quitamos los símbolos de puntuación
dataset["tokenized_clean"] = dataset["tokenized_clean"].apply(lambda x: quitar_puntuacion(x))
# Lematizamos
dataset["lematizacion"] = dataset["tokenized_clean"].apply(lambda x: lematizar(x))


# En este paso también podríamos generar nuevas características para mejorar el funcionamiento del algoritmo. Por ejemplo, podríamos utilizar TextBlob para obtener el sentimiento (tanto subjetividad y polaridad) de cada twitter. Esto se hace de la siguiente forma:
# 
# 
# ```
# from textblob import TextBlob
# Textblob(tweet_text).sentiment.subjectivity
# Textblob(tweet_text).sentiment.polarity
# ```
# 
# PAra aplicarlo a cada texto de un dataframe habría que utilizar funciones Lambda.

# ## Vectorización
# 
# Una vez hemos limpiado y procesado el texto, vamos a extraer características utilizando TFIDFVectorizer:
# - Queremos utilizar como máximo *30* features
# - unigramas, bigramas y trigramas
# - Que el sistema no considere los elementos que salgan en menos del 5% de los documentos.

# In[ ]:


# BoW Features
vectorizador = TfidfVectorizer(min_df=0.01, ngram_range = (1,3), max_features = 30)
vector_data = vectorizador.fit_transform(dataset["lematizacion"])


# In[ ]:




vector_data


# # Entrenar/testear el clasificador
# 
# En esta ocasión, además de utilizar las características de *Bag of Ngramas* generadas con TfidfVectorizer, nos interesa utilizar otro conjunto de características que podrían ser de interés para mejorar el rendimiento del clasificador.
# 
# En este caso, vamos a introducir como ejemplo las variables de sentimiento de emojis que hemos calculado.
# 
# La forma más sencilla de hacer esto es utilizar la librería *scipy* y generar una matriz sparse, comprensible por scikit-learn, que contenga tanto las características de TFIDF como las calculadas manualmente. 
# 
# En primer lugar, debemos seleccionar el conjunto de variables que queremos considerar en el entrenamiento. PAra ello hacemos uso del selector `dataframe[["nombre_columna1", "nombre_columna2"]]`:

# In[ ]:


extra_features = dataset[['sent_emoji_pos','sent_emoji_neg','sent_emoji_neu']]


# In[ ]:


vector_data.dtype


# Utilizamos la librería scipy (función sparse.hstack) para unir las características TFIDF (contenidas en ´vector_data´) con las que acabamos de seleccionar (´extra_features´). Esta unión nos generará una matriz X que utilizaremos para hacer el train-test split posteriormente:

# In[ ]:


import scipy as sp
# Extraemos las etiquetas y las asignamos a la variable y
y = dataset["molestia"].values.astype(np.float32) 
# Unimos las características TFIDF con las características previamente seleccionadas
# Extraemos los valores (values) de las extra_features, que es un dataframe  
X = sp.sparse.hstack((vector_data,extra_features.values),format='csr')


# También vamos a extraer el nombre de las caracteríticas por si quisieramos utilizarlos con posterioridad.

# In[ ]:


X_columns=vectorizador.get_feature_names()+extra_features.columns.tolist()


# Vamos a dividir nuestros datos en Train y Test, como habitualmente se hace. En este caso probablemente tuvieramos demasiadas características (303 características para 764 datos nos va a dar problemas de overfitting, así que en los ejercicios deberías tener esto en cuenta [bajas el número de características])

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)


# **Decision de modelo de ML a utilizar**
# 
# En primer lugar se ha generado una función para medir la calidad de varios modelos estándar de forma fácil y ver sus resultados. 
# 
# La función hace un KFold y evalua diferentes modelos con una métrica de evblauación:

# In[ ]:


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


# Una vez definida la función, podemos definir los modelos con los que hacer la evaluación. En este caso hemos incorporado la regresión logística y una naive bayes. 

# In[ ]:


# Cargamos los modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Definimos los modelos y generamos una lista con cada uno de ellos:
models = [
         ("Logistic",LogisticRegression(random_state=30)),
         ("GaussianNB",GaussianNB())
]

evaluation_score = "accuracy"

model_evaluation(models,  evaluation_score, X.toarray(), y)   


# Definimos las variables para hacer una grid_searc:

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]


# In[ ]:


grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = KFold(n_splits=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)


# Vamos a entrenar el grid_search para obtener el mejor parámetro para nuestro conjunto de datos.

# In[ ]:


grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Mejor accuracy: %f usando los parámetros %s" % (grid_result.best_score_, grid_result.best_params_))


# Entrenamos el modelo con los resultados ofrecidos por la grid_search:

# In[ ]:


from sklearn.model_selection import (KFold, cross_val_score,cross_validate)
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

model=LogisticRegression(C=10, penalty="l2", solver = "newton-cg")
model.fit(X_train,y_train)


# Vamos a ver como funciona el modelo haciendo el predict del test y mostrando la matriz de confusióñn y el classifciation_Report:

# In[ ]:


y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Además, podríamos mostrar el grado de importancia relativa de las variables dle modelo. Aquí hago el listado, pero lo ideal sería seleccionar las más importantes dentro del modelo para saber cuales están teniendo más influencia:

# In[ ]:


# Obtener la importancia de las variables del modelo
importance = model.coef_[0]


# A continuación utilizamos esa variable de importancia de variables, junto a los nombres de las características almacenadas anteriormente en X_columns, para listar la importancia de cada una de las variables.

# In[ ]:


# Mostrar el número de la característica, con su nombre, y su score de importancia
for i,v in enumerate(importance):
 print('Feature: %0d, Name: %s , Score: %.5f' % (i,X_columns[i],v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

