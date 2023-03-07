#!/usr/bin/env python
# coding: utf-8

# #### Problema de text mining. Puntos más importantes
# Cuando nos enfrentamos a un problema de text mining tenemos que tener en cuenta una serie de puntos que afectarán las técnicas y los recursos que se van a utilizar:
# 
# *   **Idioma**: No se utilizan los mismos recursos en lengua española que en inglés. Si trabajamos con idioma español, tendremos que asegurarnos de utilizar recursos adaptados para este idioma ya sean **modelos de spacy específicos**, lexicons generados para este idioma, o adaptar léxicons existentes a nuestra tarea específica.
# 
# De cara al ejercicio teneis que:
# - Comprobar que el modelo de spacy que utilizais es en español: nlp = spacy.load("es_core_news_sm")
# - Si optais por utilizar stemming, debeis aseguraros de utilizar la función SnowballStemmer (no PorterStemmer()), que es la que funciona en español: stenner = SnowballStemmer(language="spanish") 
# - Comprobar que estais utilizando una lista de stopwords en español: stopwords.words('spanish')
# 
# Como los textos están en español **NO** deberiais extraer el sentimiento utilizando TextBlob(). TextBlob calcula el sentimiento a través de lexicons en inglés y el resultado será relativamente malo. 
# 
# Adicionalmente, es mejor que no utiliceis el atributo max_features en TfidfVectorizer(). Si quereis limitar el número de características utilizar un valor pequeño de min_df. Por ejemplo, min_df = 0.005. 
# 
# 
# 
# 
# 
# *   **Preprocesado/normalización de los datos**: Aunque de cara al ejercicio esto no es importante, es esencial definir un pipeline de preprocesado de datos específico para cada tipo de datos. Los datos de Twitter contiene una serie de partículas específicas que requieren preprocesados específicos, los datos médicos otros... Lo mejor suele ser **consultar bibliografía** [Link ejemplo](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232525)
# 
# 
# 
# En este notebook os presento un ejercicio para que veais como:
# 1. Rebalancear datos utilizando imblearn
# 2. Utilizar embeddings para representar documentos y transformarlos para que scikit learn los entienda
# 
# 
# 
# 
# 
# 

# ### Descarga de datos:
# 
# Vamos a descargar los datos proporcionados por la compañia [sentiment140](http://help.sentiment140.com/for-students). 

# In[ ]:


get_ipython().system('wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
get_ipython().system('unzip trainingandtestdata.zip')


# Los datos con los que vamos a trabajar están contenidos en el archivo csv llamado "testdata.manual.2009.06.14.csv". El archivo tiene 6 campos:
# 
# 
# 
# *   Polaridad del tweet:
#     - 0 --> Negativo
#     - 2 --> Neutral 
#     - 4 --> Positivo
# *   Identificador
# *   Fecha del tweet
# *   Campo no utilizado
# *   Usuario autor del tweet
# *   Texto del tweet
# 
# 

# ### Importación de datos
# 

# In[ ]:


import numpy as np
import pandas as pd

datos = pd.read_csv("training.1600000.processed.noemoticon.csv",sep=",",
                    names=["label","id","fecha","not_used","autor","texto"],
                    header=None, encoding='latin-1')


# In[ ]:


datos.head(1)


# Nos vamos a quedar sólo con los tweets con etiqueta positivo (4) y negativo (0):

# In[ ]:


datos = datos[datos.label.isin([0,4])]


# Vamos a trabajar con una muestra más reducida (un total de 20000 tweets). "Desbalanceando" el conjunto de datos para tener que gestionar el desbalanceo de clases.

# In[ ]:


neg_tweets = datos[datos.label==0].sample(13000)
pos_tweets = datos[datos.label==4].sample(7000)
datos = pd.concat([neg_tweets,pos_tweets ], ignore_index=True)


# ### Análisis exploratorio de los datos (EDA)
# 

# Como es habitual, vamos a hacer un pequeño análisis exploratorio de los datos.
# 

# 
# *   **Nº de documentos y columnas**
# 
# 

# In[ ]:


print("Tenemos un conjunto de {} documentos".format(len(datos)))
print("El dataframe tiene {} columnas".format(datos.shape[1]))


# 
# *   **Análisis simple de duplicados**
# 
# 

# In[ ]:


print("Existen {} tweets que están duplicados. Procedemos a eliminarlos...".format(np.sum(datos.duplicated(subset=["texto"]))))
# Quitaremos esos duplicados
datos = datos.drop_duplicates(subset="texto")
print("Despues de quitar duplicados tenemos un conjunto de {} noticias".format(datos.shape[0]))


# *   **Análisis de etiquetas**
# 
# Anteriormente hemos seleccionado unicamente las etiquetas 0 y 4 para simplificar el clasificador que generemos. Vamos a ver la distribución de estas:
# 
# 
# 
# 

# In[ ]:


datos["label"].value_counts()


# Como hemos visto antes, es un dataset balanceado. Nuestra clase de interes (clase positiva,4) es claramente minoritaria. Tendremos que gestionar el desbalanceo de clases en pasos posteriores.
# 
# 
# _importante para el ejercicio propuesto!!!_

# *   **Contenido de los tweets**
# 
# Antes de empezar a procesar, vamos a revisar algunos tweets de cada una de las dos clases

# Tweets positivos:

# In[ ]:


datos[datos.label==4].sample(3).texto.to_list()


# Tweets negativos:

# In[ ]:


datos[datos.label==0].sample(3).texto.to_list()


# * **Distribución de la longitud de los tweets en caracteres:**
# 
# Para seguir con el análisis exploratorio, vamos a hacer un cálculo típico: la longitud de cada uno de los textos de los documentos para despues dibujar su histograma. 
# 
# Comenzamos creando las columnas que van a almacenar las longitud en caracteres y en tokens de los documentos del corpus:

# In[ ]:


datos["char_len"] = datos["texto"].apply(lambda x: len(x))


# In[ ]:


# Importamos las librerías matplotlib y seaborn:
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(14,6))
sns.set_style("darkgrid")
# añadimos series para cada categoría (eligiendo la seríe de char_len
plt1 = sns.distplot(datos[datos["label"]==0].char_len, hist=True, label="negativos")
plt2 = sns.distplot(datos[datos["label"]==4].char_len, hist=True, label="positivos")
fig.legend(labels=['negativos','positivos'], loc = 5)


# Definimos el título de los ejes:
plt.xlabel('Caracteres', fontsize=16)
plt.ylabel('Densidad', fontsize=16)

# Finalmente mostramos el gráfico:
plt.show()


# ### Transformación
# 

# Vamos a proceder a normalizar los datos. Para ello vamos a utilizar las funciones anteriormente definidas:
# 
# - Preprocesar los textos:
#     - Primero expanderemos las contracciones de los tweets
#     - En esta ocasión no extraeremos los emojis.
#     - Tokenizaremos
#     - Quitaremos stop words
#     - Quitaremos puntuación
#     - Lematizaremos

# #### Expandir contracciones.
# Este paso sólo se hace con contenido en inglés!! 

# In[ ]:


get_ipython().system('pip install contractions')
import contractions
# Reemplazar contracciones y slang en inglés usando la librería "contractions" https://github.com/kootenpv/contractions
def replace_contraction(text):
    expanded_words = []
    # Divide el texto
    for t in text.split():
        # Aplica la función fix en cada sección o token del texto buscando contracciones y slang
        expanded_words.append(contractions.fix(t, slang = True))
    expanded_text = ' '.join(expanded_words) 
    return expanded_text


# In[ ]:


replace_contraction("I'm very happy, I dunno why")


# Ejecutamos la función

# In[ ]:


datos["tweet_texto_processed"] = datos["texto"].apply(lambda x: replace_contraction(x))


# #### Transformamos a minúsculas
# Así reducimos dimensionalidad

# In[ ]:


datos["tweet_texto_processed"] = datos["tweet_texto_processed"].apply(lambda x: x.lower())


# In[ ]:


datos["tweet_texto_processed"].iloc[0]


# #### Tokenizamos
# Utilizando un tokenizador específico para TWITTER!!

# In[ ]:


from nltk.tokenize import TweetTokenizer
# Tokenizar los tweets con el tokenizador "TweetTokenizer" de NLTK
def tokenize(texto):
  tweet_tokenizer = TweetTokenizer()
  tokens_list = tweet_tokenizer.tokenize(texto)
  return tokens_list


# In[ ]:


datos["tweet_texto_processed"] = datos["tweet_texto_processed"].apply(lambda x: tokenize(x))


# In[ ]:


datos["tweet_texto_processed"].iloc[0]


# #### Quitamos stopwords
# Las stopwords cambian entre idiomas!

# In[ ]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Quitar stop words de una lista de tokens
def quitar_stopwords(tokens):
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence


# In[ ]:


datos["tweet_texto_processed"] = datos["tweet_texto_processed"].apply(lambda x: quitar_stopwords(x))


# In[ ]:


datos["tweet_texto_processed"].iloc[0]


# #### Puntuación
# Podríamos quitarla, pero en ese caso vamos a mantenerla, para ver si hay algún conjunto de caracteres como ":)" que formen alguna característica.

# In[ ]:


":-)"


# #### Lematizamos
# De nuevo, DEPENDIENTE del idioma

# In[ ]:


get_ipython().system('pip install spacy==3.2.1')
get_ipython().system('python -m spacy download en_core_web_sm')
import en_core_web_sm
from tqdm import tqdm
tqdm.pandas()

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


# In[ ]:


datos["tweet_texto_processed"] = datos["tweet_texto_processed"].progress_apply(lambda x: lematizar(x))


# In[ ]:


datos["tweet_texto_processed"].iloc[0]


# #### Extraemos sentimiento de los tweets
# De nuevo, DEPENDIENTE del idioma. Utilizaremos [textblob](https://textblob.readthedocs.io/en/dev/quickstart.html?highlight=polarity#sentiment-analysis). Para español hay que buscar alternativas.

# In[ ]:


from textblob import TextBlob
datos["sent_subjectivity_str"] = datos["tweet_texto_processed"].progress_apply(lambda x: TextBlob(x).sentiment.subjectivity)
datos["sent_polarity_str"] = datos["tweet_texto_processed"].progress_apply(lambda x: TextBlob(x).sentiment.polarity)


# In[ ]:


datos["sent_polarity_str"].hist(bins=80)


# #### Vectorización y unión  características
# 
# Una vez hemos limpiado y procesado el texto, vamos a extraer características utilizando embeddings:

# In[ ]:


import gensim.downloader as api
glove_emb = api.load('glove-twitter-25') # Descargamos y cargamosel embedding de "glove-twitter-25"


# In[ ]:


def get_average_vector(sentence):
  #retokenizamos con nuestra función
  tokens = tokenize(sentence)
  # Generamos lista de salida vacía
  lista = list()
  # Iteramos por cada token de la frase de entrada
  for i in tokens:
    # Si el token se encuentra en el embedding, añadir a la lista. 
    # Si no se encuentra (except), pasa al siguiente elemento.
    try:
      lista.append(glove_emb.get_vector(i))
    except:
      continue

  # Calculamos el valor medio de los vectores generados
  try:
    resultado = np.mean(lista, axis=0)  # 1vector - Dimension 25d
  except:
    # Si la lista está vacía, generar vector de ceros de tamaño el embedding
    resultado = np.zeros(25)
  return resultado


# In[ ]:


datos["embeddings"] = datos["tweet_texto_processed"].progress_apply(lambda x: get_average_vector(x))


# Concatenamos embeddings, y características extras:

# In[ ]:


vector_data = pd.concat([datos.embeddings.apply(pd.Series),
                datos[["sent_polarity_str","sent_subjectivity_str"]]], axis=1)


# In[ ]:


vector_data.shape


# In[ ]:


vector_data = vector_data.fillna(0)


# Utilizamos la librería scipy (función sparse.hstack) para unir las características TFIDF (contenidas en ´vector_data´) con las que acabamos de seleccionar (´extra_features´). Esta unión nos generará una matriz X que utilizaremos para hacer el train-test split posteriormente:

# In[ ]:


import scipy as sp
# Extraemos las etiquetas y las asignamos a la variable y
y = datos["label"].values.astype(np.float32) 
X = sp.sparse.csc_matrix(vector_data)


# #### Dividimos Train/Test
# 
# 

# **División train/test**

# Dividimos el conjunto de datos entre train/test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)
print(X_test.shape)


# #### Rebalanceo de clases:
# 
# 
# 
# 

# 
# 1. Entrenar un modelo sencillo para verificar que el desbalanceo provoca problemas de rendimiento. 
# 
# 2. Utilizar librerías existentes para hacer esa gestión, como por ejemplo [**imlearn**](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE). La gestión es mejor hacerla despues del análisis exploratorio. [link to smote](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE)
# 
# El código es muy simple: 
# 
# ```
# # Cargamos
# from imblearn.over_sampling import SMOTE 
# # Creamos objetos
# sm = SMOTE(random_state=42)
# # Retransformamos
# X_res, y_res = sm.fit_resample(X, y)
# ```
# 
# Sólo se debe hacer rebalanceo en el conjunto de datos de entrenamiento. Si se hace en el de test estaremos evaluando el modelo resultante en un conjunto de datos con una distribución incorrecta.
# 

# In[ ]:


# Cargamos
from imblearn.over_sampling import SMOTE 
# Creamos objetos
sm = SMOTE(random_state=42)
# Retransformamos
X_res, y_res = sm.fit_resample(X_train, y_train)


# Observamos el resultado del rebalanceo:

# In[ ]:


np.array(np.unique(y_res, return_counts=True)).T


# #### Entrenamiento/testeo modelo

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

model_evaluation(models,  evaluation_score, X_res.toarray(), y_res)   


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


grid_result = grid_search.fit(X_res, y_res)
# summarize results
print("Mejor accuracy: %f usando los parámetros %s" % (grid_result.best_score_, grid_result.best_params_))


# Entrenamos el modelo con los resultados ofrecidos por la grid_search:

# In[ ]:


from sklearn.model_selection import (KFold, cross_val_score,cross_validate)
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

model=LogisticRegression(C=10, penalty="l2", solver = "newton-cg")
model.fit(X_res,y_res)


# Vamos a ver como funciona el modelo haciendo el predict del test y mostrando la matriz de confusióñn y el classifciation_Report:

# In[ ]:


from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ### Importancia variables:
# 

# Además, podríamos mostrar el grado de importancia relativa de las variables dle modelo. Aquí hago el listado, pero lo ideal sería seleccionar las más importantes dentro del modelo para saber cuales están teniendo más influencia:

# In[ ]:


# Obtener la importancia de las variables del modelo
importance = model.coef_[0]


# En este caso, para el modelo de regreisón logística la importancia viene dada por los coeficientes de las características del modelo. Si el coeficiente tiene coeficiente positivo, indica que la presencia de esa característica nos ayua a a clasificar el tweet a nuestra clase positiva (clase de interés), si el coeficiente (importancia) es negativa indica que la presencia de ese tweet "empuja" al clasificador a clasificar el tweet en nuestra clase negativa. 

# A continuación utilizamos esa variable de importancia de variables, junto a los nombres de las características almacenadas anteriormente en X_columns, para listar la importancia de cada una de las variables.

# In[ ]:


# Mostrar el número de la característica, con su nombre, y su score de importancia
for i,v in enumerate(importance):
 print('Feature: %0d, Name: %s , Score: %.5f' % (i,X_columns[i],v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

