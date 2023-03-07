#!/usr/bin/env python
# coding: utf-8

# # Técnicas Básicas de NLP en el Text Mining
# 
# En el Text Mining y el NLP se utilizan librerías tradicionales de Data Mining, como scikit-learn o Tensorflow, pero también librerías específicas para trabajar con texto. 
# 
# Existen multitud de librerías diseñadas para preprocesar textos, transformar textos en vectores u orientadas a poner modelos de TM en producción. En este notebook introduciremos 2 de ellas, viendo algunas de sus funcionalides y como aplicar diferentes técnicas básicas de NLP utilizandolas. 

# En primer lugar, instalaremos os las librerías de programación que utilizaremos en este notebook:
# - [**NLTK**](https://www.nltk.org//): NLTK es una de las librerías principales para trabajar con textos libres que fue creada por la Universidad de Pennsylvania en el año 2001. Aunque su uso principal ha estado unido  a entornos de investigación y educación, las facilidades en su uso y sus características la convierten en una de las librerías con un mayor número de recursos de aprendizaje como libros, foros o tutoriales. Contiene una gran cantidad de conjuntos de datos típicos para el aprendizaje de NLP y es muy utilizada en tareas para el preprocesado de texto antes de introducirlo en algoritmos de Inteligencia Artificial.
# 

# In[ ]:


# Instalamos nltk
get_ipython().system('pip install nltk')
# Importamos
import nltk
# Complementos de la librería necesarios para su funcionamiento.
# Todas las opciones aquí https://www.nltk.org/nltk_data/ 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')


# - [**Spacy**](https://spacy.io/): A diferencia de NLTK, que surgió y ha sido diseñada para ser utilizada en entornos de investigación, Spacy se centra en proporcionar herramientas para poder incorporar sistemas de Text Mining en producción por facilidad. De hecho, su fácil interconexión con otras librerías del mundo de la ciencia de datos, junto a la incorporación de modelos pre-entrenados con técnicas de Deep Learning y su facilidad para trabajar con múltiples lenguajes de programación, la han convertido en una de las librarías más usadas, si no la que más, en la actualidad. 
# 
#     Descargamos la librería y los modelos pre-entretados *en_core_web_sm* y *es_core_web_sm*, modelos de DNN entrenados con noticias, blogs y comentarios en inglés y español respectivamente.
# 

# In[ ]:


# Instalamos textacy
get_ipython().system('pip install textacy')
# Instalamos spacy y uno de sus modelos
get_ipython().system('pip install spacy')
# Descargamos modelos pre-entrenados de spacy.
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('python -m spacy download es_core_news_sm')

# Descargamos datos del repositorio de github
get_ipython().system('wget "https://github.com/luisgasco/ntic_master_datos/raw/main/datasets/news_summary.csv"')


# In[ ]:


# Librerías tpipicas
import pandas as pd


# 
# ## Corpus y corpora
# 
# 

# Todo proceso de análisis textual comienza con un dataset de documentos textuales, que generalmente se llama **corpus** o *corpora* cuando tratamos con varios datasets. El corpus generalmente está compuesto de texto bruto con algunos metadatos asociados, aunque esto no tiene por qué ser así. 
# 
# En este Notebook vamos a trabajar con un corpus de noticias distribuido en la plataforma Kaggle llamado [*News summary*](https://www.kaggle.com/sunnysai12345/news-summary). Este corpus está distribuido en formato *csv*, sin embargo es normal encontrar corpus con el formato *tsv*, o disponer de corpus almacenados en base de datos como MongoDB.
# 
# En primer lugar lo leeremos de la ruta donde se ha descargado `/content/news_summary.csv`:
# 

# In[ ]:


news_summary = pd.read_csv('../content/news_summary.csv', encoding='latin-1')
news_summary.head(3)


# El dataset está compuesto por un conjunto de filas, que llamamos documentos. Cada documento tiene un conjunto de metadatos como el autor, la fecha, el titular de la noticia y la web de la noticia y el texto asociado a esta. 
# 
# Cada uno de los textos puede separarse en párrafos, frases y palabras según el tipo de documento y el tipo de análisis que se le vaya a aplicar. 
# 
# En este caso, al ser un ejercicio, únicamente vamos a trabajar con el texto de los documentos, correspondiente el campo "text", así que extraeremos y transformaremos esta columna en una lista para trabajar más comodos:
# 

# In[ ]:


# Transformar la columna "text" a una lista
texto_noticias = news_summary["text"].to_list()
print(type(texto_noticias))


# Vamos a mirar el número de noticias que contiene nuestro corpus:

# In[ ]:


print("El corpus news_summary contiene un total de {} documentos".format(len(texto_noticias)))


# ## Tokenización

# El texto bruto está compuesto por una secuencia de caracteres. Antes de su análisis los textos son divididos en fragmentos más pequeños conocidos como tokens. Un token puede ser tanto una palabra, como un símbolo de puntuación, un número o un emoticono, en el caso de estar analizando datos de redes sociales.
# 
# El proceso de división del texto en tokens se llama tokenización. Aquí se muestra el proceso tanto para la librería Spacy como para la librería NLTK para un único texto del corpus.

# ***NLTK***
# 
# El tokenizador estándar de NLTK se llama word_tokenize. Podemos ver más información dentro de la web de documentación de NLTK (dentro del módulo word_tokenize [texto del enlace](https://www.nltk.org/api/nltk.tokenize.html))
# 
# 
# También podemos utilizar la línea de código `?libreria.modulo.funcion` para que nos aparezca la ayuda de la función en la parte derecha de la pantalla.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'nltk.tokenize.word_tokenize')


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]
print(subset_noticias[4])
# Segmentar las frases de la noticia 5 (indice 4)
sentences = sent_tokenize(subset_noticias[4])
for num,sentence in enumerate(sentences):
    print('La oración número {} es: \n {}'.format(num, sentence))


# Podemos segmentar todos los tokens de un documento de forma global.

# In[ ]:


# Segmentar los tokens de la noticia 5 (indice 4)
tokens = word_tokenize(subset_noticias[4])
for num,token in enumerate(tokens):
    print('El token {} es {} '.format(num,token))


# Pero también se puede segmentar los tokens de cada una de las frases separadamente:

# In[ ]:


# Segmentar las frases de la noticia 5 (indice 4)
sentences = sent_tokenize(subset_noticias[4])
for num_sen, sentence in enumerate(sentences):
  # Segmentar los tokens de las frases de la noticia 5 (indice 4)
  tokens = word_tokenize(sentence)
  for num_token, token in enumerate(tokens):
    print("El token {} de la frase {} es: {}".format(num_token, num_sen,token))


# ***Spacy***

# En Spacy el funcionamiento es algo distinto:
# En primer lugar es necesario cargar un objeto spacy pre-entrenado proporcionado por los creadores de la librería (o por cualquier otro usuario que lo haya compartido). 
# 
# ![Picture1.png](data:image/png;base64,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)
# Los modelos incorporados con Spacy se pueden encontrar en su[ web](https://spacy.io/usage/facts-figures#benchmarks). Estos objetos permiten procesar documentos completos y extraer información de ellos como los tokens, PoS, o lemmas.
# 
# En primer lugar cargamos el modelo:

# In[ ]:


import spacy
# Cargamos el modelo preentrenado con textos en inglés
nlp=spacy.load('en_core_web_sm' )


# A continuación, vamos a coger el mismo subset de noticias y vamos a aplicar el objeto nlp creado anteriormente a cada uno de los documentos

# In[ ]:


# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]
# Obtener una lista de objetos de tipo spacy procesados por spacy
documento = nlp(subset_noticias[0])
type(documento)
lista_documentos = [nlp(noticia) for noticia in subset_noticias]


# In[ ]:


type(lista_documentos[0])


# 
# Esto ha transformado nuestros textos a objetos de tipo [`spacy.tokens.doc.Doc`](https://spacy.io/api/doc) que tiene una serie de atributos y métodos útiles para nuestro pipeline.
# 
# Como hemos hecho antes, en primer lugar vamos a extraer las frases del documento 4.
# 

# In[ ]:


# Segmentamos el texto de la noticia 4 en oraciones:
for num,sentence in enumerate(lista_documentos[4].sents):
    print('La oración número {} es: \n {}'.format(num, sentence))


# Vamos a mostrar los tokens de cada una de las frases.
# 

# In[ ]:


# Además también podemos dividir cada frase en tokens para la noticia 4:
for num,sentence in enumerate(lista_documentos[4].sents):
    print('La oración {} tiene {} tokens'.format(num, len(sentence)))
    tokens=[word for word in sentence]
    print(tokens)


# ## Unigramas, Bigramas y N-gramas

# En ocasiones, la información proporcionada por un token no es suficiente. 
# 
# Existen palabras que tienen relación con los términos previos y/o posteriores. Desde un punto de vista *naive*, la manera de conseguir el contexto de cada palabra es mediante los n-gramas.
# 
# Los n-gramas son secuencias de n tokens consecutivos provenientes de un texto. La combinación de n-gramas puede proporcionar información sobre la temática de un texto. Generalmente se generan unigramas, que son iguales que los tokens del texto. Los Bigramas, que son combinaciones pareadas de tokens y los trigramas que son triadas de tokens 

# ***NLTK***

# En NLTK los ngrams se consiguen a traves de un método dentro del módulo util de la librería.

# In[ ]:


from nltk.util import ngrams
get_ipython().run_line_magic('pinfo', 'ngrams')


# Vamos a generar una función para crear n-grams de distinto tamaño!

# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
# Función para extraer n-grams de una frase.
def extraer_ngramas(datos, numero):
    # Uso Utilizar la función ngrams para generar ngrams de textos 
    n_grams = ngrams(word_tokenize(datos), numero)
    # Transformo el resultado en una lista
    return [ ' '.join(grams) for grams in n_grams]


# Ahora vamos a generar un conjunto de bigramas, trigramas y 4-gramas de la noticia 4:

# In[ ]:



# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]

# Calculamos los bigramas, trigramas y 4 gramas de la noticia 4
print("Unigramas: ", extraer_ngramas(subset_noticias[4],1))

print("Bigramas: ", extraer_ngramas(subset_noticias[4],2))
print("Trigramas: ", extraer_ngramas(subset_noticias[4],3))
print("4-gramas: ", extraer_ngramas(subset_noticias[4],4))


# ***Spacy***

# Spacy no tiene actualmente integrado esta funcionalidad, así que utilizaremos una librería auxiliar que funciona con sus clases, llamada textacy.
# 
# Primero, importamos las librerías, el modelo en inglés y como antes, procesamos con el modelo los documentos:

# In[ ]:


import spacy
import textacy

# Cargamos el modelo preentrenado con textos en inglés
nlp=spacy.load('en_core_web_sm')

# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]
# Creamos un objeto spacy nlp con los textos para que sea preprocesado con el modelo anterior
nlp_texto = [nlp(texto_to_process) for texto_to_process in subset_noticias]


# Utilizaremos la librería textacy para extraer esta información:

# In[ ]:


# Calculamos los bigramas, trigramas y 4 gramas de la noticia 4
print("Bigramas: ", list(textacy.extract.ngrams(nlp_texto[4],2, min_freq=1, filter_stops = False, filter_punct =False)))
print("Trigramas: ", list(textacy.extract.ngrams(nlp_texto[4],3, min_freq=1, filter_stops = False, filter_punct =False)))
print("4-gramas: ", list(textacy.extract.ngrams(nlp_texto[4],4, min_freq=1, filter_stops = False, filter_punct =False)))



# 
# **Visualización:**
# 
# A continuación, vamos a calcular los tokens y bigramas de todo el corpus de documentos y vamos a generar una visualización.
# 

# In[ ]:


def frecuencia_tokens(lista): 
    # Creamos diccionario vacío 
    frecuencia = {} 
    for item in lista: 
        if (item in frecuencia): 
            frecuencia[item] += 1
        else: 
            frecuencia[item] = 1
    return frecuencia


# Primero extraemos los tokens de todos los textos y los introducimos en una lista común.

# In[ ]:


lista_tokens = list()
for i in subset_noticias:
  tokens_document = word_tokenize(i)
  # Añadimos esos tokens como nuevos elementos
  lista_tokens.extend(tokens_document)


# Después calculamos la frecuencia con la función generada.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculemos la frecuencia\ndict_freq = frecuencia_tokens(lista_tokens)\ndict_freq["Road"]')


# Tambien podemos utilizar un counter (más eficiente):

# In[ ]:


from collections import Counter


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dict_freq2 = Counter(lista_tokens)\ndict_freq2["Road"]')


# Vamos a ordenar el diccionario, para tomar sólo los valores mayores de 20:

# In[ ]:


# Ordenamos el diccionario por la frecuencia de sus palabras
dict_freq_order = sorted(dict_freq.items(), key=lambda x: x[1], reverse=True)
token_names = list()
token_freqs = list()
for i in dict_freq_order:
  if i[1] > 30:
    token_names.append(i[0])
    token_freqs.append(i[1])


# Dibujemos

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
sns_g = sns.barplot(x=token_names, y=token_freqs)
plt.xticks(rotation=45)


# Se podrían quitar los símbolos de puntuación y stopwords con:
# 
# 
# ```
# import string
# from nltk.corpus import stopwords
# punctuations = string.punctuation
# stop_words = stopwords.words('english')
# ```
# 
# 
# 

# ## Lematización y stemming
# 

# Las lemas son las formas canónicas del léxico de un idioma. Por ejemplo, en el caso del español, los verbos presentan una flexión verbal, conocida como comúnmente como conjungación, utilizada para adaptar el verbo a diferentes situaciones de contexto (número, género y tiempo verbal) y presentando distinta forma escrita. En algunas ocasiones, es útil utilizar el lema de los verbos y otras palabras para reducir la dimensionalidad en los modelos predictivos. Este proceso es conocido como lematización. Cuando se lematiza se obtienen palabras reales ya que se utilizan diccionarios jerárquicos para obtener el lema. Este diccionario jerárquico es conocido como WordNet, y será explicado con profundidad más adelante.
# 
# Un caso específico y simple de la lematización es el stemming, que consiste en utilizar reglas sintácticas para quitar la finalización de las palabras y reducirlas así una forma común llamada stem. Hay muchos stemmers populares como el de Porter o el de Snowball. **Es importante mencionar que no siempre que se hace stemming de una palabra esta es una palabra real, si no una palabra sin su última(s) letras**
# 
# A continuación se muestran ests procesos tanto para NLTK como para Spacy:

# ***NLTK***

# **Stemming**
# 
# En NLTK hay varias implementaciones de algoritmos de Stemming. Aquí mostraremos los dos más utilizados: 
# 
# - Algoritmo de Porter Stemming: Algoritmo que solo funciona en inglés y que funciona correctamente con la mayoría de las palabras en ese idioma. Sirve para quitar sustituir los sufijos de las plabras.
# 
# - Algoritmo de SnowballStemmer: Algoritmo de Stemming que soporta 13 lenguas en NLTK, incluyendo español. Es una versión mejorada del algoritmo de de Porter Stemming.
# 

# In[ ]:


from nltk import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

# In English
list_of_english_words = ["Speaking","speaks","Speaker","dogs","buses", "pieces",'compute', 'computer', 'computed', 'computing']
SStemmer = PorterStemmer()


# In[ ]:


from nltk import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

# In English
list_of_english_words = ["Speaking","speaks","Speaker","dogs","buses", "pieces",'compute', 'computer', 'computed', 'computing']
# Cargamos los Stemer
SStemmer = PorterStemmer()
PStemmer = SnowballStemmer("english")
print("Términos en inglés:")
for word in list_of_english_words:
  print("Palabra original: {}, Porter Stemmer: {}, Snowball: {}".format(word,PStemmer.stem(word),SStemmer.stem(word)))


# In[ ]:


#En español
lista_de_palabras_espano = ["Hablando", "Habla", "Hablador", "Hablará", "ha hablado"]

SStemmer_spanish = SnowballStemmer("spanish")
print("Términos en español:")
for word in lista_de_palabras_espano:
  print("Palabra original: {}, Snowball: {}".format(word,SStemmer_spanish.stem(word)))


# **Lematización**

# En español no se puede utilizar este método, dado que WordNet solo tiene términos en inglés. 
# En este ejemplo vamos a lematizar palabras individuales, sin una categoría gramatical asignada. Si tuvieramos la PoS funcionaría con mejor rendimiento.
# 

# In[ ]:


from nltk.stem import WordNetLemmatizer

lematizador = WordNetLemmatizer()
list_of_english_words = ["Speaking","speaks","Speaker","dogs","buses", "pieces",'compute', "computes", 'computer', 'computed', 'computing']
for word in list_of_english_words:
  print(" {} ---> {}".format(word,lematizador.lemmatize(word)))


# Notesé que, a diferencia del stemmer, las palabras en plural que no se forman solo con una s y que se producían errores, aquí lo hace sin problemas.

# ***Spacy***

# Debido al funcionamiento de Spacy, que funciona con modelos pre-entrenados de DeepLearning que incorporan distintas características, no existen funciones para hacer stemming y si para lematizar. Este proceso lo hace a partir de los conocimientos adquiridos en el proceso de entrenamiento del modelo con millones de textos.

# In[ ]:


import spacy

# Cargamos el modelo preentrenado con textos en inglés
nlp=spacy.load('en_core_web_sm')

english_sentence = "I bought five tickets on the internet, after a long wait 5 buses passed by, but none of them was the correct one"

word_sp = nlp(english_sentence)
for word in word_sp:
  print(word.text, "---->", word.lemma_)


# ## Part-Of-Speech Tagging

# El *Part-Of-Speech Tagging* o la asignación de categorías gramaticales a una frase es el proceso en el que a una lista de palabras es etiquetada con su categoría gramatical, es decir que identifica si la palabra es un nombre, un adjetivo, un verbo, un adverbio, etc.
# 
# La asignación de etiquetas gramaticales es interesante cuando se quiere hacer análisis gramatical de una oración, para saber si una palabra tiene una acepción u otra, o incluso para extraer características artificiales cuando se quiere hacer una clasificación textual o similar.
# 

# **NLTK**

# El listado de etiquetas de NLTK es el utilizado por UPenn (University of Pennsylvania). Para ver el listado completo solo hay que ejecutar la siguiente línea de código:
# 
# 
# ```
# nltk.help.upenn_tagset()
# ```
# 
# 

# In[ ]:


nltk.help.upenn_tagset()


# In[ ]:


from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]


# Segmentamos los tokens
tokens = word_tokenize(subset_noticias[4])

# Utilizamos la función pos_tag() de nltk para obtener las etiquetas
pos_tag(tokens)


# **Spacy**

# El listado de etiquetas POS utilizadas en Spacy es el siguiente:
# 
# 
# 
# ```
# SPACY_POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
#                   "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", 
#                   "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
# ```
# 
# Si se necesita recordar que significa una de las abreviaciones se puede utilizar el código:
# 
# ```
# spacy.explain("NN")
# # 'noun, singular or mass'
# ```
# 
# 
# 

# In[ ]:


spacy.explain("NP")


# In[ ]:


# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]


# Segmentamos los tokens
tokens = nlp(subset_noticias[4])
for w in tokens:
    print( "The word '{}' is a {} ".format(w.text, w.pos_))


# ## Named-entiy recognition

# La extración de entidades de un documento es una labor esencial en la análitica de textos. En algunas ocasiones puede ser interesante si se nombra a una persona, a una ciudad, un país o incluso a un medicamento, en el caso de los textos clínicos. 
# 
# Existen sistemas NER (Named-entity recognition) específicos para cada campo de aplicación. Las librerías de NLTK y Spacy disponen de modelos para detectar entidades de ámbito general, aunque existen modelos mucho más espcificos para reconocer entidades muy específicos como por ejemplo síntomas en textos clinicos (mención a BSC).

# **NLTK**

# En NLTK antes de detectar es necesario la obtención de la tokenización y la POS tag antes de identificar entidades, ya que utiliza las etiquetas POS y reglas internas para encontrar que elementos son personas u otro tipo de entidad. 
# 
# Cuando los textos son extraidos de internet, hay que quitar los espacios extras que puede haber en una frase, para que se extraiga mejor las categorías gramaticales de éstos y poder así reconocer mejor las organizaciones o personas en el texto. En este caso están bastante limpios, así que no hace falta hacerlo. 
# 
# Importamos la función `ne_chunk`, que necesita un conjunto de tokens etiquetados PoS previamente. Así que antes hay que preprocesar el documento. Lo haremos con la función `preprocesar()`

# In[ ]:


from nltk.chunk import ne_chunk
get_ipython().run_line_magic('pinfo', 'ne_chunk')


# In[ ]:


from nltk.chunk import ne_chunk
subset_noticias = texto_noticias[0:100]

def preprocess(documento):
  # word_tokenizer
  documento_tok = word_tokenize(documento)
  # pos_tag
  documento_pos = pos_tag(documento_tok)
  return documento_pos


# Ejecutemos la función sobre la noticia 4 y extraigamos las named-entities:
# 
# Primero preprocesamos, que vemos que nos devuelve una lista de tuplas:

# In[ ]:


noticia = preprocess(subset_noticias[4])
noticia[0:5]


# Utilizamos esa salida para observar la presencia de entidades nombradas:

# In[ ]:


ne_tree = ne_chunk(noticia)
print(ne_tree)


# **Spacy**

# En Spacy es mucho más sencillo. Cuando procesamos un documento con el modelo importado, automáticamente se le aplica un conjunto de instrucciones internamente para detectar tokens, lemas... y también las entidades nombradas, a las que se puede acceder iterando sobre el atributo `ents` y extrayendo la etiqueta.

# In[ ]:


# Cogemos un subset de las noticias para acelerar el proceso:
subset_noticias = texto_noticias[0:100]

# Segmentamos los tokens
tokens = nlp("John was born in Chicken, Alaska, and studies at Cranberry Lemon University. John likes to go to Starbucks.")
print([(X.text, X.label_) for X in tokens.ents])


# Además, spacy incorpora un módulo para visualizar estas entidades en un gráfico. 

# In[ ]:


from spacy import displacy
displacy.render(tokens, jupyter=True, style='ent')


# ## Estructura de la frase

# In[ ]:


from spacy import displacy
tokens = nlp("John was born in Chicken, Alaska, and studies at Cranberry Lemon University. John likes to go to Starbucks.")
displacy.render(tokens,style='dep',jupyter=True)


# ## WordNet

# WordNet 3.0 es un diccionario jerárquico desarrollado por la Universidad de Princeton, que categoría las acepciones de todas las palabras del inglés en relaciones semánticas con otras.
# 
# Se puede acceder fácilmente a Wordnet utilizando NLTK mediante la función wordnet del módulo corpus.
# 
# Por ejemplo, busquemos la palabra "bank":

# In[ ]:


from nltk.corpus import wordnet

syn = wordnet.synsets("bank")
print(syn)


# Al mostrar los synsets de esa palbra se observa que hay varios que contienen la palabra bank. 
# Vamos a mostrar la definición y ejemplos de uso de alguno de ellos:
# 
# 

# In[ ]:


print("bank.n.01 definition: " + syn[0].definition())
print(syn[0].examples())

print("bank.n.04 definition: " + syn[5].definition())
print(syn[5].examples())


# Se pueden buscar la distancia exitente entre los synsets para intentar comprender su similitud semántica:

# In[ ]:


from nltk.corpus import wordnet

dog = wordnet.synset('dog.n.01')
cat = wordnet.synset('cat.n.01')
fox = wordnet.synset('fox.n.01')

print("path similarity between dog and cat: ",dog.path_similarity(cat))
print("path similarity between dog and fox: ",dog.path_similarity(fox))
print("path similarity between cat and fox: ",cat.path_similarity(fox))


# ## Embeddings
# 
# Es interesante saber cargar embeddings pre-entrenados. En la clase de mañana utilizaremos embeddings para ayudar a sistemas de análisis de sentimiento, así que vamos a aprender a cargarlos hoy. [LINK](https://nlp.stanford.edu/projects/glove/)

# In[ ]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
# Unzip


# In[ ]:


get_ipython().system('unzip glove.6B.zip')
# Get the path of the zip
get_ipython().system('ls')
get_ipython().system('pwd')


# In[ ]:


# Librerías tpipicas
import pandas as pd
import numpy as np
def cargaGlove(gloveFile):
    print("Cargando modelo Glove")
    f = open(gloveFile,'r')
    modelo = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        modelo[word] = embedding
    print("Finalizado.",len(modelo)," palabras")
    return modelo


# In[ ]:


glove_model = cargaGlove("/content/glove.6B.100d.txt")


# In[ ]:


rana = glove_model["frog"]
print(rana)
lagarto = glove_model["lizard"]
perro = glove_model["dog"]
libro = glove_model["book"]
humano = glove_model["person"]


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([rana],[lagarto]))
print(cosine_similarity([rana],[perro]))
print(cosine_similarity([rana],[libro]))
print(cosine_similarity([humano],[perro]))
print(cosine_similarity([humano],[rana]))


# In[ ]:


print(glove_model["sad"])
print(glove_model["happy"])
