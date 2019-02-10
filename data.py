import os
import numpy as np
import pandas as pd

np.random.seed(8)
seed = np.random.RandomState(8)

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


workingdir = '/content/gdrive/My Drive/WorkingDir/sentiment_word_embedding'
cwd = os.getcwd()

if cwd != workingdir:
  os.chdir(workingdir)



def download_imdb():
  """
  Downloads and extracts the IMDB dataset  
  """

  url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

  download_folder = 'raw_data'
  download_name = 'imdb.tar.gz'
  path_name = os.path.join(workingdir, download_folder, download_name)

  # Download data to the specified path
  urllib.request.urlretrieve(url, path_name)

  #extract_path = os.path.join(path, download_folder)

  with tarfile.open(path_name) as tar:
    #tar.extractall(path=extract_path)
    tar.extractall()


def load_imdb(path):
  """
  Loads train and test data into dataframes.

  Params
  ---------------------
  path: str
    The path to the unzipped aclImdb folder.

  Returns
  ---------------------
  df_train, df_test: tuple of of pandas df
    The dataframes created from data
  """

  data = {}

  for split in ['train', 'test']:
    data[split] = []

    for label in ['pos', 'neg']:
      sentiment = 1 if label == 'pos' else 0
      file_names = os.listdir(os.path.join(path, split, label))

      for file_name in file_names:
        file_path = os.path.join(path, split, label, file_name)
        with open(file_path, "r") as f:
          review = f.read()

          data[split].append([review, sentiment])

  np.random.shuffle(data['train'])        
  df_train = pd.DataFrame(data['train'], columns=['review', 'sentiment'])

  np.random.shuffle(data['test'])
  df_test = pd.DataFrame(data['test'], columns=['review', 'sentiment'])

  return df_train, df_test


def save_df(df, path, save_name):
  file_name = save_name + '.csv'
  path_name = os.path.join(path, file_name)
  
  df.to_csv(path_name, index=False)


def load_df(frac=1.0):
  """
  Loads train test dataframes
  
  Params
  ----------
  frac: float
    The percentage of random samples to return
  path: str
    The path of workingdir
  
  Returns
  ----------
  df_train, df_test: pandas df
      
  """
  df_train = pd.read_csv('./data/df_train.csv')
  df_test = pd.read_csv('./data/df_test.csv')
  
  df_train = df_train.sample(frac=frac, random_state=seed)
  df_test = df_test.sample(frac=frac, random_state=seed)
  
  return df_train, df_test 


def load_data_embedded():
  """
  Loads the spaCy embedded
  """
  path = './gdrive/My Drive/WorkingDir/sentiment_word_embedding'
  
  X_train = np.loadtxt(os.path.join(path, 'X_train_vector.csv'), delimiter=",")
  X_test = np.loadtxt(os.path.join(path, 'X_test_vector.csv'), delimiter=",")
  
  y_train, y_test = df_train['sentiment'], df_test['sentiment']

  return X_train, y_train, X_test, y_test


def vectorize(col, vocab_size, max_length):
  """
  Vectorizes input texts by limiting to max length, applying Keras one-hot encoding and padding
  
  Params
  ----------------------
  col: Pandas series
    The df column to vectorize
  vocab_size: int
    Vocabulary size to use
  max_length: int
    Max length of reviews to use
  
  Returns
  ----------------------
  X: Numpy array  
    The vectorized feature column
  """
  # Keep only up to max length for each review
  col = col.apply(lambda x: ' '.join(x.split()[:max_length]))
  # Vectorize with one-hot encoding 
  col_oh = col.apply(lambda x: one_hot(text=x, n=vocab_size))
  # Pad reviews shorter than max_length with 
  X = pad_sequences(col_oh, maxlen=max_length, padding='post')
 
  return X


def load_split_data(df_train, df_test, 
                    col_name='review_processed', 
                    val_split=0.2, 
                    vocab_size=10000, 
                    max_length=200):
  """
  Loads the train, validation and test data as numpy arrays
  """
  X_train = vectorize(col=df_train[col_name], 
                      vocab_size=vocab_size,
                      max_length=max_length)
  y_train = df_train['sentiment']
  
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=val_split, 
                                                    random_state=seed)
  
  X_test = vectorize(col=df_test[col_name], 
                     vocab_size=vocab_size,
                     max_length=max_length)
  y_test = df_test['sentiment']
  
  return X_train, y_train, X_val, y_val, X_test, y_test 