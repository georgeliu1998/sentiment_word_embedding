import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt



def preprocess_text(text, lemmatizing=False):
    """
    Preprocesses text by removing all stop words and lemmatizing.
    
    Params
    --------------
    text: str
      the string to be cleaned
    
    Returns
    --------------
    text: str
      the cleaned string

    """
    text = text.lower()
    
    if lemmatizing:
      lemmatizer = WordNetLemmatizer()
      text = ' '.join([lemmatizer.lemmatize(word, pos='v') for word in text.split()])
    
    # Remove html tags
    text = re.sub(r'<.*?>', '', text)
    
    # Replace punctuation with spaces
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    
    # Remove additional white spaces
    text = ' '.join(text.split())
    
    return text



def get_vocab(df):
  """
  Gets the vocabulary in the review column and prints the size of the vocab
  
  Params
  ----------------------
  df: pandas dataframe
    The input dataframe
  
  Returns
  ----------------------
  vocab: Python set
    The unique vocabulary
  
  """
  vocab = set()
  
  for i in df['review_processed'].str.split():
    vocab.update(i)
  
  print("Vocabulary size: {}".format(len(vocab)))
  return vocab


def get_vocab_frequency(df):
  """
  
  """
  return df['review_processed'].str.split(expand=True).stack().value_counts()



  def plot_metrics(history):
  """
  Plots the accuracy and loss results using the given hisotry_dict.
  
  Params
  ---------------------
  history:
    The history object by model.fit()
  
  Returns
  ---------------------
  None
  
  """
  history_dict = history.history
  
  acc = history_dict['acc']
  val_acc = history_dict['val_acc']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
  
  # "bo" is for "blue dot"
  ax1.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  ax1.plot(epochs, val_loss, 'b', label='Validation loss')
  ax1.set_title('Training and validation loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.legend()

  ax2.plot(epochs, acc, 'bo', label='Training acc')
  ax2.plot(epochs, val_acc, 'b', label='Validation acc')
  ax2.set_title('Training and validation accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.legend()
  
  plt.show() 