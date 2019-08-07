from fastai.text import * 
from pathlib import Path
import os 

class AccountClassifier():
  """A model to classify a short statement about an account into one of the following categories:    
   * Depreciation and amortisation
   * An adjusting item 
   * Can be ignored
   
  Attributes: 
      path (string): Path to your data. Default='/data/'
      fname (string): Data filename in csv format. Default='sample.csv'
      header (boolean): See Pandas.read_csv
      text_cols (int): Position of text columns 
      label_cols (int): Position of label columns
      
  Methods:
    train(self)
      Function to train the model and save it into models directory
  
    predict(self, account_details)
      Classifies the account detail.
      Returns:
        Category (string), Confidence (float)
  """
  
  def __init__(self, data_path='data', fname='sample.csv', header=None, text_cols=0, label_cols=1):
    self.data_path = data_path
    args = (Path(data_path), fname)
    kwargs = {"header":header, "text_cols":text_cols, "label_cols":label_cols}
    self.data_lm = TextLMDataBunch.from_csv(*args, **kwargs)
    self.data_clas = TextClasDataBunch.from_csv(*args, **kwargs,
                                                vocab=self.data_lm.train_ds.vocab,
                                                bs=32)

  def train(self):
    self.train_lm()
    self.train_classifier()
    
  def train_lm(self):
    learn = language_model_learner(self.data_lm, AWD_LSTM, drop_mult=0.5)
    learn.fit_one_cycle(10, 1e-2)
    learn.unfreeze()
    learn.fit_one_cycle(5, 1e-3)
    learn.save_encoder('ft_enc')
   
  def train_classifier(self):
    learn = text_classifier_learner(self.data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder('ft_enc')
    learn.fit_one_cycle(1, 1e-2)
    learn.freeze_to(-2)
    learn.fit_one_cycle(5, slice(5e-3/2., 5e-3))
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(2e-3/100, 2e-3))
    learn.export('models/export.pkl')
    self.classifier = learn
  
  def predict(self, account_details):
    """  
    Takes as input a short statement about an account and classifies it 
    into one of the following categories:    
     * Depreciation and amortisation
     * An adjusting item 
     * Can be ignored
    """
    try:
      self.classifier = load_learner(Path(f'{self.data_path}/models/'))
    except:
      print(f'Weights file export.pkl not found in models. Training model')
      self.train()
    pred, index, probs = self.classifier.predict(account_details)
    return pred.obj, float(probs[index])*100