import contractions
import re
import numpy as np
import random

from sklearn.model_selection import train_test_split

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def preprocess(q):
  q = str(q).lower().strip()
  q = contractions.fix(q) ## correcting the contractions
  q = re.sub(r"https?://\S+|www\.\S+", "", q) ## remove the urls from string
  html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")  ##removing the html tags
  q = re.sub(html, "", q)
  #replace certain special characters
  q = q.replace('%', ' percent')
  q = q.replace('$', ' dollar')
  q = q.replace('₹', ' rupee ')
  q = q.replace('@', ' at')
  q = q.replace('€', ' euro')
  q = q.replace('[math]','')
  q = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", q) ##puntuation
  return q

def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


