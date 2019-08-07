from classifier import AccountClassifier

def test_categories():
  learn = AccountClassifier()
  assert all([a == b for a, b in zip(learn.data_clas.y.classes, ['adjustment', 'd_and_a', 'other'])])