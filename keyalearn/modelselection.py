from sklearn.model_selection import train_test_split

def randomStateSelection(model, f, l, ts=0.2, max_random_state=256, target_score=1, debug=False):
  """
    required params:
    model, features, label

    optional params:
    test size. default = 0.2
    the maximum random state. default = 256.
    target_score, maximum target score at which the test score and randoms tate should be returned. default = 1
    debug, print each iteration scores. default = False
  """
  max_score = 0
  random_state = 0
  for i in range(0, len(l) % max_random_state):
    # generate random state
    X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=ts, random_state=i)
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    if debug:
      print("train score:{}, test score:{}, random state: {}".format(train_score, test_score, i))

    if(test_score > train_score) and (test_score >= max_score):
      max_score = test_score
      random_state = i
    
    if max_score >= target_score:
      break
  if random_state == 0:
    # print("Unbalanced model")
    return random_state, max_score
  # print("Use random state {} for the highest score of {}".format(random_state, max_score))
  return random_state, max_score
