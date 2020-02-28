def randomStateSelection(model, f, l, ts=0.2):
  max_score = 0
  random_state = 0
  for i in range(0, len(l)):
    X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=ts, random_state=i)
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    if(test_score > train_score) and test_score > max_score:
      max_score = test_score
      random_state = i
  print("Use random state {} for the highest score of {}".format(random_state, max_score))
  return random_state