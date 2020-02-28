# method for outlier range
def outlierRange(datacolumn):
  #Sort the data in ascending order
  sorted(datacolumn)
  
  #GET Q1 and Q3
  Q1,Q3 = np.percentile(datacolumn, [25,75])
  
  #Calc IQR
  IQR = Q3 - Q1
  
  #Calc LowerRange
  lr = Q1 - (1.5 * IQR)
  
  #Calc Upper Range
  ur = Q3 + (1.5 * IQR)
  
  return lr, ur

# method for outlier detection
def outlierPresent(datacolumn):
  '''pass a pandas row, will return true if an outlier is present'''
  lr, ur = outlierRange(datacolumn)
  if (datacolumn.min() < lr) | (datacolumn.max() > ur):
      return True
  else:
      return False
