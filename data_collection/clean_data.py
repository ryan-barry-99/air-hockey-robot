import pandas as pd
import numpy as np
 
# reading csv file 
'''df = pd.read_csv("data_to_clean.csv")
print(df.head)
df = df[df['Puck_XH'].notna()]
df = df[df['Table_XH'].notna()]
print(df.head)
df.to_csv(path_or_buf='cleaned_data.csv',index=False)'''

df = pd.read_csv("cleaned_data.csv")
print(df.head)
print(df['Puck_cen_Y'][0])
prev_left=0
prev_right=0
for i in range(df.shape[0]-1):
   #direction
   if df['Puck_cen_X'][i] < df['Puck_cen_X'][i+1]:
      df['direction'][i] = 1
   elif df['Puck_cen_X'][i] > df['Puck_cen_X'][i+1]:
      df['direction'][i] = -1

   # future bounce
   if df['Puck_cen_X'][i] == 0:
      for j in range(prev_left, (i+1)):
         df['Cross_Left'][j] = df['Puck_cen_Y'][i]
      prev_left = i

   elif df['Puck_cen_X'][i] > 0 and df['Puck_cen_X'][i+1] < 0:
      for j in range(prev_left, (i+1)):
         weighted_avg = df['Puck_cen_X'][i] / (df['Puck_cen_X'][i] + df['Puck_cen_X'][i+1]) # i+1 is negative
         cross =(1-weighted_avg) * df['Puck_cen_Y'][i] + weighted_avg * df['Puck_cen_Y'][i+1]
         df['Cross_Left'][j] = cross
      prev_left = i
 
   elif df['Puck_cen_X'][i] == 1:
      for j in range(prev_right, (i+1)):
         df['Cross_right'][j] = df['Puck_cen_Y'][i]
      prev_right = i

   elif df['Puck_cen_X'][i] < 1 and df['Puck_cen_X'][i+1] > 1:
      for j in range(prev_right, (i+1)):
         weighted_avg = (1-df['Puck_cen_X'][i]) / (df['Puck_cen_X'][i+1] - df['Puck_cen_X'][i]) # where is 1 with respect to i and i+1
         cross = (1-weighted_avg) * df['Puck_cen_Y'][i] + weighted_avg * df['Puck_cen_Y'][i+1]
         df['Cross_right'][j] = cross
      prev_right = i

df.to_csv(path_or_buf='cleaned_data.csv',index=False)
print(df.head)