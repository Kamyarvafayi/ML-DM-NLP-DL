import pandas as pd
import numpy as np
import random
data = pd.read_excel(r'C:\Users\totian\Desktop\همکاران جشنواره سال 1400\همکاران ستاد جشنواره پژوهش و فناوری.xlsx')
data['rand'] = 0
data['rand2'] = 0
for i in range(data.shape[0]):    
    data['rand'][i] = random.randint(10,1000) 
    data['rand2'][i] = random.uniform(-100,200)

data['result'] = data['rand'] * data['rand2']
print(data['result'][data['result']<0])
print(data.at[0,'result'])
print(data.iloc[5,4])
print(data.loc[3].at['result'])
print(data.loc[3]['result'])
print(data.iloc[3][-1])
print(np.std(data['result']))
temp = data.cov()
print(data.cov())
# Sorting
data2 = data.sort_values('result',ascending=True)
data3 = data.sort_index()
#Plots
pd.plotting.scatter_matrix(data,diagonal = 'kde')
pd.plotting.boxplot(data)
# Changing Index
data = data.set_index('نام و نام خانوادگی')
# Drop
data.drop(['واحد','شماره ملی'],axis = 1, inplace = True)
data4any = data.dropna(axis=0 , how = 'any')
data4all = data.dropna(axis=0,how = 'all')

print(data[pd.isna(data)])
print(data[~pd.isna(data)])
data5 =  data.drop_duplicates(subset = ["واحد"], keep = 'first', inplace = False)
# Creating series (like a vector in numpy)
a = {"Ford" : 120, "BMW" : 150, "BENZ" : 160}
aserie = pd.Series(a)
print(aserie)
print(aserie.loc["Ford"])
newdataframe = pd.DataFrame([120,150,160],index = a.keys(),columns = ["car"])
# adding a new row
New_row = pd.Series( data = {"car":220}, name = "Lamburgini")
newdataframe = newdataframe.append(New_row, ignore_index = False)
New_rows  = [pd.Series( data = {"car":280}, name = "Lamburgini2"),pd.Series( data = {"car":300}, name = "Lamburgini3")]
newdataframe = newdataframe.append(New_rows, ignore_index = False)
New_rows2  = [pd.Series( data = {"car":5}, name = "Pride"),pd.Series( data = {"car":10}, name = "Paykan")]
newdataframe = newdataframe.append(New_rows2, ignore_index = False)

newdataframe.loc['BMW2'] = 2000
newdataframe.loc['1'] = 5000
newdataframe.loc[2] = 3000
# Renaming an index or a column
newdataframe.rename(index={"BMW2":"BMWW"},inplace = True)
newdataframe.rename(columns={"car":"IranPrice"},inplace = True)
# Filtering
filter1 = newdataframe['IranPrice'][newdataframe['IranPrice']>120]
filter1 = newdataframe.loc[newdataframe.iloc[:,0]>120]
newdataframe['price'] = 0
for i in range(newdataframe.shape[0]):    
    newdataframe['price'][i] = random.randint(10,1000) 
filter2 = newdataframe.loc[newdataframe['IranPrice']> 20]
filter3 = filter2.loc[newdataframe['price']>500]

# using format for creating new indices
for i in range(3):
    newdataframe['results {}'.format(i)] = [random.randint(100,10000) for k in range(11)]
# Scatter matrix
pd.plotting.scatter_matrix(newdataframe,diagonal = 'hist')
