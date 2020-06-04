# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dataset = pandas.read_csv('Набор_2_цены_квартир.csv' , delimiter = ';' )
dataset.head(10)

print(dataset.shape)

preparedData = pandas.DataFrame()

# Столбец price
data = dataset['price']

plt.hist(data , bins = 50 )
plt.show()

# Выброс в районе 750 
data = numpy.clip(data , 0 , 500)

plt.hist(data , bins = 50 )
plt.show()

# Не очень равномерное распределение 

plt.hist(numpy.log(data) , bins = 50)
plt.show()

plt.hist(data ** 0.5 , bins = 50 )
plt.show()

data = numpy.log(data)

# Теперь данные имеют следующую область значений
print(numpy.min(data))
print(numpy.max(data))

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data , bins = 50)
plt.show()

plt.plot(data)
plt.show()

print(data.ndim)

data = data.flatten()

preparedData['price'] = data

# Столбец totsp
data = dataset['totsp']

plt.hist(data, bins = 50)
plt.show()

data = numpy.clip (data , 0 , 160)

plt.hist(data , bins = 50)
plt.show()

plt.hist(numpy.log(data), bins = 50)
plt.show()

plt.hist(data ** 0.5, bins = 50)
plt.show()



data = data ** 0.5

# Теперь данные имеют следующую область значений
print(numpy.min(data))
print(numpy.max(data))

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data , bins = 50 )
plt.show()

plt.plot(data)
plt.show()

print(data.ndim)

data = data.flatten()

preparedData['totsp'] = data

# Столбец livesp
data = dataset['livesp']

plt.hist(data , bins = 50 )
plt.show()

data = numpy.clip(data , 0 , 90)
plt.hist(data , bins = 50 )
plt.show()

plt.hist(numpy.log(data), bins = 50)
plt.show()

plt.hist(data ** 0.5, bins = 50)
plt.show()

data = numpy.log(data)

print(numpy.min(data))
print(numpy.max(data))

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data, bins = 50)
plt.show()

plt.plot(data)
plt.show()

print(data.ndim)

data = data.flatten()
preparedData['livesp'] = data

# Столбец kitsp
data = dataset['kitsp']

plt.hist(data , bins = 50 )
plt.show()

plt.hist(numpy.log(data), bins = 50)
plt.show()

plt.hist(data ** 0.5, bins = 50)
plt.show()

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data, bins = 50)
plt.show()

plt.plot(data)
plt.show()

data = data.flatten()

preparedData['kitsp'] = data

# Обработка dist 
data = dataset['dist']

plt.hist(data , bins = 50 )
plt.show()

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data, bins = 50)
plt.show()

plt.plot(data)
plt.show()

print(data.ndim)

data = data.flatten()

preparedData['dist'] = data

# Столбец metrdist
data = dataset['metrdist']

plt.hist(data , bins = 50 )
plt.show()

data = numpy.clip(data , 0 , 17.5)

plt.hist(numpy.log(data), bins = 50)
plt.show()

plt.hist(data ** 0.5, bins = 50)
plt.show()

data = data ** 0.5 

scaler = MinMaxScaler()
data = numpy.array(data).reshape(-1,1)
data = scaler.fit_transform(data)

plt.hist(data, bins = 50)
plt.show()

plt.plot(data)
plt.show()

print(data.ndim)\

data = data.flatten()

preparedData['metrdist'] = data

# Столбец walk brick floor
preparedData['walk'] = dataset['walk']
preparedData['brick'] = dataset['brick']
preparedData['floor'] = dataset['floor']

# Столбец code
data = dataset['code']

data = pandas.get_dummies(data)

preparedData = preparedData.join(data)

preparedData.head(10)

preparedData.to_csv('prepared_data.csv')

