#!/usr/bin/env python
# coding: utf-8

# # Анализ данных и кластеризация покупателей.

# ### Контекст
# ##### Набор данных содержит все транзакции произошедшие с 01/12/2010 по 09/12/2011
# у Британской компании продающей подарки. Клиенты - оптовики.
# ### Бизнес-цель
# ##### Мы будем использовать международный набор данных для розничной торговли онлайн,
# чтобы построить RFM-кластер и выбрать лучший набор клиентов, на которых должна ориентироваться компания.

# ### Переменные:
# InvoiceNo - номер счёта
# StockCode - торговый код
# Description - описание товара
# Quantity - количество
# InvoiceDate - дата выставления счёта
# UnitPrice - цена товара
# CustomerID - ID покупателя
# Country - страна

pip install patool
import patoolib
patoolib.extract_archive("../data/OnlineRetail.rar", outdir="../data/")

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore') # отключит предупреждения Anaconda
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # для чётких графиков в svg")

import sklearn
from sklearn.preprocessing import StandardScaler # приближение значений к единице
from sklearn.cluster import KMeans # метод ближайших соседей для класстеризации
from sklearn.metrics import silhouette_score # силуэт - внут. метрика, значения будут в [-1, 1]

from scipy import stats
from scipy.cluster.hierarchy import linkage # выполняет иерархическую класстеризацию
from scipy.cluster.hierarchy import cut_tree # обрубает лишние листья в дереве
from pylab import rcParams
rcParams['figure.figsize'] = 8,5 # увлеличить размер графиков

# Считывание данных и вывод нескольких строк.
df = pd.read_csv('../data/OnlineRetail.csv', encoding = "ISO-8859-1")

# Уберем строки где нет значения null
df = df.dropna()

# Преобразование строки InvoiceDate в формат Datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

# Description можно убрать
df = df.drop(['Description'], axis=1)

# Заменяем символы кода в число, т.е. 3565A -> 356510
unique_list = []
for string in df['StockCode']:
    for char in string:
        if char not in unique_list:
            unique_list.append(char)
print(len(unique_list))
unique_list.sort()

mapping = {char: str(i) for i, char in enumerate(unique_list[1:])}
mapping.update({' ': "50"})

df['StockCode'] = df['StockCode'].replace(to_replace=mapping, regex=True)
# изменяем тип object на float64
df['StockCode'] = df['StockCode'].astype('float64')

# тоже самое и для номера
unique_list = []
for string in df['InvoiceNo']:
    for char in string:
        if char not in unique_list:
            unique_list.append(char)
print(len(unique_list))
unique_list.sort()

mapping = {char: str(i) for i, char in enumerate(unique_list)}

df['InvoiceNo'] = df['InvoiceNo'].replace(to_replace=mapping, regex=True)
df['InvoiceNo'] = df.InvoiceNo.astype('float64')

np.random.seed(seed=42)
# Добавим 3 переменные, используя модель RFM(давность, частота, деньги))
#R (Recency(Давность)): Количество дней с последнего заказа
#F (Frequency(Частота)): Количество транзакций
#M (Monetary(Деньги)): Общая выручка


# Полученные  деньги с заказа
df['amount'] = df['Quantity']*df['UnitPrice']

# Выручка для каждого клиента
grouped_df = df.groupby('CustomerID')['amount'].sum()
grouped_df = grouped_df.reset_index()

# Количество транзакций у каждого клиента
frequency = df.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID', 'frequency']

# Соединяем в один датасет
grouped_df = pd.merge(grouped_df, frequency, on='CustomerID', how='inner')

# Разница между последним заказом и заказом из данных
df['diff'] = max(df['InvoiceDate']) - df['InvoiceDate']

# Давность сделки
last_purchase = df.groupby('CustomerID')['diff'].min()
last_purchase = last_purchase.reset_index()

# Соединяем в ещё больший датасет
grouped_df = pd.merge(grouped_df, last_purchase, on='CustomerID', how='inner')
grouped_df.columns = ['CustomerID', 'amount', 'frequency', 'recency']

# Только количество дней
grouped_df['recency'] = grouped_df['recency'].dt.days

# Убираем выбросы у amount
Q1 = grouped_df.amount.quantile(0.05)
Q3 = grouped_df.amount.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.amount >= Q1 - 1.5*IQR) & (grouped_df.amount <= Q3 + 1.5*IQR)]

# Убираем выбросы у recency
Q1 = grouped_df.recency.quantile(0.05)
Q3 = grouped_df.recency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.recency >= Q1 - 1.5*IQR) & (grouped_df.recency <= Q3 + 1.5*IQR)]

# Убираем выбросы у frequency
Q1 = grouped_df.frequency.quantile(0.05)
Q3 = grouped_df.frequency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.frequency >= Q1 - 1.5*IQR) & (grouped_df.frequency <= Q3 + 1.5*IQR)]

# Приближаем значения к единице
rfm_df = grouped_df[['amount', 'frequency', 'recency']]

scaler = StandardScaler()

rfm_df_scaled = scaler.fit_transform(rfm_df)

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['amount', 'frequency', 'recency']

# модель со случайным k
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)

# Определим наилучшее количество кластеров
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)

# анализ силуэта(метрики кластеринга)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

# Финальная модель с k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)

# Присвоим значения
grouped_df['cluster_id'] = kmeans.labels_# Используем иерархическую полную класстеризацию.

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
grouped_df['cluster_labels'] = cluster_labelscluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
# ### Вывод
# С помощью RFM-анализа можно судить о типе маркетинга, который следует использовать для нацеливания на
# клиентские сегменты и для перемещения их между сегментами.
# Частота и денежные значения расчитывают пожизненную ценность клиента (CLV);
# Новизна может показать вернётся ли клиент скова.
# Получили 3 кластера где:
# 2-Лучший покупатель, т.е высокая лояльность, большое кол-во покупок и их частота
# 1-Пассивный клиент - нечастый клиент с низкой суммой покупок и их частотой
# 0-Потенциальный покупатель: покупатель недавно совершил покупку несколько раз


