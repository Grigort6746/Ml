import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import csv
from sklearn import preprocessing

# df = pd.read_csv('main4.csv')
# print(df.describe())
# print(df.info)
#
# # убираем значения -1
# df = df.replace(-1, np.nan)
# df = df.replace("-1", np.nan)
# df = df.replace(-1.0, np.nan)
# df = df.replace("-1.0", np.nan)
#
# # удаляем строки с пропущенными значениями
# df = df.dropna(subset=['street'])
#
# # меняем тип данных где нужно
# df['floor'] = df['floor'].astype(int)
# df['floors_count'] = df['floors_count'].astype(int)
# df['total_meters'] = df['total_meters'].astype(float)
#
# # удаляем не нужные столбцы
# df.drop([
#     'house_number',
#     'author_type',
#     'phone',
#     'deal_type',
#     'accommodation_type',
#     'object_type',
#     'heating_type',
#     'house_material_type',
#     'finish_type',
#     'author',
#     'url'],
#     axis=1, inplace=True)
#
# # визуализируем пропущенные значения
# sns.heatmap(df.isnull(), cmap='viridis')
# plt.show()
# #
# # убираем лишние символы
# df['living_meters'] = df['living_meters'].replace(to_replace=' м²', value='')
#
# # заполняем пропущенные значения медианой
# df['living_meters'] = df['living_meters'].fillna(df['living_meters'].median())
# df['kitchen_meters'] = df['kitchen_meters'].fillna(df['kitchen_meters'].median())
#
# # заполняем недостающие значения района значением локации
# df['temp_district'] = df['district']
# msk_condition = (df['location'] == 'Москва') & df['district'].isna()
# df.loc[msk_condition & df['underground'].notna(), 'temp_district'] = df['underground']
# df.loc[msk_condition & df['underground'].isna(), 'temp_district'] = 'Москва'
# other_condition = df['district'].isna() & (df['location'] != 'Москва')
# df.loc[other_condition, 'temp_district'] = df['location']
# df['district'] = df['temp_district']
# df.drop(columns=['temp_district'], inplace=True)
#
# # меняем тип данных
# df['total_meters'] = pd.to_numeric(df['total_meters'], errors='coerce').astype('float64')
#
# # создаём новый файл в который запишем среднюю цену за квадратный метр по городам
# dict_city = df['location'].unique()
#
#
# def price_for_meter(location):
#     city = df[df['location'] == location]
#     price_for_city = city['price'].sum()
#
#     clean_data = city['total_meters'].sum()
#
#     return round(price_for_city / clean_data, 2)
#
#
# with open('info.csv', 'w', newline='', encoding='UTF-8') as csvfile:
#     names = ['city', 'price_for_meter']
#     writer = csv.DictWriter(csvfile, fieldnames=names)
#     writer.writeheader()
#     for city in dict_city:
#         writer.writerow({'city': city, 'price_for_meter': price_for_meter(city)})
#
# # отсортируем по возрастанию
# info = pd.read_csv('info.csv')
# info_sorted = info.sort_values(by='price_for_meter', ascending=False)
# info_sorted.to_csv("info.csv", index=False)
#
# визуализируем Цена за м² по городам
sns.set_style("darkgrid")
info = pd.read_csv('info.csv')
plt.figure(figsize=(24, 8))
sns.barplot(hue='city', legend=False, x='city', y='price_for_meter', data=info, color='Blue')
plt.title('Цена за м² по городам')
plt.xlabel('Города')
plt.ylabel('Цена за м²')
plt.xticks(rotation=110, ha='right')
plt.show()
#
# df['price_per_meter'] = df['price'] / df['total_meters']
#
#
# # функция, которая принимает на вход наши данные, кодирует числовыми значениями категориальные признаки
# # и возвращает обновленный данные и сами кодировщики
#
#
# def number_encode_features(init_df):
#     result = init_df.copy() #копируем нашу исходную таблицу
#     encoders = {}
#     for column in result.columns:
#         if result.dtypes[column] == object: # np.object -- строковый тип / если тип столбца - строка, то нужно его закодировать
#             encoders[column] = preprocessing.LabelEncoder() #для колонки column создаем кодировщик
#             result[column] = encoders[column].fit_transform(result[column]) #применяем кодировщик к столбцу и перезаписываем столбец
#     return result, encoders
#
#
# encoded_data, encoders = number_encode_features(df) #Теперь encoded data содержит закодированные категориальные признаки
# print(encoded_data.head()) #проверяем
#
# encoded_data.to_csv("main5.csv", index=False)
#
#
# df = pd.read_csv('main5.csv')
#
# #
# #выведем матрицу кореляции
# temp3 = df.copy()
# corr = temp3.corr()
# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(18, 11))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# plt.show()
#
# plt.figure()
# sns.catplot(x='year_of_construction', y='price', data=df, kind='point', aspect=5, color='orange')
# plt.title("Влияние количества комнат на рост цены")
# plt.xlabel("Количество комнат")
# plt.ylabel("Цена")
# plt.show()
#
# df.to_csv("main4.csv", index=False)
