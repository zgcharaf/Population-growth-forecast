import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


os.chdir('/Users/lucasrouleau/Downloads')

sweden = pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month Sweden (2000-2023) SCB.xlsx",header=None)
japan = pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month Japan(1995-2023) e-stat.xlsx",header=None)
france=pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month France (1975-2023) INSEE.xlsx",header=None)
usa=pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month USA (1959-2023) FRED.xlsx",header=None)
columbia= pd.read_excel("/Users/lucasrouleau/Downloads/colombia.xlsx", header=None)


#Data cleaning Sweden
sweden=sweden.T

#Supprimer la premiere ligne inutile 
sweden= sweden.iloc[1:]

# Mettre les dates au format datetime et en index
sweden.iloc[:, 0] = pd.to_datetime(sweden.iloc[:, 0])
sweden.set_index(sweden.columns[0], inplace=True)

#Renommer la 1ère colonne 
sweden = sweden.rename(columns={sweden.columns[0]: 'Sweden'})

# Convertir  en float
sweden=sweden.astype(float)

#Diviser par 1 000 000
sweden= sweden / 1000000



#Data cleaning Columbia
#Supprimer la premiere ligne inutile 
columbia=columbia.iloc[1:]

#Inverser la dataframe
columbia = columbia.iloc[::-1]

# Mettre les dates au format datetime et en index
columbia.iloc[:, 0] = pd.to_datetime(columbia.iloc[:, 0])
columbia.set_index(columbia.columns[0], inplace=True)

#Renommer la 1ère colonne 
columbia = columbia.rename(columns={columbia.columns[0]: 'Columbia'})

# Convertir toutes les colonnes en float
columbia=columbia.astype(float)

#Diviser par 1 000 000
columbia= columbia / 1000000



#Data cleaning Japan
japan=japan.T

# Supprimer la première ligne 
japan= japan.iloc[1:]

# Mettre les dates au format datetime et en index
japan.iloc[:, 0] = pd.to_datetime(japan.iloc[:, 0])
japan.set_index(japan.columns[0], inplace=True)

# Renommer l'unique colonne du DataFrame "japan" en "Japan"
japan = japan.rename(columns={japan.columns[0]: 'Japan'})

# Convertir en float
japan=japan.astype(float)

# Diviser toutes les valeurs par 1 000 000
japan= japan / 1000000



#Data cleaning France
france= france.T

# Supprimer la première ligne 
france= france.iloc[1:]
#Inverser le dataframe pour avoir des données pour des dates croissantes
france = france.iloc[::-1]

# Mettre les dates au format datetime et en index
france.iloc[:, 0] = pd.to_datetime(france.iloc[:, 0])
france.set_index(france.columns[0], inplace=True)

#Renommer la 1ère colonne 
france = france.rename(columns={france.columns[0]: 'France'})
# Convertir en float
france=france.astype(float)

# Diviser toutes les valeurs par 1000
france = france / 1000



#Data cleaning USA
usa=usa.T

# Supprimer la première ligne 
usa= usa.iloc[1:]

# Mettre les dates au format datetime et en index
usa.iloc[:, 0] = pd.to_datetime(usa.iloc[:, 0])
usa.set_index(usa.columns[0], inplace=True)

#Renommer la 1ère colonne 
usa = usa.rename(columns={usa.columns[0]: 'Usa'})

# Convertir en float
usa=usa.astype(float)

# Diviser toutes les valeurs par 1000
usa= usa / 1000


#Nous utilisons les données communes pour tous les pays  entre 2001-01-01 et 2023-05-01
japan = japan['2001-01-01':'2023-05-01']
columbia= columbia['2001-01-01':'2023-05-01']
france = france['2001-01-01':'2023-05-01']
sweden = sweden['2001-01-01':'2023-05-01']
usa = usa['2001-01-01':'2023-05-01']


#fusionner les dataframes 
dataframes_to_add = [france, usa, columbia, sweden]

# Concaténer les colonnes des DataFrames
countries  = pd.concat([japan] + dataframes_to_add, axis=1)


#exporter  Excel
export_path = "population_data_bis.xlsx"
countries.reset_index().to_excel(export_path, index=False)

#Descriptive statistics 
print(countries.describe())

# Plot 
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(countries.index, countries['France'], label='France')
plt.plot(countries.index, countries['Usa'], label='Usa')
plt.plot(countries.index, countries['Columbia'], label='Columbia')
plt.plot(countries.index, countries['Sweden'], label='Sweden')
plt.plot(countries.index, countries['Japan'], label='Japan')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Population (millions)')
plt.title('Monthly Population Data')
plt.legend()  # Add legend

# Show the plot
plt.grid(True)  # Add grid lines
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()







#Créer un df growth_rate pour calculer le taux de croissance mensuel de la population 
growth_rate= countries.copy()


#Créer les colonnes avec les taux de croissance: 

payss =['France', 'Usa', 'Columbia','Sweden', 'Japan']

for pays in payss:
    growth_rate[f'{pays}_Growth'] = growth_rate[pays].pct_change() * 100


growth_rate['Average_Growth'] = growth_rate[[f'{pays}_Growth' for pays in payss]].mean(axis=1)


# Supprimez les colonnes contenant les nombres d'habitants pour chaque pays
columns_to_drop = ['France', 'Usa', 'Columbia', 'Sweden', 'Japan']
growth_rate.drop(columns=columns_to_drop, inplace=True)


# Supprimer la première ligne 
growth_rate= growth_rate.iloc[1:]


#Statistiques descriptives 
print(growth_rate.describe())



#Créer un nouveau df 'avg' contenant uniquement les moyennes des taux de croissance des pays 
data = {
    'France_Growth': growth_rate['France_Growth'].mean(),
    'Usa_Growth': growth_rate['Usa_Growth'].mean(),
    'Columbia_Growth': growth_rate['Columbia_Growth'].mean(),
    'Sweden_Growth': growth_rate['Sweden_Growth'].mean(), 
    'Japan_Growth': growth_rate['Japan_Growth'].mean()
}

avg = pd.DataFrame(data, index=[0])

# Tracer le graphique à barres
plt.figure(figsize=(10, 6))
ax = avg.plot(kind='bar', legend=True)
ax.set_title('Average Monthly Population Growth Rate')  # Titre du graphique
#ax.set_xlabel('Countries')  # Titre pour l'axe des abscisses
ax.set_ylabel('Average Rate')  # Titre pour l'axe des ordonnées
ax.set_xticklabels(['Countries'], rotation=0)  # Définir les étiquettes de l'axe X
ax.axhline(0, color='black', linewidth=1.5)  # Axe horizontal au niveau 0

plt.tight_layout()  # Ajustement du layout

plt.show()


