import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd 


os.chdir('/Users/lucasrouleau/Downloads')

sweden = pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month Sweden (2000-2023) SCB.xlsx",header=None)
japan = pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month Japan(1995-2023) e-stat.xlsx",header=None)
france=pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month France (1975-2023) INSEE.xlsx",header=None)
usa=pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month USA (1959-2023) FRED.xlsx",header=None)
columbia= pd.read_excel("/Users/lucasrouleau/Downloads/colombia.xlsx", header=None)
world= pd.read_excel("/Users/lucasrouleau/Downloads/Pop_month World(2000-2023).xlsx", header=None) 


#Data cleaning Sweden
sweden=sweden.T

#Delete the first unnecessary line
sweden= sweden.iloc[1:]

# Put dates in datetime and index format
sweden.iloc[:, 0] = pd.to_datetime(sweden.iloc[:, 0])
sweden.set_index(sweden.columns[0], inplace=True)

#Rename 1st column
sweden = sweden.rename(columns={sweden.columns[0]: 'Sweden'})

# Convert to float
sweden=sweden.astype(float)

#Divide by 1000000
sweden= sweden / 1000000



#Data cleaning Columbia
#Delete the first unnecessary line
columbia=columbia.iloc[1:]

#Invert the dataframe
columbia = columbia.iloc[::-1]

# Put dates in datetime and index format
columbia.iloc[:, 0] = pd.to_datetime(columbia.iloc[:, 0])
columbia.set_index(columbia.columns[0], inplace=True)

#Rename 1st column
columbia = columbia.rename(columns={columbia.columns[0]: 'Columbia'})

# Convert to float
columbia=columbia.astype(float)

#Divide by 1000,000
columbia= columbia / 1000000



#Data cleaning Japan
japan=japan.T

#Delete the first unnecessary line 
japan= japan.iloc[1:]

# Put dates in datetime and index format
japan.iloc[:, 0] = pd.to_datetime(japan.iloc[:, 0])
japan.set_index(japan.columns[0], inplace=True)

#Rename 1st column
japan = japan.rename(columns={japan.columns[0]: 'Japan'})

# Convert to float
japan=japan.astype(float)

#Divide by 1000,000
japan= japan / 1000000



#Data cleaning France
france= france.T

#Delete the first unnecessary line
france= france.iloc[1:]
#Invert the dataframe
france = france.iloc[::-1]

# Put dates in datetime and index format
france.iloc[:, 0] = pd.to_datetime(france.iloc[:, 0])
france.set_index(france.columns[0], inplace=True)

#Rename 1st column
france = france.rename(columns={france.columns[0]: 'France'})
# Convert to float
france=france.astype(float)

#Divide by 1000
france = france / 1000



#Data cleaning USA
usa=usa.T

#Delete the first unnecessary line 
usa= usa.iloc[1:]

# Put dates in datetime and index format
usa.iloc[:, 0] = pd.to_datetime(usa.iloc[:, 0])
usa.set_index(usa.columns[0], inplace=True)

#Rename 1st column
usa = usa.rename(columns={usa.columns[0]: 'Usa'})

# Convert to float
usa=usa.astype(float)

#Divide by 1000
usa= usa / 1000


#Data cleaning world
#Delete the first unnecessary line
world= world.iloc[1:]

# Put dates in datetime and index format
world.iloc[:, 0] = pd.to_datetime(world.iloc[:, 0])
world.set_index(world.columns[0], inplace=True)

#Rename 1st column 
world = world.rename(columns={world.columns[0]: 'World'})

# Convert to float
world=world.astype(float)

#Multiply by 1000
world= world * 1000


#We use common data for all countries between 2001-01-01 and 2023-05-01
japan = japan['2001-01-01':'2023-06-01']
columbia= columbia['2001-01-01':'2023-06-01']
france = france['2001-01-01':'2023-06-01']
sweden = sweden['2001-01-01':'2023-06-01']
usa = usa['2001-01-01':'2023-06-01']
world = world['2001-01-01':'2023-06-01']

#merge dataframes
dataframes_to_add = [france, usa, columbia, sweden, world]

# Concatenate columns of DataFrames
countries  = pd.concat([japan] + dataframes_to_add, axis=1)


#export Excel
export_path = "population_data_final_.xlsx"
countries.reset_index().to_excel(export_path, index=False)

#Descriptive statistics 
print(countries.describe())

# Plot countries 
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



#Plot World 
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(countries.index, countries['World'], label='World')

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


#Create a df growth_rate to calculate the monthly population growth rate
growth_rate= countries.copy()


#Create columns with growth rates:

payss =['France', 'Usa', 'Columbia','Sweden', 'Japan', 'World']

for pays in payss:
    growth_rate[f'{pays}_Growth'] = growth_rate[pays].pct_change() * 100




## Remove columns containing the population numbers for each country
columns_to_drop = ['France', 'Usa', 'Columbia', 'Sweden', 'Japan', 'World']
growth_rate.drop(columns=columns_to_drop, inplace=True)


#Delete the first unnecessary line
growth_rate= growth_rate.iloc[1:]


#descriptive statistics
growth_rate.describe()


#Graph growth rates over time 
for country in growth_rate:
    plt.figure()  # Create a new figure for each plot
    plt.plot(growth_rate.index, growth_rate[country], label=country)
    plt.title(country)
    plt.show()


#Create a new df 'avg' containing only country growth rate averages
data = {
    'France_Growth': growth_rate['France_Growth'].mean(),
    'Usa_Growth': growth_rate['Usa_Growth'].mean(),
    'Columbia_Growth': growth_rate['Columbia_Growth'].mean(),
    'Sweden_Growth': growth_rate['Sweden_Growth'].mean(), 
    'Japan_Growth': growth_rate['Japan_Growth'].mean(), 
    'World_Growth': growth_rate['World_Growth'].mean()
}


avg = pd.DataFrame(data, index=[0])

#Plot tge graph 
plt.figure(figsize=(10, 10))
ax = avg.plot(kind='bar', legend=True)
ax.set_title('Average Monthly Population Growth Rate')  # Title
#ax.set_xlabel('Countries')  # Title for x-axis
ax.set_ylabel('Average Rate')  # Title for y-axis
ax.set_xticklabels(['Countries'], rotation=0)  # Set the X-axis labels
ax.axhline(0, color='black', linewidth=1.5)  # Horizontal axis at level 0

plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()  # Layout adjustment

plt.show()


