#Bibliotecas necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb

#Descargo el CSV
csv_file_url = 'https://drive.google.com/file/d/1reUBOFFwRdL9s5SMaYMC5wVnCaWnAH-k/view?usp=sharing'
file_id = csv_file_url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id
df = pd.read_csv(dwn_url, sep=';')
df = df.set_index(['user_uuid', 'course_uuid', 'particion'])

#################################################################################################
## CALCULO SEGÚN LA NOTA FINAL LAS PROBABILIDADES DE ABANDONO
#################################################################################################

# Defino la variable dependiente (abandono)
# Esta variable asume que el estudiante abandona el curso si su nota final es menor que 4
df['abandono'] = (df['nota_final_materia'] < 4).astype(int)

# Agrupo por 'user_uuid', 'course_uuid', 'particion' y calculo la mediana
medianas = df.groupby(['user_uuid', 'course_uuid', 'particion']).median(numeric_only=True)

# Resetear los índices
df.reset_index(inplace=True)
medianas.reset_index(inplace=True)

# Relleno los valores NaN con la mediana del grupo correspondiente
df.fillna(medianas, inplace=True)

# Si aún quedan valores NaN (porque algún grupo completo era NaN), los relleno con la mediana global
#df = df.fillna(df.median())

# Si aún quedan valores NaN (porque algún grupo completo era NaN), los relleno con 0
df.fillna(0, inplace=True)

# Defino las variables independientes (predictores)
X = df[['nota_final_materia', 'nota_parcial', 'score']] 

# Defino la variable dependiente (abandono)
Y = df['abandono']

# Divido el conjunto de datos en entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Ajusto el modelo de regresión logística
model = LogisticRegression()
model.fit(X, Y)

# Obtengo la probabilidad de abandono para cada observación del conjunto de evaluación
y_proba = model.predict_proba(X)
y_proba_abandono = y_proba[:, 1]

# Creo el dataframe con las probabilidades y el mismo índice que df
Datos_Final = pd.DataFrame({'user_uuid': df['user_uuid'], 'course_uuid': df['course_uuid'], 'probabilidad':y_proba_abandono[:]}, index=df.index)

# Filto el dataframe
df_filtrado = Datos_Final.loc[Datos_Final['probabilidad'] > 0.8]


# Muestro la probabilidad en forma de porcentaje
#df_filtrado['porcentaje'] = df_filtrado['probabilidad'].apply(lambda x: round(x*100, 2))
df_filtrado.loc[:, 'porcentaje'] = df_filtrado['probabilidad'].apply(lambda x: round(x*100, 2))

# Agrupo por 'user_uuid' y 'course_uuid' y calculo el promedio del porcentaje
df_agrupado = df_filtrado.groupby(['user_uuid', 'course_uuid'])['porcentaje'].mean()

# Muestro el dataframe agrupado
print('-----------------------------------Resultados---------------------------------------------\n',df_agrupado)



""" 
sb.set_style('whitegrid')
plt.figure(figsize=(10, 1))
sb.heatmap(df_filtrado, cmap='RdYlGn', cbar=False, annot=True)
plt.xlabel('Probabilidad de abandono')
plt.yticks([])
plt.show() 
"""
