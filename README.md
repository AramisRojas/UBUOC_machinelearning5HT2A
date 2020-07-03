# Predicción de actividad inhibitoria de moléculas pequeñas sobre el receptor serotoninérgico 5-HT2A mediante modelos de machine learning.
TFM MSc Bioinformática y bioestadística UB-UOC
Lenguaje: Python // Autor@: Aramis Adriana Rojas Mena

## Descripcion
En el siguiente repositorio se encuentran los scripts utilizados para llevar a cabo la ejecución de los modelos de machine learning para la predicción de la actividad inhibitoria sobre este receptor 5-HT2A particular. Los datos originales utilizados se encuentran en este repositorio bajo la extension ".csv" y han sido obtenidos mediante el gestor Postgres con volcado de toda la BBDD de ChEMBL (v.26). Se facilitan los datos de entrada (input) necesarios para reproducir el estudio de forma idéntica al ejecutado por la autora.  

Si se desea realizar un estudio diferente, la variable descriptora esencial sería disponer de las moléculas en notación SMILES, y como variable regresora (a predecir) la Ki, que aquí es "pchembl_value" (constante de inhibición; recomendable su normalización logaritmica si no lo estuviera)

## Contenido del repositorio
- tfm_script_balanced.py (modelos de clasificación con balanceo)
- regression_script.py (modelos de regresión con balanceo)
- graphics_script.py (gráficos)
- 5ht2a_definitive_nosalts.csv (set de datos limpio)
- inchikeys_2.txt (necesario para eliminar los duplicados  
Se recomienda ejecutarlo en este orden: 1) regression_script.py, 2) tfm_script_balanced.py, 3) graphics_script.py

## Cómo utilizarlo
Al abrir el primer script es importante establecer un directorio de trabajo en la máquina en la que se trabaje, donde también han de estar localizados los archivos _5ht2a_definitive_nosalts.csv_ y _inchikeys_2.txt_
