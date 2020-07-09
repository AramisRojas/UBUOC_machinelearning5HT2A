# Predicción de actividad inhibitoria de moléculas pequeñas sobre el receptor serotoninérgico 5-HT2A mediante modelos de machine learning.
TFM MSc Bioinformática y bioestadística UB-UOC
Lenguaje: Python // Autor@: Aramis Adriana Rojas Mena

## Descripcion
En el siguiente repositorio se encuentran los scripts utilizados para llevar a cabo la ejecución de los modelos de machine learning para la predicción de la actividad inhibitoria sobre este receptor 5-HT2A particular. Los datos originales utilizados se encuentran en este repositorio bajo la extension ".csv" y han sido obtenidos mediante el gestor Postgres con volcado de toda la BBDD de ChEMBL (v.26). Se facilitan los datos de entrada (input) necesarios para reproducir el estudio de forma idéntica al ejecutado por la autora.  

Si se desea realizar un estudio con datos diferentes, la variable descriptora esencial sería disponer de las moléculas en notación SMILES (que posteriormente será transformada a un objeto de clase ROMol), y como variable regresora (a predecir) la Ki, que aquí es "pchembl_value" (constante de inhibición; recomendable su normalización logaritmica si no lo estuviera)
```python
#*Morgan fingerprints by default*
PandasTools.AddMoleculeColumnToFrame(df_final,smilesCol='canonical_smiles')
mfps = rdFingerprintGenerator.GetFPs(list(df_final['ROMol']))
df_final['MFPS'] = mfps
```  

## Contenido del repositorio
- _tfm_script_balanced.py_ (modelos de clasificación con balanceo)
- _regression_script.py_ (modelos de regresión con balanceo)
- _graphics_script.py_ (gráficos)
- _5ht2a_definitive_nosalts.csv_ (set de datos limpio)
- _inchikeys_2.txt_ (notación InChiKeys que importa el código para adicionar una de las columnas; necesario para eliminar los duplicados)
Se recomienda ejecutarlo en este orden: 1) _regression_script.py_, 2) _tfm_script_balanced.py_, 3) _graphics_script.py_

## Cómo utilizarlo
Se ha utilizado el IDE Spyder dentro del entorno Anaconda, pero también se podría utilizar Jupyter notebooks. 
Es necesario instalar previamente el módulo rdkit https://www.rdkit.org/docs/Install.html#how-to-install-rdkit-with-conda y quizá sea necesario, si existen problemas para reconocer el módulo rdkit en Spyder/Jupyter, dentro del environment de rdkit, instalar ipython (Anaconda prompt).
```
conda create -c rdkit -n my-rdkit-env rdkit
conda activate my-rdkit-env
conda install ipython
```
Al abrir el primer script es importante establecer un directorio de trabajo en la máquina en la que se trabaje, donde también han de estar localizados los archivos _5ht2a_definitive_nosalts.csv_ y _inchikeys_2.txt_  
A continuación, se muestra como ejemplo la ruta donde estaban estos archivos en la máquina original: 
```python
import os 
os.chdir(r"C:/Users/usuario/OneDrive/EstadisticaUOC/4-SEMESTRE/TFM/Datos_recuperados_ChEMBL")
```
**Se recomienda ejecutarlo en este orden: 1) _regression_script.py_, 2) _tfm_script_balanced.py_, 3) _graphics_script.py_**

Si se quiere reproducir el estudio exactamente igual, es conveniente no realizar cambios; especialmente, no modificar las líneas relativas a los warnings.
```python
from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings
```
Por el resto de código, se debería de poder ejecutar de forma secuencial sin problemas. Las librerías necesarias están en los diferentes scripts, con sus respectivas llamadas cuando son necesitadas.  
Aparecen: _pandas, numpy, yellowbrick, yellowbrick.classifier, yellowbrick.model_selection, rdkit, rdkit.Chem, seaborn, matplotlib.pyplot, sklearn.cluster, sklearn, sklearn.model_selection, rdkit.DataStructs.cDataStructs, sklearn.metrics, sklearn.linear_model, sklearn.ensemble, imblearn.under_sampling, sklearn.neighbors, sklearn.naive_bayes_.

## Contacto
Si se encuentra algún bug en el código o se quiere hacer algún comentario, escribe a: aramisrojas.farmacia@gmail.com

