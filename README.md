Los archivos "i2w.pk", "w2i.pk", "inference_model.py", "my_language_model_3.hdf5", "w2i.pk" deben estar en el mismo directorio.

El proyecto esta desarrollado en Python 3.

Primero se debe ejecutar el siguiente comando en el terminal para instalar todas las librerías necesarias.

$ pip install requirements.txt


Luego para generar el archivo txt con el poema:
$ python inference_model.py INPUT

INPUT es la palabra inicial que ingresará el usuario (recomendable probar con palabras como: amor, llorar, pena, etc.)