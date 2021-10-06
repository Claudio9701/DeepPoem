# DeepPoem

Text Generation LSTM Keras model trained with Cesar Vallejo poems.

👉 Check out our interactive [web app!](https://deep-demos.herokuapp.com/poetNN/)

## Quick Start

The project is developed in Python 3. The files "i2w.pk", "w2i.pk", "inference_model.py", "my_language_model_3.hdf5", "w2i.pk" must be in the same directory.

First, the following command must be executed in the terminal to install all the necessary libraries:

```sh
$ pip install requirements.txt
```

Then to generate the txt file with the poem:

```sh
$ python inference_model.py INPUT
```

*INPUT is the initial word that the user will enter (it is advisable to try words like: love, cry, grief, etc.)*
