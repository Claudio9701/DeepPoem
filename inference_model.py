from sys import argv, stdout
from pickle import load
from nltk import download, word_tokenize
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
from datetime import datetime
import numpy as np

def Get_inferenceModel():
	# define model
	inference_model = Sequential()
	inference_model.add(Embedding(vocab_size, EMBED_SIZE, input_length=1, batch_input_shape=(1,1), name='embeddings'))
	# alterada para LSTM
	for i in range(LSTM_LAYERS):
                inference_model.add(LSTM(HIDDEN_SIZE[i], stateful=True, return_sequences=True, name = "lstm_{}".format(i)))
	inference_model.add(Dense(vocab_size, name='linear'))
	inference_model.add(Activation('softmax', name='softmax'))

	inference_model.load_weights('my_language_model_3.hdf5')
	return inference_model

def printRed(texto, end="\n"):
    print('\033[31m' + str(texto) +'\033[0;0m', end=end)

def printBlue(texto, end="\n"):
    print('\033[34m' + str(texto) +'\033[0;0m', end=end)

def query_sentences(INPUT, verbose=0):
    INPUT = INPUT.lower()
    INPUT = word_tokenize(INPUT , language='spanish')

    inference_model.reset_states()
    w = np.zeros((1, 1))
    valIndice = len(INPUT)-1
    
    gen_poem = []
    
    for i, word in enumerate(INPUT[0:valIndice]):
        w[0,0] = word2index[word] 
        inference_model.predict(w)

    startWord = np.zeros((1, 1))
    
    startWord[0,0] = word2index[INPUT[valIndice]]
    if verbose==1:
      print_Red(index2word[startWord[0,0]] + " " , end="")
    gen_poem.append(index2word[startWord[0,0]])
    for i in range(80):         
        
        nextWordProbs = inference_model.predict(startWord)                
        nextWordId = nextWordProbs.squeeze().argmax()

        startWord[0,0] = nextWordId
        if verbose==1:
          print_blue(index2word[nextWordId] + " ", end = "")
        gen_poem.append(index2word[nextWordId])
    
    return " ".join(gen_poem)

def query_sentences_multinomial(INPUT, verbose=0):
    INPUT = INPUT.lower()
    INPUT = word_tokenize(INPUT , language='spanish')

    inference_model.reset_states()
    
    valIndice = len(INPUT)-1

    gen_poem = []

    w = np.zeros((1, 1))

    for i, word in enumerate(INPUT[0:valIndice]):
        printRed(word + " " , end="")
        w[0,0] = word2index[word]
        inference_model.predict(w)


    startWord = np.zeros((1, 1))
    startWord[0,0] = word2index[INPUT[valIndice]]
    if verbose==1:
      print_Red(index2word[startWord[0,0]] + " " , end="")
    gen_poem.append(index2word[startWord[0,0]])   
    for i in range(70):         

        nextWordProbs = inference_model.predict(startWord)  
        nextWordProbs = nextWordProbs/nextWordProbs.sum(dtype=np.float32) 
        x = nextWordProbs.squeeze()
        norm =[float(i) for i in x]/sum(x)
        nextWordId = np.random.multinomial(1, norm , 1).argmax()

        startWord[0,0] = nextWordId 
        if verbose==1:
          print_blue(index2word[nextWordId] + " ", end = "")
        gen_poem.append(index2word[nextWordId])
    
    return " ".join(gen_poem)

if __name__ == '__main__':
	download('punkt', quiet=True)
	word2index = load(open('w2i.pk','rb'))
	index2word = load(open('i2w.pk','rb'))
	vocab_size = len(word2index)
	SEQ_LEN = 1
	HIDDEN_SIZE = [128, 64, 64]
	EMBED_SIZE = 256
	LSTM_LAYERS = 3
	inference_model = Get_inferenceModel()
	initial_word = str(argv[1])
	generated_poem = query_sentences(initial_word)
	now = datetime.now().strftime("%d-%m-%y-%H_%M_%S")
	txt_file = open('texto.txt', 'w')
	txt_file.write(generated_poem)
	txt_file.close

