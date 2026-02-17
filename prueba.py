#Esta parte del código se obtuvo del archivo "test.py" para el código en "network.py"
#las fotos relacionadas a la ejecución del archivo "network.py" y este archivo se 
#encuentran en la carpeta "Fotos evidencia".

import mnist_loader #Se cargan los datos de mnist.
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#load_data_wrapper() segmenta el conjunto de imagenes en conjuntos para entrenamiento,
#validación y prueba.
training_data = list(training_data) #Se convierten los datos entrenados a una lista.

import network # Se importa la red neuronal creada anteriormente.


net = network.Network([784, 30, 10]) #Esto es, la red tiene 784 neuronas de entrada,
#30 neuronas ocultas y 10 de salida, relacionandose con los digitos del 0 al 9.
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)#Se entrena la red.
