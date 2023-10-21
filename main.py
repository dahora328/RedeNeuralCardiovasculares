import json
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import Metric

# função read_csv que recebe o caminho da pasta que contem os dados de entrada e saída
# retorna um objeto DataFrame uma estrutura tabular bidimensional.
# camada de entrada
previsores = pd.read_csv('C:/Users/Rodri/PycharmProjects/RedeNeuralCardiovasculares/heart.csv')
# camada de saída
classe = pd.read_csv('C:/Users/Rodri/PycharmProjects/RedeNeuralCardiovasculares/saida.csv')

#inicialização da rede
classificador = Sequential()


#adicionado na rede uma camada,camada oculta com 10 neurônios
#função de ativação ReLU, pesos inicializado aleatoriamente uniformes
#camada de entrada com 13 neurônios
classificador.add(Dense(units=10, activation='relu',
                        kernel_initializer='random_uniform', input_dim=13))
#Feito dropout na rede
classificador.add(Dropout(0.2))
#Adicionado mais uma camada, camada oculta com 10 neurônios
#função de ativação ReLU, pesos inicializado aleatoriamente uniformes
classificador.add(Dense(units=10, activation='relu',
                        kernel_initializer='random_uniform'))
#Feito segundo dropout na rede
classificador.add(Dropout(0.2))

classificador.add(Dense(units=5, activation='relu',
                        kernel_initializer='random_uniform'))
#Feito segundo dropout na rede
classificador.add(Dropout(0.2))


#adicionado a camada de saída, com 1 neurônio de saída
#função sigmoid
classificador.add(Dense(units=1, activation='sigmoid'))


# Configuração da otimização
otimizador = keras.optimizers.Adam(learning_rate=0.0001, decay=0.0001, clipvalue=0.5)
# Aplicação da otimização, função de perda e metricas
classificador.compile(optimizer='adam', loss='hinge',
                      metrics=[keras.metrics.BinaryAccuracy(), 'mse'])

# aplição do treinamento da rede
historico = classificador.fit(previsores, classe, batch_size=10, epochs=100)

# salva o resultado da rede neural em arquivo json
historico_json = json.dumps(historico.history)
# #mse = historico.history['mse']
with open('historico_treinamento.json', 'w') as json_file:
    json_file.write(historico_json)



"""
#exibir gráfico

# Carregar o arquivo JSON com o histórico de treinamento
with open('historico_treinamento.json', 'r') as json_file:
    historico = json.load(json_file)

# Extrair as métricas de interesse do histórico
epochs = range(1, len(historico['loss']) + 1)
loss = historico['loss']
accuracy = historico['binary_accuracy']

# Plotar o gráfico de perda
plt.plot(epochs, loss, 'b', label='Loss')
plt.title('Histórico de Treinamento')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotar o gráfico de acurácia
plt.plot(epochs, accuracy, 'r', label='Accuracy')
plt.title('Histórico de Treinamento')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotar o gráfico do MSE
plt.plot(mse)
plt.title('Gráfico do Erro Quadrático Médio (MSE)')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.show()
"""
