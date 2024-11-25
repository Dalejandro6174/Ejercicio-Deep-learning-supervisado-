"En el siguiente ejemplo se definirá como construir una ANN para un problema de clasificación
utilizando Keras y TensorFlow, para ello se utilizará una versión reducida del conjunto de datos
https://www.kaggle.com/datasets/ealaxi/paysim1, pero suficientemente grande para poder
entrenar adecuadamente una ANN de este tipo.
El conjunto de datos contiene información de transacciones bancarias hechas en diferentes
franjas horarias y con una serie de entidades relevantes asociadas a las mismas. El objetivo será
detectar si una transacción es o no fraudulenta en base al conocimiento pasado que se tiene
sobre las transacciones que ya se han detectado como tal fraudulentas, siendo por lo tanto un
problema de aprendizaje supervisado de clasificación binaria.
Las entidades relevantes que se elegirán para este problema son:
▪ type: Tipo de transacción. Variable categórica no ordinal que puede tomar los valores
‘CASH-IN’, ‘CASH-OUT’, ‘DEBIT’, ‘PAYMENT’, ‘TRANSFER’
▪ amount: Total de la transacción referido siempre a una misma moneda
▪ oldbalanceOrg: Balance inicial de la cuenta origen antes de la transacción
▪ oldbalanceDest: Balance inicial de la cuenta destino antes de la transacción
▪ isFraud: Variable categórica, es fraude (1) o no lo es (0)
El resto de variables no se consideran por ser combinaciones lineales de las ya mencionadas o
por simplificar el modelo para este ejemplo.
Con ello se puede definir una ANN con dos capas intermedias que tengan como función de
activación una ReLU y la capa de salida use una función sigmoide para que los datos se expresen
como una categoría binaria"

En primer lugar se cargarían los datos en cuestión y se preprocesarían para que la variable
categórica no ordinal se expresase como distintas categorías con valores binarios. Las ANN
necesitan los datos tengan magnitudes similares, por lo que también se estandarizan.
# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#############################################################################
##
# Data Preparation
#############################################################################
##
# Dataset -> https://www.kaggle.com/ntnu-testimon/paysim1
#dataset = pd.read_csv('paysim.csv')
#dataset = dataset.sort_values(by='isFraud', ascending=False)
#dataset.reset_index(drop=True, inplace=True)
#d = dataset[dataset['isFraud']==1]
#dataset = dataset.iloc[0:len(d)*2]
#dataset.to_csv('paysim_reduced.csv')
dataset = pd.read_csv('paysim_reduced.csv')
# Data Preparation
df_aux = pd.get_dummies(dataset['type']).astype('int')
dataset.drop(['type', 'Unnamed: 0', 'nameOrig',
 'nameDest', 'isFlaggedFraud', 'step',
 'newbalanceOrig', 'newbalanceDest'], axis=1, inplace=True)
dataset = dataset.join(df_aux)
X = dataset.loc[:, dataset.columns != 'isFraud'].values
y = dataset['isFraud'].values
# Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Con todo ello, el siguiente paso es crear una ANN usando las clases Dense y Sequential de
Keras. Sequential sirve para definir un marco general de una NN a la que se le irán añadiendo
capas con su método add; se crea un objeto que corresponde a la NN y se le van añadiendo sus
componentes estructurales.
Dense permite crea una capa de hidden units interconectadas entre sí y con la entrada y
vinculadas a una función de activación. Es importante definir correctamente la dimensionalidad
de las mismas, algo que se hace de acuerdo al esquema mostrado en la imagen de la
arquitectura de la NN.
La función de coste usada será la entropía cruzada para el caso binario y se usará un optimizador
que podrá ser alguno de los vistos previamente, como el sgd29 o Adam.
#############################################################################
##
# ANN Build
#############################################################################
##
# Importing the Keras libraries and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
def create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum,
decay):
 """
 Funcion para crear una NN para clasificacion binaria usando 2 HL

 """


 # Initialising the ANN
 model = Sequential()

 # First HL
 # [batch_size x n_features] x [n_features x w_in]
 model.add(Dense(units = w_in, input_dim = n_features,
 kernel_initializer = 'normal',
 activation = 'relu'))
 # Second HL
 # [batch_size x w_in] x [w_in x w_h1]
 model.add(Dense(units = w_h1, input_dim = w_in,
 kernel_initializer = 'normal',
 activation = 'relu'))

 # Output Layer
# [batch_size x w_h1] x [w_h1 x w_out]
 model.add(Dense(units = n_var_out,
 kernel_initializer = 'normal',
 activation = 'sigmoid'))

 # Compile Model
 # Loss Function -> Cross Entropy (Binary)
 # Optimizer -> sgd, adam...
 if optimizer == 'sgd':
 keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay,
nesterov=False)
 model.compile(loss='binary_crossentropy', optimizer='sgd',
metrics=['accuracy'])
 else:
 model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])

 return model
Una vez definida la estructura de la NN se entrena el modelo con unos hiperparámetros
adecuados y con un tamaño concreto de bacth size y un numero de epochs. Estos parámetros
se deberán buscar muchas veces mediante prueba y error y comprobando sobre los datos de
validación que combinación da mejores resultados.
# Parametros
n_features = np.shape(X_train)[1]
w_in = 12
w_h1 = 8
n_var_out = 1
batch_size = 100
nb_epochs = 100
optimizer = 'adam'
lr = 0.1
momentum = 0.01
decay = 0.0
# Create NN
model = create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum,
decay)

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epochs)
Por último se evalúan los resultados y se obtiene una matriz de confusión para ver los casos
correctamente clasificados. 
#############################################################################
##
# ANN Predictions
#############################################################################
##
# Predict
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

