import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

PASOS=7



def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test


datos = pd.read_csv('datasets/LineaTiempo2.csv',parse_dates=[0])
#print(datos.head)
plt.plot(datos)
#plt.show()
names = list(datos.columns)

for name in names[1:]:
    df = pd.DataFrame(data=datos[name].values, index=datos['tiempo'])
    # load dataset
    
   
    values = df.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, PASOS, 1)

    values = reframed.values
    n_train_days = 250- (30+PASOS)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    # split into input and outputs
    x_train, y_train = train[:, :-1], train[:, -1]
    x_val, y_val = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    EPOCHS=100
    model = crear_modeloFF()
    history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)

#    results=model.predict(x_val)
#    plt.scatter(range(len(y_val)),y_val,c='g')
#    plt.scatter(range(len(results)),results,c='r')
#    plt.title('validate')
#    plt.show()
#
#
    
    
    df = pd.DataFrame(data=datos[name].values, index=datos['tiempo'])
    ultimosDias = df['2020-06-15':'2020-06-30']
    siguienteQuin = df['2020-07-01':'2020-07-07'].values
    #print(ultimosDias)
    values = ultimosDias.values
    values = values.astype('float32')
    # normalize features
    values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, PASOS, 1)
    reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
    #print(reframed.head(7))
    values = reframed.values
    x_test = values[6:, :]
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    

 
    results=[]
    for i in range(7):
        parcial=model.predict(x_test)
        results.append(parcial[0])
        #print(x_test)
        x_test=agregarNuevoValor(x_test,parcial[0])
    
    adimen = [x for x in results]    
    inverted = scaler.inverse_transform(adimen)

    
    print("para serie: ",name)
    for i,j in zip(inverted,siguienteQuin):
        print(i,j)
   
