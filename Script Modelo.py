import numpy as np
import pandas as pd
import pmdarima as pm
from datetime import datetime
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import mysql.connector

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

mydb = mysql.connector.connect(
  host="localhost",
  user="",
  password="",
  database="tesis_test"
)
mycursor = mydb.cursor()

mycursor.execute("SELECT id_producto FROM producto")
productos = mycursor.fetchall()
flag_inf_general = 0

df_exog_var_gen = pd.DataFrame(columns=['year', 'month'])
df_exog_var_gen.set_index(['year', 'month'], inplace=True)
now = datetime.now()

for producto in productos:
    #INFORMACIÓN DE VENTAS
    mycursor.execute(f"SELECT YEAR(fecha_entrega) AS anho, MONTH(fecha_entrega) AS mes, SUM(cantidad) AS cantidad_total_vendida FROM Linea_Orden_Venta WHERE fid_producto = {producto[0]} GROUP BY YEAR(fecha_entrega), MONTH(fecha_entrega) ORDER BY anho, mes")
    ventas = mycursor.fetchall()
    df_ventas = pd.DataFrame(ventas, columns=['year', 'month', 'ventas'])
    df_ventas.set_index(['year', 'month'], inplace=True)

    #INFORMACIÓN EXTRA GENERAL
    if flag_inf_general == 0:
        flag_inf_general = 1
        mycursor.execute(f"SELECT id_informacion_extra FROM informacion_extra where activo = True and exclusiva_producto = False")
        informacion_extra = mycursor.fetchall()
        id = 0
        for inf_extra in informacion_extra:
            mycursor.execute(f"SELECT anho, mes, valor from  tiempo, valor_informacion_extra_general where fid_informacion_extra = {inf_extra[0]} and fid_tiempo = id_tiempo")
            valores_informacion_extra = mycursor.fetchall()
            df = pd.DataFrame(valores_informacion_extra, columns=['year', 'month', f'exog_{id}'])
            id +=1
            df.set_index(['year', 'month'], inplace=True)
            df_exog_var_gen = pd.concat([df_exog_var_gen, df], axis=1)
        
        # df_exog_var_gen.reset_index(inplace=True)
        df_exog_var_gen = df_exog_var_gen.dropna()

    #INFORMACIÓN POR PRODUCTO
    mycursor.execute(f"SELECT id_informacion_extra FROM informacion_extra where activo = True and exclusiva_producto = True")
    informacion_extra_prod = mycursor.fetchall()
    df_exog_var_prod = pd.DataFrame(columns=['year', 'month'])
    df_exog_var_prod.set_index(['year', 'month'], inplace=True)
    for inf_extra in informacion_extra_prod:
        mycursor.execute(f"SELECT anho, mes, valor from  tiempo, valor_informacion_extra_producto where fid_informacion_extra = {inf_extra[0]} and fid_producto = {producto[0]} and fid_tiempo = id_tiempo ")
        valores_informacion_extra = mycursor.fetchall()
        df = pd.DataFrame(valores_informacion_extra, columns=['year', 'month', f'exog_{id}'])
        id +=1
        df.set_index(['year', 'month'], inplace=True)
        df_exog_var = pd.concat([df_exog_var_prod, df], axis=1)


    # df_exog_var_prod.reset_index(inplace=True)
    df_exog_var_prod= df_exog_var_prod.dropna()
    
    df_exog = pd.concat([df_exog_var_gen, df_exog_var_prod], axis=1)
    exog_vars = df_exog.columns
    df_data = pd.concat([df_ventas, df_exog], axis=1)
    

    df_data.reset_index(inplace=True)
    while df_data.isna().any(axis=0).any():
        df_exog_predict = df_data.iloc[-12:]
        df_data = df_data.iloc[:len(df_data)-12]
    
    train_size = int(len(df_data)*0.9333333)
    train, test = df_data['ventas'][:train_size], df_data['ventas'][train_size:]

    print(f"EMPEZÓ EL ENTRENAMIENTO DEL MODELO AUTO-ARIMA para el producto {producto[0]}:")

    # Entrenar modelo ARIMAX en el conjunto de entrenamiento
    modelo = pm.auto_arima(train.values,
                        X=df_data[exog_vars][:train_size].values,
                        seasonal=True,
                        m=12,
                        suppress_warnings=True,
                        stepwise=True)
    
    predicciones = modelo.predict(n_periods=len(test), X=df_data[exog_vars][train_size:].values)

    meanError = mean_absolute_error(test.values, predicciones)
    rootmeanSquaredError = rmse(test.values, predicciones)
    r2Score = r2_score(test.values, predicciones)

    print("\nResultados Métricas:  AUTO_ARIMA ")
    print(f"Mean Error en el conjunto de prueba: {meanError:.2f} Kg")
    print(f"Root Mean Squared Error en el conjunto de prueba:  vs {rootmeanSquaredError:.2f} Kg")
    print(f"R2 en el conjunto de prueba: {r2Score:.2f}")


    modelo = pm.auto_arima(df_data['ventas'].values,
                       X=df_data[exog_vars].values,
                       seasonal=True,
                       m=12,
                       suppress_warnings=True,
                       stepwise=True)

    predicciones_anual = modelo.predict(n_periods=12, X=df_exog_predict[exog_vars].values)

    print("\nResultado valores futuros AUTO-ARIMA:")
    print(predicciones_anual)

    print("===================================================================")

    ultimo_year = df_data['year'].iloc[-1]
    ultimo_mes = df_data['month'].iloc[-1]

    sql = "INSERT INTO ejecucion_modelo (fecha_ejecucion, error_promedio, ajuste_modelo, rmse) VALUES (%s, %s, %s, %s)"
    val = (now, float(meanError), float(r2Score), float(rootmeanSquaredError))
    mycursor.execute(sql, val)

    idEjecucion = mycursor.lastrowid

    for inf_extra in informacion_extra:
        sql = "INSERT INTO informacion_extra_ejecucion (fid_ejecucion_modelo,fid_informacion_extra) VALUES (%s, %s)"
        val = (int(idEjecucion),int(inf_extra[0]))
        mycursor.execute(sql, val)
        
    for inf_extra in informacion_extra_prod:
        sql = "INSERT INTO informacion_extra_ejecucion (fid_ejecucion_modelo,fid_informacion_extra) VALUES (%s, %s)"
        val = (int(idEjecucion),int(inf_extra[0]))
        mycursor.execute(sql, val)

    for value in predicciones_anual:
        new_month = ultimo_mes + 1
        new_year = ultimo_year
        if new_month > 12:
            new_year += 1
            new_month -= 12
        sql = "INSERT INTO demanda_estimada (fid_producto,fid_ejecucion_modelo,mes,anho,valor) VALUES (%s, %s, %s, %s, %s)"
        val = (int(producto[0]),int(idEjecucion), int(new_month), int(new_year), float(value))
        mycursor.execute(sql, val)

        ultimo_mes = new_month
        ultimo_year = new_year

    mydb.commit()

