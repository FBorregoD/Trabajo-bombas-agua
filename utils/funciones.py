# funciones.py

# los primeros import
from enum import unique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def tipos_col(df):

    # devuelve un diccionario con los nombres de las columnas que cumplen el tipo de dato de la columna
    # argumento de entrada: df Dataframe de entrada

    list_num=[]
    list_obj=[]
    list_bool=[]
    list_obj_type=["object"]
    list_num_type=["int64","float64"]
    list_bool_type=["bool"]
    for col in df:
        if df[col].dtypes in list_bool_type:
            list_bool.append(col)
        elif df[col].dtypes in list_num_type:
            list_num.append(col)
        elif df[col].dtypes in list_obj_type:
            list_obj.append(col)
    return {"num":list_num,"obj":list_obj,"bool":list_bool}

def valores_nulos_show(df):
    # devuelve un dataframe indicando qué columnas tienen valores nulos y qué porcentaje reprensentan estos

    data={
        "tipo datos":df.dtypes,  
        "valores nulos":df.isna().sum(),
        "% nulos":np.round(df.isna().sum()/df.shape[0]*100,2)
        }
    data_f=pd.DataFrame(data)
    return data_f.loc[data_f["valores nulos"]!=0]

def histograma_num(df,rows=1):
    # no se usa en el EDA

    df_cols=tipos_col(df)
    cols=int(np.ceil(len(df_cols["num"])/rows))
    # Create subplots 
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16,8))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Histograma')

    for i in range(len(df_cols["num"])):
        name=df_cols["num"][i]
        plt.subplot(rows,cols,i+1, title=name)
        df[df_cols["num"][i]].hist()

        # Generate histograms
    plt.show()


def cuenta_valores_unicos(df,target):
    # función interna, cuanta valores únicos y calcula una media no demasiado útil que divide el número de valores entre el número de valores únicos
    uniq=len(df[target].unique())
    sal=pd.DataFrame({"valores_unicos":uniq,
                    "media":df.shape[0]/uniq},index=[target])
    return sal

def cuenta_valores_per(df,target_col,estilo=True):
    # ordena en una lista los valores únicos de una columna "target_col" e indica su prevalencia en dicha columan en porcentaje
    # si el argumento estilo es True la salida sólo puede mostrarse para gráficas
    if estilo == True :
        print(cuenta_valores_unicos(df,target_col))
        return pd.DataFrame({"total":df[target_col].value_counts(),"%":100*df[target_col].value_counts()/df[target_col].count()}).head(10).style.format('{:.2f}',subset=["%"])
    else:
        return pd.DataFrame({"total":df[target_col].value_counts(),"%":100*df[target_col].value_counts()/df[target_col].count()})

def resumen_tipos_unicos(df,cols):
    # Devueve un dataframe para las columnas pedidas del dataframe df, 
    # donde cada línea indica el número de valores únicos 
    # y qué porcentaje tiene el valor más abundante
    salida=pd.DataFrame(index=["valores_unicos","porcentaje_mayor_valor"],columns=cols)
    for col in cols:
        salida[col]=[len(df[col].unique()),cuenta_valores_per(df,col,False).values[0,1]]
    return salida.T

def simplifica_categoricas(df,obj,por_mayor_valor_min=20,per_umbral=5):
    res=cuenta_valores_per(df,obj,False).reset_index().rename(columns = {'index':'valor'})
    res[obj+"_s"]=np.where((res.index<por_mayor_valor_min) & (res["%"]>per_umbral),res["valor"],"other")
    return res

def agrupa_categoricas(df,obj,por_mayor_valor_min=4,per_umbral=5):
    """
    Devuelve una columna donde se han simplificado aquellos valores de esa columna, 
    eliminado aquellos con existencia por debajo del % en 'per_umbral=5' 
    y quedándose sólo con los 'por_mayor_valor_min=4' primeros.
    """
    diccio=simplifica_categoricas(df,obj,por_mayor_valor_min,per_umbral)
    salida=df[obj].map(dict(zip(diccio["valor"],diccio[obj+"_s"])))
    return(salida)

def fill_object_nan(df):
    for col in df:
        if df[col].dtype == "object":
            df[col]=df[col].fillna("unknown")
    return df

def tabl_contingencia(df,grupo_cat,target, relativo=False,ordena=False):
    target_cat=df[target].unique()
    index_sal=df[grupo_cat].unique()
    sal=pd.DataFrame(index=index_sal)
    if ordena:sal=sal.sort_index(ascending=True)

    for linea in target_cat:
        sal[linea]=df.loc[df[target]==linea][grupo_cat].value_counts()
    sal.fillna(0)

    if relativo:
        sal=sal[sorted(sal.columns)]
        lel=pd.DataFrame({"porcentaje":df[target].value_counts()/df[target].count()}).transpose()[sorted(sal.columns)]
        lel.fillna(0)
        sal=sal.div(sal.sum(axis=1),axis=0)
        sal=sal.divide(lel.values[0])

    sal.fillna(0)

    return sal


def plot_bar_cat(df,grupo_cat,target,tamx=30,tamy=12,relativo=False):
    sal= tabl_contingencia(df,grupo_cat,target, relativo)
    ax = sal.plot.bar(rot=0)
    ax.legend(loc='lower right')
    ax.figure.set_size_inches([tamx,tamy])

def error_mayoritario(df,grupo_cat,target,ordena=True):
    target_cat=df[target].unique()
    index_sal=df[grupo_cat].unique()
    sal=pd.DataFrame(index=index_sal)
    if ordena: sal=sal.sort_index(ascending=True)

    for linea in target_cat:
        sal[linea]=df.loc[df[target]==linea][grupo_cat].value_counts()
    sal.fillna(0)
    
    sal=sal[sorted(sal.columns)]
    lel=pd.DataFrame({"porcentaje":df[target].value_counts()/df[target].count()}).transpose()[sorted(sal.columns)]
    lel.fillna(0)
    sal=sal.div(sal.sum(axis=1),axis=0)
    sal.fillna(0)

    sal=sal.divide(lel.values[0])-1
    sal.fillna(0)
    sal=pow((sal),2)
    return pd.DataFrame({grupo_cat:pow(sal.mean().mean(),0.5)}, index=[target])

def variabilidad(df,target,umbral=20):
    col_ojb=tipos_col(df)["obj"]
    errores=pd.DataFrame(index=[target])

    for col in col_ojb:
        cuenta = len(df[col].unique())
        if col != target and cuenta <=umbral:
            errores[col]=error_mayoritario(df,col,target,False)
    return errores

def hist_var(df,target,variable,bins=20,modo_seaborn=True,log_scale=False):
    if modo_seaborn==False:
        df.groupby(df[variable])[target]\
            .value_counts()\
            .unstack(1)\
            .plot(kind="hist", bins=bins,stacked=True,\
                title="Histograma de la distribución de pozos según "+target+" y variable "+variable)
    else:
        sns.histplot(x=df[variable], hue=df[target], element='step',log_scale=log_scale,bins=bins,kde=True).set( \
            title="Histograma de la distribución de pozos según "+target+" y variable "+variable)



def grafica_tanzania(df,ruta,grupo,size=12):

    df=df.loc[df["longitude"]!=0]

    BBox = [df.longitude.min(),   df.longitude.max(),      
         df.latitude.min(), df.latitude.max()]
    print(BBox)

    fig, ax = plt.subplots(figsize = (size,size))
    groups = df.groupby(grupo)
    img_map = plt.imread(ruta)
    for name, group in groups:
        ax.scatter(group.longitude, group.latitude, label=name, marker='o',zorder=1, alpha= 0.8, s=13)
    ax.legend(numpoints=1)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(img_map, zorder=0, extent = BBox, aspect= 'equal')

def lugar_mas_cercano(lugar,tipo="euclidea",verbose=False):
    """
    Función creada para buscar la distancia más cercana de una matriz de dos columnas. Es extremadamente lenta.
    """
    print("método nuevo, distancia máxmimo 500 km")
    if tipo=="euclidea":
        modo_tipo=1
    elif tipo=="gps":
        modo_tipo=2
    else:
        modo_tipo=3
    cont=0
    if type(lugar)==pd.DataFrame:
        region=lugar.values[:,2]
        lugar=lugar.values[:,0:2]
    


    lugares=len(lugar)
    dist_cerca=np.zeros(lugares)
    rango=range(lugares)
    min_inf=np.Inf
    r=6370

    for i in rango:
        cont = cont+1
        if (verbose):
            print(f"{cont} de {lugares}, van {round(100*cont/lugares,3)}%")
        minimo=min_inf
        for j in rango:
            if i==j or region[i]!=region[j]:
                continue
            if modo_tipo== 1:  
                dist=np.linalg.norm(lugar[i] - lugar[j])
            elif modo_tipo== 2: 
                lat1,lon1=lugar[i]
                lat2,lon2=lugar[j]
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.power(np.sin(dlat / 2),2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2)**2,2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                dist= r*c
            else:print("la variable tipo debe ser o euclidea o gps")
            
            if dist < minimo:
                minimo=dist
        if minimo >500:
            minimo=500
        dist_cerca[i]=minimo

    dist_cerca=dist_cerca.tolist()
    return(dist_cerca)

def lugar_mas_cercano_xtest(lugar_conocido,lugar_desconocido,tipo="euclidea",verbose=False):
    """
    Función creada para buscar la distancia más cercana de una matriz de dos columnas.
    Busca las distancias de los puntos X a los puntos de lugar conocido sin tener en cuenta los otros puntos desconocidos.
    Es extremadamente lenta.
    Esta recibe dos dataframes: la de lugares ya conocidos y la de lugares desconocidos.
    """
    if tipo=="euclidea":
        modo_tipo=1
    elif tipo=="gps":
        modo_tipo=2
    else:
        modo_tipo=3
    
    cont=0

    region_con=lugar_conocido.values[:,2]
    lugar_con=lugar_conocido.values[:,0:2]
    region_des=lugar_desconocido.values[:,2]
    lugar_des=lugar_desconocido.values[:,0:2]

    lugares_con=len(lugar_con)
    lugares_des=len(lugar_des)

    dist_cerca=np.zeros(lugares_des)
    rango_con=range(lugares_con)
    rango_des=range(lugares_des)
    min_inf=np.Inf
    r=6370

    for i in rango_des:
        cont = cont+1
        if (verbose):
            print(f"{cont} de {lugares_des}, van {round(100*cont/lugares_des,3)}%")
        minimo=min_inf
        for j in rango_con:
            if region_des[i]!=region_con[j]:
                continue
            if modo_tipo== 1:  
                dist=np.linalg.norm(lugares_des[i] - lugar_con[j])
            elif modo_tipo== 2: 
                lat1,lon1=lugar_des[i]
                lat2,lon2=lugar_con[j]
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.power(np.sin(dlat / 2),2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2)**2,2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                dist= r*c
            else:print("la variable tipo debe ser o euclidea o gps")
            
            if dist < minimo:
                minimo=dist
        if minimo >500:
            minimo=500
        dist_cerca[i]=minimo

    dist_cerca=dist_cerca.tolist()
    return(dist_cerca)


def aplica_dummies(df,lista_categoricas):
    print("aplicamos dummies") 
    if lista_categoricas!=[]:
        dummies=pd.DataFrame(index=df.index)
        for columna in lista_categoricas:
            print(f"columna: {columna}")
            dummi=pd.get_dummies(df[columna],prefix=columna)
            dummies=dummies.join(dummi)
        df_dm=df.copy()
        df_dm.drop(columns=lista_categoricas,inplace=True)
        df_dm=df_dm.join(dummies)
    else:
        df_dm=df.copy()
    return df_dm


import pandas as pd
import datetime as dt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def transforma_X(X_ruta,y_train_ruta,x_train_ruta,carpeta_datos):
        
    ruta=carpeta_datos
    # Primer dataset de entrenamiento

    # y_train es una lista con dos columnas, id y estado de la bomba
    y_train_filename=y_train_ruta
    df_ytrain=pd.read_csv(ruta+y_train_filename)

    # x_train es una lista con los datos de cada bomba, sin el estado.
    x_train_filename=x_train_ruta
    df_xtrain=pd.read_csv(ruta+x_train_filename)

    # segundo dataset que se usará en el proceso de machine learning
    x_test_filename=X_ruta
    df_xtest=pd.read_csv(ruta+x_test_filename)
    df_xtest.set_index('id')
    df_xtest.sort_index(inplace=True)

    # Fusionamos nuestras dos primeras bases de datos
    df_train=df_xtrain.merge(df_ytrain,how='inner',on='id',sort=True)
    df_train.set_index('id')
    df_train.sort_index(inplace=True)    

    # Comprobamos valores duplicados y llenamos NAN
    df_train.drop_duplicates(inplace=True)
    df_xtest.drop_duplicates(inplace=True)

    df_train=fill_object_nan(df_train)
    df_xtest=fill_object_nan(df_xtest)

    # Introducimos columnas de información para el date recorded.
    def add_date_columns(df):
        df["date_recorded"]=pd.to_datetime(df['date_recorded'],format='%Y-%m-%d',errors='coerce')
        df["date_recorded_year"]=df["date_recorded"].dt.year
        df["date_recorded_month"]=df["date_recorded"].dt.month

        # Establecemos 1999 como año de cosntrucción de aquellos bombas que no se conocen.
        df.construction_year=np.where(df.construction_year>0,df.construction_year,1999)
        df.population=np.where(df.population>0,df.population,\
            round(np.median(df.population.loc[df.population>0].mean()),0))
        return df

    df_train=add_date_columns(df_train)
    df_xtest=add_date_columns(df_xtest)

    
    for col in tipos_col(df_train)["obj"]:
        # simplificamos las variables categóricas, limitándolas a los diez valores por variable con presencia de > 2%.
        df_train[col] = agrupa_categoricas(df_train,col,10,0)
        
        # conservamos mismos tipos escogidos en xtest
        if col in df_xtest.columns:
            valores_categoricas=df_train[col].unique()
            df_xtest[col] = df_xtest[col].apply(lambda x: x if x in valores_categoricas else 'unknown')

        
    # Simplificamos el target para obtener 2 en primicia
    df_train['status_group_simpl']=df_train['status_group']
    df_train['status_group_simpl']=np.where(df_train['status_group_simpl']=='functional','functional','non functional')

    #Substutimos valores extra por "desconocidos"
    df_train.replace("other","unknown",inplace=True)
    df_xtest.replace("other","unknown",inplace=True)

    " Existen muchos valores con longitud y latitud cero. sin embargo, tienen una región asignada. Por ello, vamos a asignarles valores al azar entre los puntos de cada región"

    df_train_local=df_train.loc[df_train["longitude"]!=0]

    regiones_maximas=df_train_local[["gps_height","longitude","latitude","region_code"]].groupby(by=["region_code"]).max().add_suffix("_max")
    regiones_minimas=df_train_local[["gps_height","longitude","latitude","region_code"]].groupby(by=["region_code"]).min().add_suffix("_min")

    regiones=regiones_maximas.join(regiones_minimas)
    
    #inputamos valores de colocación basados en región en una nueva variable (df_train_one)
    df_train_zero=df_train.loc[df_train["longitude"]==0]
    df_xtest_zero=df_xtest.loc[df_xtest["longitude"]==0]

    var_imputables=["gps_height","longitude","latitude"]

    for region in regiones.index:
        for variable in var_imputables:

            min=regiones.loc[[region]][variable+"_min"].values[0]
            max=regiones.loc[[region]][variable+"_max"].values[0]

            tam_train=df_train_zero[variable].loc[df_train_zero["region_code"]==region].shape
            tam_xtest=df_xtest_zero[variable].loc[df_xtest_zero["region_code"]==region].shape

            valores_train=np.random.random_sample(size=tam_train)*(max-min)+np.ones(tam_train)*min
            valores_xtest=np.random.random_sample(size=tam_xtest)*(max-min)+np.ones(tam_xtest)*min

            df_train_zero[variable].loc[df_train_zero["region_code"]==region]=valores_train
            df_xtest_zero[variable].loc[df_xtest_zero["region_code"]==region]=valores_xtest

    df_train_one=df_train.copy()
    df_xtest_one=df_xtest.copy()

    df_train_one.loc[df_train_zero.index, var_imputables]=df_train_zero[var_imputables]
    df_xtest_one.loc[df_xtest_zero.index, var_imputables]=df_xtest_zero[var_imputables]
    
    
    # Añadimos quién es el pozo más cercano. 
    # Para esto nos basaremos UNICAMENTE en los pozos de la misma región del set de entrenamiento. 
    # Esto se dará incluso si estamos en el set de test.
    # Es un proceso largo, puede durar más de una hora. Por eso, miraré si he calculado ya el punto.

    # Miramos si ya he trabajado estos datos antes. Si no, los creamos. Esta operación puede llevar hasta una hora.
    archivo_gps_train='df_train_one_gps.pkl'
    path= Path(ruta+archivo_gps_train)
    if path.is_file():
        print("archivo de datos train gps encontrado")
        print(f"leyendo de {ruta+archivo_gps_train}")
        df_train_one_gps=joblib.load(ruta+archivo_gps_train).copy()
        df_train_one=df_train_one.join(df_train_one_gps['dist_mas_cercano'],how='inner')
    else:
        df_train_one=lugar_mas_cercano(df_train_one[["longitude","latitude","region_code"]],"gps",True)
        joblib.dump(df_train_one, ruta+archivo_gps_train)

    # Repetimos el paso con los datos del dataset objetivo.
    archivo_gps_xtest='df_xtest_prueba.pkl'
    path= Path(ruta+archivo_gps_xtest)
    if path.is_file():
        print("archivo de datos xtest gps encontrado")
        print(f"leyendo de {ruta+archivo_gps_xtest}")
        df_xtest_one_gps=joblib.load(ruta+archivo_gps_xtest)
        df_xtest_one=df_xtest_one.join(df_xtest_one_gps['dist_mas_cercano'],how='inner')
    else:
        df_xtest_one["dist_mas_cercano"]=lugar_mas_cercano_xtest(df_train_one[["longitude","latitude","region_code"]],\
            df_xtest_one[["longitude","latitude","region_code"]],"gps",True)
        joblib.dump(df_xtest_one, ruta+archivo_gps_xtest)


    #Guardamos variables, guardamos indices como indice.

    df_train_one.set_index('id',inplace=True)
    df_xtest_one.set_index('id',inplace=True)

    # Categorías a trabajar
    lista_numericas=['amount_tsh', 'construction_year', 'longitude', 'latitude', 'gps_height','population','dist_mas_cercano']
    listas_targets=['status_group','status_group_simpl']

    lista_categoricas=["quantity_group","extraction_type_class","waterpoint_type"]
    listas=[lista_numericas,lista_categoricas,listas_targets]
    listas=sum(listas,[])
    listas_test=sum([lista_numericas,lista_categoricas],[])

    df_train_two=df_train_one[listas]
    df_xtest_two=df_xtest_one[listas_test]

    # Aplicamos dummies

    print("aplicamos dummies a df_train")
    df_train_two_dm=aplica_dummies(df_train_two,lista_categoricas)
    # primero conseguido

    print("aplicamos dummies a df_xtest")
    df_xtest_three=aplica_dummies(df_xtest_two,lista_categoricas)
    print("ambos conseguidos")

    # guardamos en nueva df

    df_train_two_dm['target_trio']=df_train_two_dm['status_group'].replace(['functional', 'functional needs repair','non functional'],[0, 1,2])
    df_train_two_dm['target_duo']=df_train_two_dm['status_group_simpl'].replace(['functional', 'functional needs repair','non functional'],[0, 1,1])
    df_train_three=df_train_two_dm.drop(columns=['status_group','status_group_simpl'])

    # si hay columnas extra de dummmies en df_train_two que no hay en df_xtest, las añadimos ahora, con valor cero en todas ellas.

    col_comunes=set(df_train_three.columns).symmetric_difference(set(df_xtest_three.columns)).symmetric_difference(set(['target_duo','target_trio']))
    for col in col_comunes:
        df_xtest_three[col] = 0

    col_sin_target=list(set(df_train_three.columns).intersection(set(df_xtest_three.columns)))
    col_sin_target.sort()
    col_con_target=col_sin_target[:]
    col_con_target.extend(['target_duo','target_trio'])

    df_train_three=df_train_three[col_con_target]
    df_xtest_three=df_xtest_three[col_sin_target]

    return df_xtest_three

def predice_X(df_xtest_three,modelo_ruta,df_scores_ruta,ruta_results,nombre_salida):
    pipelines=joblib.load(ruta_results+modelo_ruta)
    scores_df=joblib.load(ruta_results+df_scores_ruta)

    X_real_test=df_xtest_three.loc[:,~df_xtest_three.columns.isin(['target_duo', 'target_trio'])]
    
    # Predecimos empleando el modelo encontrado con mejor posición.
    indice_mejor=scores_df[['Accuracy']].idxmax()
    salida_prediccion=pipelines[indice_mejor.values[0]].predict(X_real_test)


    df_xtest_three["respuesta"]=salida_prediccion
    df_xtest_three["status_group"]=df_xtest_three["respuesta"].replace([0, 1,2],['functional', 'functional needs repair','non functional'])
 
    file=ruta_results+nombre_salida
    df_xtest_three['status_group'].to_csv(file)

    return df_xtest_three['status_group']