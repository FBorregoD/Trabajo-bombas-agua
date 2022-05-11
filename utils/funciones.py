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
    # devuelve una columna donde se han simplificado aquellos valores de esa columna, eliminado aquellos con existencia por debajo de 4% y quedándose sólo con los cinco primeros.
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

    fig, ax = plt.subplots(figsize = (size,size))
    groups = df.groupby(grupo)
    img_map = plt.imread(ruta)
    for name, group in groups:
        ax.scatter(group.longitude, group.latitude, label=name, marker='o',zorder=1, alpha= 0.8, s=13)
    ax.legend(numpoints=1)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(img_map, zorder=0, extent = BBox, aspect= 'equal')