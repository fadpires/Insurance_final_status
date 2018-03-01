import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#Abrir base de dados - Casos Encerrados

zurich = pd.read_csv('/home/sbk2/Desktop/Zurich/Final/train1.csv', decimal=',')

#Usar apenas dois decimais após a virgula
pd.options.display.float_format = '{:.2f}'.format

#Criação de um subset de colunas apenas com features relevantes para a "Questão de Negócio"

zurich_categ = zurich[['Ano','Mês Encerramento','Segmento','Produto','Cobertura Reclamada','Motivo da Ação','Motivo Seguradora','Status Encerramento','Classificação da Contingência','Objeto']]

#Padronização dos dados - caixa baixa e retirada de espaços extras
zurich_categ = zurich_categ.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
zurich_categ = zurich_categ.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#Inicio da transformação dos dados para numéricos

zurich_categ['Segmento'] = zurich_categ['Segmento'].replace('life', 1)
zurich_categ['Segmento'] = zurich_categ['Segmento'].replace('gi', 2)

zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('indenização parcial', 'sinistro liquidado parcialmente')
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace(['notificação sem regulação','negado pré-analise','ctg reportada.','notificação não atendida'], 'outros')
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('sinistro negado', 1)
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('sinistro liquidado parcialmente', 2)
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('sinistro não avisado', 3)
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('sinistro liquidado', 4)
zurich_categ['Motivo da Ação'] = zurich_categ['Motivo da Ação'].replace('outros', 5)

zurich_categ = zurich_categ[zurich_categ['Classificação da Contingência'] != 'remoto' ]
zurich_categ = zurich_categ[zurich_categ['Classificação da Contingência'] != '0' ]

zurich_categ['Classificação da Contingência'] = zurich_categ['Classificação da Contingência'].replace('condenação', 1)
zurich_categ['Classificação da Contingência'] = zurich_categ['Classificação da Contingência'].replace('sucumbência', 2)

zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('seguro acidentes pessoais', 'acidentes pessoais')
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('seguro de automóvel', 'seguro automóvel')
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace(['cartão de crédito','seguro de saúde'], 'seguros diversos')
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace(['seguro automóvel','seguro prestamista','seguro empresarial','previdência privada'], 'seguros diversos')

zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('seguro de vida', 1)
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('seguro residencial', 2)
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('acidentes pessoais', 3)
zurich_categ['Objeto'] = zurich_categ['Objeto'].replace('seguros diversos', 4)

#Features que utilizaremos no modelo
features_teste = list(zurich_categ.columns[[2,5,8,9]])
print("* features_teste:", features_teste, sep="\n")


#Outro modo de transformar uma coluna em numéricos
def encode_target(df, target_column):

    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

zurich_categ_target, targets = encode_target(zurich_categ, "Status Encerramento")
print("* zurich_categ_target.head()", zurich_categ_target[["Target", "Status Encerramento"]].head(20),
      sep="\n", end="\n\n")



#Modelo de arvore de decisão

y = zurich_categ_target["Target"]
X = zurich_categ_target[features_teste]
dt = DecisionTreeClassifier(min_samples_split=50, random_state=99)
dt.fit(X, y)

#Vizualisação da árvore criada
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


visualize_tree(dt, features_teste)

#Inicio segunda parte

final_abrir = pd.read_csv('/home/sbk2/Desktop/Zurich/Final/limpo_2.csv', decimal=',', encoding='utf-8')
pd.options.display.float_format = '{:.2f}'.format

final = final_abrir[['Segmento','Motivo da Ação','Classificação da Contingência','Objeto', 'Status Encerramento','Nº do processo']]
final = final.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
final = final.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

final['Segmento'] = final['Segmento'].replace('life', 1)
final['Segmento'] = final['Segmento'].replace('gi', 2)

final['Motivo da Ação'] = final['Motivo da Ação'].replace('indenização parcial', 'sinistro liquidado parcialmente')
final['Motivo da Ação'] = final['Motivo da Ação'].replace(['notificação sem regulação','negado pré-analise','ctg reportada.','notificação não atendida'], 'outros')
final['Motivo da Ação'] = final['Motivo da Ação'].replace('sinistro negado', 1)
final['Motivo da Ação'] = final['Motivo da Ação'].replace('sinistro liquidado parcialmente', 2)
final['Motivo da Ação'] = final['Motivo da Ação'].replace('sinistro não avisado', 3)
final['Motivo da Ação'] = final['Motivo da Ação'].replace('sinistro liquidado', 4)
final['Motivo da Ação'] = final['Motivo da Ação'].replace('outros', 5)

final = final[final['Classificação da Contingência'] != 'remoto' ]
final = final[final['Classificação da Contingência'] != '0' ]
final['Classificação da Contingência'] = final['Classificação da Contingência'].replace('condenação', 1)
final['Classificação da Contingência'] = final['Classificação da Contingência'].replace('sucumbência', 2)

final['Objeto'] = final['Objeto'].replace('seguro acidentes pessoais', 'acidentes pessoais')
final['Objeto'] = final['Objeto'].replace('seguro de automóvel', 'seguro automóvel')
final['Objeto'] = final['Objeto'].replace(['cartão de crédito','seguro de saúde'], 'seguros diversos')
final['Objeto'] = final['Objeto'].replace(['seguro automóvel','seguro prestamista','seguro empresarial','previdência privada'], 'seguros diversos')

final['Objeto'] = final['Objeto'].replace('seguro de vida', 1)
final['Objeto'] = final['Objeto'].replace('seguro residencial', 2)
final['Objeto'] = final['Objeto'].replace('acidentes pessoais', 3)
final['Objeto'] = final['Objeto'].replace('seguros diversos', 4)

final['Status Encerramento'] = final['Status Encerramento'].replace('acordo', 0)
final['Status Encerramento'] = final['Status Encerramento'].replace('pagamento zero', 1)
final['Status Encerramento'] = final['Status Encerramento'].replace('procedência parcial', 2)
final['Status Encerramento'] = final['Status Encerramento'].replace('procedência total', 3)

final = final.dropna()

previsao_final = final['Nº do processo'] 

x_final = final[['Segmento','Motivo da Ação','Classificação da Contingência', 'Objeto']]

final['Previsao'] = dt.predict(x_final)

prediction = pd.DataFrame(final, columns=['Nº do processo','Previsao']).to_csv('prediction.csv')
