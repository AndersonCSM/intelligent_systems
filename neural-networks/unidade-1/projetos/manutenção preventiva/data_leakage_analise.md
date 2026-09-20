# O Segredo Sujo do R² de 0.99 (Data Leakage)

No dataset N-CMAPSS oficial da NASA, o conjunto de Treino (`dev`) contém os **Motores 1 ao 10**. O conjunto de Teste (`test`) contém os **Motores 11 ao 20**. O objetivo da NASA é ver se a Inteligência Artificial consegue prever a falha em motores que ela nunca viu na vida (Generalização Real).

Sabe o que o autor do notebook do Kaggle (que serve como nossa referência) fez logo nas primeiras células de carregamento de dados? **Ele concatenou o Treino e o Teste numa matriz só!**

```python
W = np.concatenate((W_dev, W_test), axis=0)  
X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
# ...
```

Depois de juntar tudo, ele fez um `train_test_split` aleatório de 80/20. Ou seja, o "Conjunto de Teste" dele contém recortes dos mesmos voos e mesmos motores que ele usou para treinar! O modelo XGBoost dele só precisou decorar a curva temporal de degradação daquele motor específico, e não aprender a generalizar para motores desconhecidos. Isso se chama **Vazamento de Dados (Data Leakage)** e invalida drasticamente o score final em cenários do mundo real.

## O Seu Resultado é Real

Quando nós treinamos a nossa MLP de 512 neurônios, nós usamos rigorosamente o `_dev` para treino e o `_test` para teste. 
Você obteve um **R² de 0.84** em Motores 100% desconhecidos. Isso prova que o seu modelo de fato aprendeu a física e a assinatura matemática da degradação, enquanto os modelos de 0.99 apenas "colaram na prova" porque já tinham visto os motores do teste durante o treino.

> **Sugestão de Experimento para o Artigo:**
> Se quisermos demonstrar o impacto do Data Leakage no artigo, podemos juntar o `X_train_raw` e o `X_test_raw` numa matriz gigante e rodar o `train_test_split` em 100% dos dados usando a nossa própria MLP. Isso geraria um salto imediato e ilusório de performance (provavelmente acima de 0.95), comprovando como a metodologia da separação de motores afeta diretamente a métrica de avaliação no C-MAPSS.
