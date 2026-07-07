# Previsão de Falhas com Rede Neural

Este projeto demonstra uma aplicação simples de Deep Learning em Engenharia. 
O objetivo é prever risco de falha em um equipamento industrial usando dados simulados de sensores, como corrente, tensão, temperatura, vibração e tempo desde a última manutenção.

O código foi desenvolvido em Python e executado pelo GitHub Actions.


## Tema

O exemplo simula dados de um motor industrial usando variaveis comuns em manutencao:

- corrente eletrica;
- tensao;
- temperatura;
- vibracao;
- dias desde a ultima manutencao.

A rede neural aprende a classificar se a condicao do equipamento indica:

- `0`: operacao normal;
- `1`: risco de falha.

## Por que isso se relaciona com Deep Learning?

Redes neurais artificiais aprendem padroes a partir de exemplos. Neste projeto, a rede recebe dados de funcionamento do equipamento e ajusta seus pesos internos para reduzir o erro entre a previsao e a classe correta.

Mesmo sendo um exemplo pequeno, ele mostra a ideia central:

```text
dados de sensores -> rede neural -> previsao de risco de falha
```

## Como executar

1. Instale o Python 3.
2. Instale a dependencia:

```bash
pip install -r requirements.txt
```

3. Execute o projeto:

```bash
python main.py
```


## O que o codigo faz

1. Gera uma base de dados simulada de equipamentos industriais.
2. Normaliza os dados de entrada.
3. Divide os dados em treino e teste.
4. Treina uma rede neural pequena com uma camada oculta.
5. Mostra a acuracia, matriz de confusao e exemplos de previsao.



## Limitacoes

Este projeto usa dados simulados, portanto nao deve ser usado para decisoes reais de manutencao. Em um sistema real, seria necessario usar dados historicos confiaveis, validar sensores, acompanhar o desempenho do modelo e ter revisao de especialistas.
