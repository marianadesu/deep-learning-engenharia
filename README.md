# Previsao de Falhas em Equipamentos com Rede Neural

Projeto simples para demonstrar, em um contexto de Engenharia, como uma rede neural pode aprender padroes em dados de sensores e estimar risco de falha em equipamentos industriais.

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

## Como colocar no GitHub

Depois de criar um repositorio vazio no GitHub, execute estes comandos dentro da pasta do projeto:

```bash
git init
git add .
git commit -m "Projeto de deep learning aplicado a engenharia"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
git push -u origin main
```

Troque `SEU-USUARIO` e `NOME-DO-REPOSITORIO` pelos dados do seu GitHub.

## O que o codigo faz

1. Gera uma base de dados simulada de equipamentos industriais.
2. Normaliza os dados de entrada.
3. Divide os dados em treino e teste.
4. Treina uma rede neural pequena com uma camada oculta.
5. Mostra a acuracia, matriz de confusao e exemplos de previsao.

## Exemplo de uso em apresentacao

Este codigo pode ser apresentado como uma aplicacao pratica de Deep Learning em Engenharia:

> "A rede neural observa variaveis de sensores e aprende padroes associados a falhas. Em uma aplicacao real, isso poderia apoiar manutencao preditiva, ajudando a reduzir paradas inesperadas e aumentar a seguranca operacional."

## Limitacoes

Este projeto usa dados simulados, portanto nao deve ser usado para decisoes reais de manutencao. Em um sistema real, seria necessario usar dados historicos confiaveis, validar sensores, acompanhar o desempenho do modelo e ter revisao de especialistas.
