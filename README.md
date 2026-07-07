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

## Relatório Técnico com IA Generativa

Além da rede neural para previsão de falhas, o projeto também possui uma segunda etapa voltada para IA Generativa.

Essa etapa está no arquivo `relatorio_ia_generativa.py`.

O objetivo desse arquivo é gerar um relatório técnico de manutenção a partir dos dados do equipamento analisado.

### Como Funciona

O código utiliza os seguintes dados do equipamento:

- corrente elétrica;
- tensão;
- temperatura;
- vibração;
- dias desde a última manutenção;
- probabilidade estimada de falha;
- classificação do equipamento.

Com essas informações, o programa monta um prompt técnico e tenta gerar um relatório usando uma API de IA Generativa.

O relatório contém:

1. Diagnóstico do equipamento;
2. Possíveis causas do risco;
3. Recomendações de manutenção;
4. Nível de urgência.

### Tratamento de Erro da API

Caso a chave da API não esteja configurada ou a conta esteja sem cota disponível, o código não interrompe a execução.

Nesse caso, ele gera um relatório local automaticamente, usando os mesmos dados do equipamento.

Isso permite que o projeto continue funcionando mesmo sem acesso ativo à API.

### Exemplo de Saída

```text
RELATORIO TECNICO DE MANUTENCAO

Diagnostico:
O equipamento apresenta risco de falha, com probabilidade estimada de falha de 92.16%.

Possiveis causas:
A temperatura de 96 C, a vibracao de 5.2 mm/s e o periodo de 145 dias sem manutencao indicam uma condicao operacional critica.

Recomendacoes:
Recomenda-se realizar inspecao preventiva, verificar rolamentos, conexoes eletricas, sistema de ventilacao e historico de manutencao do equipamento.

Nivel de urgencia:
Alto. O equipamento deve ser avaliado antes de continuar operando por longos periodos.

```

## Limitações

Este projeto usa dados simulados, portanto nao deve ser usado para decisoes reais de manutencao. Em um sistema real, seria necessario usar dados historicos confiaveis, validar sensores, acompanhar o desempenho do modelo e ter revisao de especialistas.
