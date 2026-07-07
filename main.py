Sim! Você pode substituir o `main.py` por essa versão comentada. Ela faz a mesma coisa, só que com explicações usando `#`:

```python
import numpy as np

# Funcao de ativacao sigmoid.
# Ela transforma qualquer valor em um numero entre 0 e 1.
# Isso e util porque a saida da rede sera uma probabilidade de falha.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada da sigmoid.
# E usada no treinamento para ajustar os pesos da rede neural.
def sigmoid_derivative(x):
    return x * (1 - x)


# Cria uma base de dados simulada para equipamentos industriais.
# Cada linha representa um equipamento com dados de sensores.
def create_dataset(samples=1200, seed=42):
    rng = np.random.default_rng(seed)

    # Dados simulados dos sensores:
    # corrente em amperes, tensao em volts, temperatura em graus Celsius,
    # vibracao em mm/s e dias desde a ultima manutencao.
    current = rng.normal(45, 8, samples)
    voltage = rng.normal(220, 8, samples)
    temperature = rng.normal(72, 13, samples)
    vibration = rng.normal(3.0, 1.1, samples)
    days_since_maintenance = rng.integers(1, 180, samples)

    # Calcula uma pontuacao de risco.
    # Temperatura alta, vibracao alta, muitos dias sem manutencao
    # e tensao fora do valor ideal aumentam o risco de falha.
    risk_score = (
        0.04 * (current - 45)
        + 0.08 * (temperature - 72)
        + 0.85 * (vibration - 3.0)
        + 0.012 * (days_since_maintenance - 90)
        + 0.03 * np.abs(voltage - 220)
    )

    # Adiciona ruido para deixar os dados mais parecidos com uma situacao real.
    noise = rng.normal(0, 0.8, samples)

    # Se a pontuacao de risco passar do limite, classificamos como falha.
    # 0 = operacao normal
    # 1 = risco de falha
    failure = (risk_score + noise > 1.35).astype(int)

    # Junta todas as variaveis de entrada em uma matriz.
    features = np.column_stack(
        [current, voltage, temperature, vibration, days_since_maintenance]
    )

    return features, failure.reshape(-1, 1)


# Divide os dados em treino e teste.
# Treino: usado para a rede aprender.
# Teste: usado para verificar se a rede aprendeu bem.
def train_test_split(features, labels, test_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(features))
    rng.shuffle(indices)

    test_size = int(len(features) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return (
        features[train_indices],
        features[test_indices],
        labels[train_indices],
        labels[test_indices],
    )


# Normaliza os dados para ficarem em escalas semelhantes.
# Isso evita que variaveis com numeros maiores, como tensao,
# dominem o treinamento apenas por causa da escala.
def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1

    return (train - mean) / std, (test - mean) / std, mean, std


# Treina uma rede neural simples com:
# 5 entradas, 8 neuronios na camada oculta e 1 saida.
def train_neural_network(x_train, y_train, epochs=3000, learning_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)

    samples = x_train.shape[0]
    input_neurons = x_train.shape[1]
    hidden_neurons = 8
    output_neurons = 1

    # Inicializacao dos pesos e bias.
    # Os pesos comecam aleatorios e vao sendo ajustados durante o treinamento.
    w1 = rng.normal(0, 0.15, (input_neurons, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    w2 = rng.normal(0, 0.15, (hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))

    # Loop de treinamento.
    # Cada epoca representa uma passagem pelos dados de treino.
    for epoch in range(epochs):
        # Propagacao para frente: a rede faz uma previsao.
        hidden_input = np.dot(x_train, w1) + b1
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, w2) + b2
        predictions = sigmoid(final_input)

        # Calcula o erro entre a previsao e o valor correto.
        output_delta = predictions - y_train

        # Calcula quanto a camada oculta contribuiu para o erro.
        hidden_delta = np.dot(output_delta, w2.T) * sigmoid_derivative(hidden_output)

        # Atualiza os pesos e bias para reduzir o erro.
        w2 -= learning_rate * np.dot(hidden_output.T, output_delta) / samples
        b2 -= learning_rate * output_delta.sum(axis=0, keepdims=True) / samples
        w1 -= learning_rate * np.dot(x_train.T, hidden_delta) / samples
        b1 -= learning_rate * hidden_delta.sum(axis=0, keepdims=True) / samples

        # A cada 500 epocas, mostra a perda do modelo.
        if (epoch + 1) % 500 == 0:
            eps = 1e-8
            loss = -np.mean(
                y_train * np.log(predictions + eps)
                + (1 - y_train) * np.log(1 - predictions + eps)
            )
            print(f"Epoca {epoch + 1:4d} | perda: {loss:.4f}")

    return w1, b1, w2, b2


# Usa a rede treinada para fazer previsoes.
def predict(x, weights):
    w1, b1, w2, b2 = weights

    hidden_output = sigmoid(np.dot(x, w1) + b1)
    probability = sigmoid(np.dot(hidden_output, w2) + b2)

    # Se a probabilidade for maior ou igual a 0.5,
    # classifica como risco de falha.
    return probability, (probability >= 0.5).astype(int)


# Calcula a matriz de confusao.
# Ela mostra os acertos e erros do modelo.
def confusion_matrix(y_true, y_pred):
    true = y_true.flatten()
    pred = y_pred.flatten()

    tp = np.sum((true == 1) & (pred == 1))
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))

    return tp, tn, fp, fn


# Classifica um novo equipamento usando a rede ja treinada.
def classify_new_equipment(data, mean, std, weights):
    normalized = (np.array(data, dtype=float) - mean) / std
    probability, prediction = predict(normalized.reshape(1, -1), weights)

    return probability[0, 0], prediction[0, 0]


def main():
    # Cria os dados simulados.
    features, labels = create_dataset()

    # Divide os dados em treino e teste.
    x_train, x_test, y_train, y_test = train_test_split(features, labels)

    # Normaliza os dados.
    x_train, x_test, mean, std = standardize(x_train, x_test)

    print("Treinando rede neural para prever risco de falha...\n")

    # Treina a rede neural.
    weights = train_neural_network(x_train, y_train)

    # Faz previsoes nos dados de teste.
    probabilities, y_pred = predict(x_test, weights)

    # Calcula a acuracia.
    accuracy = np.mean(y_pred == y_test)

    # Calcula a matriz de confusao.
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

    print("\nResultados no conjunto de teste")
    print(f"Acuracia: {accuracy * 100:.2f}%")

    print("\nMatriz de confusao")
    print(f"Verdadeiro positivo: {tp}")
    print(f"Verdadeiro negativo: {tn}")
    print(f"Falso positivo:      {fp}")
    print(f"Falso negativo:      {fn}")

    # Exemplo de um equipamento especifico para classificar.
    # Dados: corrente, tensao, temperatura, vibracao e dias sem manutencao.
    example = [58, 214, 96, 5.2, 145]

    probability, prediction = classify_new_equipment(example, mean, std, weights)

    print("\nExemplo de equipamento analisado")
    print(
        "Dados: corrente=58 A, tensao=214 V, temperatura=96 C, "
        "vibracao=5.2 mm/s, manutencao=145 dias"
    )
    print(f"Probabilidade estimada de falha: {probability * 100:.2f}%")
    print("Classificacao:", "risco de falha" if prediction == 1 else "operacao normal")


# Garante que a funcao main so sera executada quando o arquivo for rodado diretamente.
if __name__ == "__main__":
    main()
```

No GitHub, é só abrir o `main.py`, clicar no lápis, substituir pelo código comentado e dar **Commit changes**.

