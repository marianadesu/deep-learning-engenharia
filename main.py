import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def create_dataset(samples=1200, seed=42):
    rng = np.random.default_rng(seed)

    current = rng.normal(45, 8, samples)
    voltage = rng.normal(220, 8, samples)
    temperature = rng.normal(72, 13, samples)
    vibration = rng.normal(3.0, 1.1, samples)
    days_since_maintenance = rng.integers(1, 180, samples)

    risk_score = (
        0.04 * (current - 45)
        + 0.08 * (temperature - 72)
        + 0.85 * (vibration - 3.0)
        + 0.012 * (days_since_maintenance - 90)
        + 0.03 * np.abs(voltage - 220)
    )

    noise = rng.normal(0, 0.8, samples)
    failure = (risk_score + noise > 1.35).astype(int)

    features = np.column_stack(
        [current, voltage, temperature, vibration, days_since_maintenance]
    )

    return features, failure.reshape(-1, 1)


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


def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1

    return (train - mean) / std, (test - mean) / std, mean, std


def train_neural_network(x_train, y_train, epochs=3000, learning_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    samples = x_train.shape[0]
    input_neurons = x_train.shape[1]
    hidden_neurons = 8
    output_neurons = 1

    w1 = rng.normal(0, 0.15, (input_neurons, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    w2 = rng.normal(0, 0.15, (hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        hidden_input = np.dot(x_train, w1) + b1
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, w2) + b2
        predictions = sigmoid(final_input)

        output_delta = predictions - y_train
        hidden_delta = np.dot(output_delta, w2.T) * sigmoid_derivative(hidden_output)

        w2 -= learning_rate * np.dot(hidden_output.T, output_delta) / samples
        b2 -= learning_rate * output_delta.sum(axis=0, keepdims=True) / samples
        w1 -= learning_rate * np.dot(x_train.T, hidden_delta) / samples
        b1 -= learning_rate * hidden_delta.sum(axis=0, keepdims=True) / samples

        if (epoch + 1) % 500 == 0:
            eps = 1e-8
            loss = -np.mean(
                y_train * np.log(predictions + eps)
                + (1 - y_train) * np.log(1 - predictions + eps)
            )
            print(f"Epoca {epoch + 1:4d} | perda: {loss:.4f}")

    return w1, b1, w2, b2


def predict(x, weights):
    w1, b1, w2, b2 = weights
    hidden_output = sigmoid(np.dot(x, w1) + b1)
    probability = sigmoid(np.dot(hidden_output, w2) + b2)
    return probability, (probability >= 0.5).astype(int)


def confusion_matrix(y_true, y_pred):
    true = y_true.flatten()
    pred = y_pred.flatten()

    tp = np.sum((true == 1) & (pred == 1))
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))

    return tp, tn, fp, fn


def classify_new_equipment(data, mean, std, weights):
    normalized = (np.array(data, dtype=float) - mean) / std
    probability, prediction = predict(normalized.reshape(1, -1), weights)
    return probability[0, 0], prediction[0, 0]


def main():
    features, labels = create_dataset()

    x_train, x_test, y_train, y_test = train_test_split(features, labels)
    x_train, x_test, mean, std = standardize(x_train, x_test)

    print("Treinando rede neural para prever risco de falha...\n")
    weights = train_neural_network(x_train, y_train)

    probabilities, y_pred = predict(x_test, weights)
    accuracy = np.mean(y_pred == y_test)
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)

    print("\nResultados no conjunto de teste")
    print(f"Acuracia: {accuracy * 100:.2f}%")
    print("\nMatriz de confusao")
    print(f"Verdadeiro positivo: {tp}")
    print(f"Verdadeiro negativo: {tn}")
    print(f"Falso positivo:      {fp}")
    print(f"Falso negativo:      {fn}")

    example = [58, 214, 96, 5.2, 145]
    probability, prediction = classify_new_equipment(example, mean, std, weights)

    print("\nExemplo de equipamento analisado")
    print("Dados: corrente=58 A, tensao=214 V, temperatura=96 C, vibracao=5.2 mm/s, manutencao=145 dias")
    print(f"Probabilidade estimada de falha: {probability * 100:.2f}%")
    print("Classificacao:", "risco de falha" if prediction == 1 else "operacao normal")


if __name__ == "__main__":
    main()
