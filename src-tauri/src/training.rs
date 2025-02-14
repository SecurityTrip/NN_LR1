use crate::perceptron::{Neuron, Perceptron};
use std::fs::File;
use std::io::{BufReader, BufRead};

/// Тип для обучающего примера: (вектор пикселей, метка)
pub type Sample = (Vec<f32>, usize);

/// Загружает датасет из CSV‑файла.
/// Ожидается, что каждая строка имеет 785 значений: первое — метка (0..9),
/// далее 784 значения пикселей (0..255), разделённых запятыми.
pub fn load_dataset(path: &str) -> Result<Vec<Sample>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let tokens: Vec<&str> = line.split(',').collect();
        if tokens.len() < 785 {
            continue;
        }
        let label: usize = tokens[0]
            .trim()
            .parse::<usize>()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let mut pixels = Vec::with_capacity(784);
        for i in 1..785 {
            let p: f32 = tokens[i]
                .trim()
                .parse::<f32>()
                .map_err(|e: std::num::ParseFloatError| e.to_string())?;
            // Нормируем пиксели в диапазон [0, 1]
            pixels.push(p / 255.0);
        }
        dataset.push((pixels, label));
    }
    Ok(dataset)
}

/// Функция обучения персептрона.
/// Для каждого примера, если модель ошибается, обновляем веса двух нейронов:
/// - для правильного нейрона: увеличиваем веса и смещение;
/// - для ошибочно предсказанного нейрона: уменьшаем веса и смещение.
pub fn train_perceptron(perceptron: &mut Perceptron, dataset: &[Sample], epochs: usize, lr: f32) {
    for epoch in 0..epochs {
        let mut correct = 0;
        for (input, label) in dataset.iter() {
            let prediction = perceptron.predict(input);
            if prediction == *label {
                correct += 1;
            } else {
                // Обновление для нейрона с правильной меткой: увеличиваем веса
                {
                    let neuron = &mut perceptron.neurons[*label];
                    for j in 0..neuron.weights.len() {
                        neuron.weights[j] += lr * input[j];
                    }
                    neuron.bias += lr;
                }
                // Обновление для ошибочно предсказанного нейрона: уменьшаем веса
                {
                    let neuron = &mut perceptron.neurons[prediction];
                    for j in 0..neuron.weights.len() {
                        neuron.weights[j] -= lr * input[j];
                    }
                    neuron.bias -= lr;
                }
            }
        }
        let accuracy = correct as f32 / dataset.len() as f32;
        println!("Epoch {}: Accuracy = {:.2}%", epoch + 1, accuracy * 100.0);
    }
}
