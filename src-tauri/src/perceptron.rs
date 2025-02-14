// Модель формального нейрона и персептрона

/// Модель формального нейрона
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl Neuron {
    /// Создаёт нового нейрона с заданными весами и смещением
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias }
    }

    /// Вычисляет линейную комбинацию: sum(w_i * x_i) + bias
    pub fn linear(&self, input: &[f32]) -> f32 {
        let mut sum = self.bias;
        for (w, x) in self.weights.iter().zip(input.iter()) {
            sum += w * x;
        }
        sum
    }
}

/// Персептрон – набор нейронов, каждый из которых отвечает за один класс.
/// При предсказании выбирается нейрон с наибольшим значением выхода.
pub struct Perceptron {
    pub neurons: Vec<Neuron>,
}

impl Perceptron {
    /// Создаёт персептрон из вектора нейронов
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    /// Вычисляет выход каждого нейрона и возвращает индекс нейрона с наибольшим значением
    pub fn predict(&self, input: &[f32]) -> usize {
        let mut best_index = 0;
        let mut best_value = f32::MIN;
        for (i, neuron) in self.neurons.iter().enumerate() {
            let output = neuron.linear(input);
            if output > best_value {
                best_value = output;
                best_index = i;
            }
        }
        best_index
    }
}
