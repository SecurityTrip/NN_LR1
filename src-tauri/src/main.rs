#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// Подключаем только Tauri (сторонних библиотек для нейросети или обработки изображений не используем)
use tauri::Manager;

#[tauri::command]
fn recognize_digit(pixelData: Vec<f32>) -> Result<String, String> {
    // Ожидаем, что входной вектор имеет длину 28x28 = 784
    if pixelData.len() != 28 * 28 {
        return Err(format!(
            "Неверная длина входных данных: ожидалось 784, получено {}",
            pixelData.len()
        ));
    }

    // Выполняем прямой проход по нейросети
    let prediction = forward(&pixelData);
    Ok(prediction.to_string())
}

/// Функция, выполняющая прямой проход по сети
fn forward(input: &[f32]) -> usize {
    // Скрытый слой: 128 нейронов
    let w1 = get_w1();
    let b1 = get_b1();
    let hidden = linear_layer(&w1, &b1, 128, 28 * 28, input);
    // Функция активации ReLU
    let hidden_relu: Vec<f32> = hidden.into_iter().map(|x| if x < 0.0 { 0.0 } else { x }).collect();

    // Выходной слой: 10 нейронов
    let w2 = get_w2();
    let b2 = get_b2();
    let output = linear_layer(&w2, &b2, 10, 128, &hidden_relu);

    // Возвращаем индекс максимального значения – это и будет распознанная цифра
    argmax(&output)
}

/// Реализует линейный слой: y = W*x + b  
/// - weights имеет размер out_size x in_size  
/// - bias имеет длину out_size  
fn linear_layer(weights: &[f32], bias: &[f32], out_size: usize, in_size: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; out_size];
    for i in 0..out_size {
        let mut sum = bias[i];
        for j in 0..in_size {
            sum += weights[i * in_size + j] * input[j];
        }
        output[i] = sum;
    }
    output
}

/// Возвращает индекс максимального элемента в векторе
fn argmax(vec: &[f32]) -> usize {
    let mut max_index = 0;
    let mut max_value = vec[0];
    for (i, &value) in vec.iter().enumerate().skip(1) {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }
    max_index
}

/// Для демонстрации задаём фиксированные веса и смещения.
/// В реальном приложении здесь должны быть обученные параметры.
fn get_w1() -> Vec<f32> {
    // Размер: 128 x (28*28) = 128 x 784
    vec![0.001; 128 * 28 * 28]
}

fn get_b1() -> Vec<f32> {
    vec![0.0; 128]
}

fn get_w2() -> Vec<f32> {
    // Размер: 10 x 128
    vec![0.001; 10 * 128]
}

fn get_b2() -> Vec<f32> {
    vec![0.0; 10]
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![recognize_digit])
        .run(tauri::generate_context!())
        .expect("Ошибка при запуске приложения");
}
