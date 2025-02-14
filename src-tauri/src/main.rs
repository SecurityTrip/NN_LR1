#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// Подключаем необходимые модули и библиотеки
use std::sync::Mutex;
use tauri::Manager;

mod perceptron;
mod training;

use perceptron::{Neuron, Perceptron};
use training::{load_dataset, train_perceptron};

/// Глобальное состояние приложения.
/// Модель хранится в Mutex, чтобы обеспечить безопасный доступ из разных команд.
struct AppState {
    model: Mutex<Perceptron>,
}

/// Инициализация персептрона с 10 нейронами, каждый с 784 входами.
/// Начальные веса и смещения выставлены в 0.0.
fn init_model() -> Perceptron {
    let mut neurons = Vec::new();
    for _ in 0..10 {
        neurons.push(Neuron::new(vec![0.0; 28 * 28], 0.0));
    }
    Perceptron::new(neurons)
}

/// Tauri-команда для распознавания цифры по входному вектору (784 значения).
/// Использует модель из глобального состояния.
#[tauri::command]
fn recognize_digit(pixelData: Vec<f32>, state: tauri::State<AppState>) -> Result<String, String> {
    if pixelData.len() != 28 * 28 {
        return Err(format!(
            "Неверное количество пикселей: {} (ожидалось 784)",
            pixelData.len()
        ));
    }
    let model = state.model.lock().unwrap();
    let prediction = model.predict(&pixelData);
    Ok(prediction.to_string())
}

/// Tauri-команда для обучения модели.
/// Параметры:
/// - dataset_path: путь к CSV‑файлу с датасетом (например, "mnist_train.csv")
/// - epochs: число эпох обучения
/// - lr: скорость обучения
/// После обучения обновляет модель в глобальном состоянии.
#[tauri::command]
fn train_model(
    dataset_path: String,
    epochs: usize,
    lr: f32,
    state: tauri::State<AppState>,
) -> Result<String, String> {
    let dataset = load_dataset(&dataset_path)?;
    println!("Загружено примеров: {}", dataset.len());

    // Блокируем состояние и получаем изменяемую ссылку на модель
    let mut model = state.model.lock().unwrap();

    // Обучаем модель
    train_perceptron(&mut model, &dataset, epochs, lr);

    // Оценка точности на обучающем наборе
    let mut correct = 0;
    for (input, label) in dataset.iter() {
        let prediction = model.predict(input);
        if prediction == *label {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / dataset.len() as f32;
    Ok(format!(
        "Обучение завершено. Точность на обучающем наборе: {:.2}%",
        accuracy * 100.0
    ))
}

fn main() {
    // Инициализируем модель и передаём её в глобальное состояние
    let state = AppState {
        model: Mutex::new(init_model()),
    };

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![recognize_digit, train_model])
        .run(tauri::generate_context!())
        .expect("Ошибка при запуске приложения");
}
