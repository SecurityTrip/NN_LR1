#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::Mutex;
use tauri::Manager;

mod perceptron;
mod training;

use perceptron::{Neuron, Perceptron};
use training::{load_dataset, train_perceptron};

struct AppState {
    model: Mutex<Perceptron>,
}

const TRAIN_DATASET_PATH: &str = "D:/university/6_term/neural networks/NN_LR1/src-tauri/src/mnist_train.csv";
const TEST_DATASET_PATH: &str = "D:/university/6_term/neural networks/NN_LR1/src-tauri/src/mnist_test.csv";

fn simple_random(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32) / (u32::MAX as f32)
}

fn init_model() -> Perceptron {
    let mut neurons = Vec::new();
    let mut seed = 42u32;
    for _ in 0..10 {
        let weights: Vec<f32> = (0..(28 * 28))
            .map(|_| (simple_random(&mut seed) - 0.5) * 0.02)
            .collect();
        let bias = (simple_random(&mut seed) - 0.5) * 0.02;
        neurons.push(Neuron::new(weights, bias));
    }
    Perceptron::new(neurons)
}

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

#[tauri::command]
fn train_model(epochs: usize, lr: f32, state: tauri::State<AppState>) -> Result<String, String> {
    let dataset = load_dataset(TRAIN_DATASET_PATH)?;
    println!("Загружено обучающих примеров: {}", dataset.len());

    let mut model = state.model.lock().unwrap();

    train_perceptron(&mut model, &dataset, epochs, lr);

    // Прогон на обучающем датасете
    let mut correct = 0;
    let mut per_class_correct = vec![0; 10];
    let mut per_class_total = vec![0; 10];
    for (input, label) in dataset.iter() {
        let prediction = model.predict(input);
        per_class_total[*label] += 1;
        if prediction == *label {
            correct += 1;
            per_class_correct[*label] += 1;
        }
    }
    let train_accuracy = correct as f32 / dataset.len() as f32 * 100.0;

    // Прогон на тестовом датасете
    let testset = load_dataset(TEST_DATASET_PATH)?;
    println!("Загружено тестовых примеров: {}", testset.len());

    let mut test_correct = 0;
    let mut test_per_class_correct = vec![0; 10];
    let mut test_per_class_total = vec![0; 10];
    for (input, label) in testset.iter() {
        let prediction = model.predict(input);
        test_per_class_total[*label] += 1;
        if prediction == *label {
            test_correct += 1;
            test_per_class_correct[*label] += 1;
        }
    }
    let test_accuracy = test_correct as f32 / testset.len() as f32 * 100.0;

    let mut report = format!(
        "Обучение завершено.\nТочность на обучающем наборе: {:.2}%\nТочность на тестовом наборе: {:.2}%\n",
        train_accuracy, test_accuracy
    );

    report.push_str("Результаты по классам (тестовый набор):\n");
    for i in 0..10 {
        let class_accuracy = if test_per_class_total[i] > 0 {
            test_per_class_correct[i] as f32 / test_per_class_total[i] as f32 * 100.0
        } else {
            0.0
        };
        report.push_str(&format!(
            "Цифра {}: {}/{} ({:.2}%)\n",
            i, test_per_class_correct[i], test_per_class_total[i], class_accuracy
        ));
    }

    Ok(report)
}

fn main() {
    let state = AppState {
        model: Mutex::new(init_model()),
    };

    tauri::Builder::default()
        .manage(state)
        .invoke_handler(tauri::generate_handler![recognize_digit, train_model])
        .run(tauri::generate_context!())
        .expect("Ошибка при запуске приложения");
}
