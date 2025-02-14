// Импортируем Tauri API для вызова команд из Rust
const { invoke } = window.__TAURI__.core;

let canvasEl, resultEl, recognizeBtn, clearBtn;
let datasetPathInput, epochsInput, lrInput, trainForm, trainResultEl;

window.addEventListener("DOMContentLoaded", () => {
  // Элементы для распознавания
  canvasEl = document.querySelector("#drawingArea");
  resultEl = document.querySelector("#result");
  recognizeBtn = document.querySelector("#recognize");
  clearBtn = document.querySelector("#clear");

  // Элементы для обучения
  datasetPathInput = document.querySelector("#dataset-path");
  epochsInput = document.querySelector("#epochs");
  lrInput = document.querySelector("#lr");
  trainForm = document.querySelector("#train-form");
  trainResultEl = document.querySelector("#train-result");

  // Настраиваем рисование на canvas
  let drawing = false;
  const ctx = canvasEl.getContext("2d");
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;

  canvasEl.addEventListener("mousedown", (e) => {
    drawing = true;
    ctx.beginPath();
    const { x, y } = getCursorPosition(e);
    ctx.moveTo(x, y);
  });
  canvasEl.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const { x, y } = getCursorPosition(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  });
  canvasEl.addEventListener("mouseup", () => (drawing = false));
  canvasEl.addEventListener("mouseleave", () => (drawing = false));

  function getCursorPosition(e) {
    const rect = canvasEl.getBoundingClientRect();
    // Вычисляем соотношение между атрибутами canvas (canvasEl.width/height)
    // и размерами, полученными из rect
    const scaleX = canvasEl.width / rect.width;
    const scaleY = canvasEl.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  // Очистка canvas
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    resultEl.textContent = "";
  });

  // Распознавание цифры
  recognizeBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    // Создаём оффскрин‑canvas для уменьшения изображения до 28x28
    const offscreen = document.createElement("canvas");
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext("2d");
    offCtx.drawImage(canvasEl, 0, 0, 28, 28);
    // Получаем данные пикселей
    const imageData = offCtx.getImageData(0, 0, 28, 28).data;
    let pixels = [];
    for (let i = 0; i < imageData.length; i += 4) {
      // Берём значение красного канала (при условии, что рисуем белым на чёрном фоне)
      let gray = imageData[i] / 255;
      pixels.push(gray);
    }
    if (pixels.length !== 28 * 28) {
      resultEl.textContent = "Ошибка: неверное количество пикселей";
      return;
    }
    try {
      const digit = await invoke("recognize_digit", { pixelData: pixels });
      resultEl.textContent = `Распознанная цифра: ${digit}`;
    } catch (error) {
      console.error("Ошибка при распознавании:", error);
      resultEl.textContent = "Ошибка распознавания";
    }
  });

  // Обучение модели: обработка формы
  trainForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const datasetPath = datasetPathInput.value.trim();
    const epochs = parseInt(epochsInput.value);
    const lr = parseFloat(lrInput.value);

    try {
      trainResultEl.textContent = "Обучение...";
      const result = await invoke("train_model", {
        datasetPath,
        epochs,
        lr,
      });
      trainResultEl.textContent = result;
    } catch (error) {
      console.error("Ошибка при обучении:", error);
      trainResultEl.textContent = "Ошибка обучения";
    }
  });
});
