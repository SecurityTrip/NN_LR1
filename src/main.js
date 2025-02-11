const { invoke } = window.__TAURI__.core;

let canvasEl;
let resultEl;
let recognizeBtn;
let clearBtn;

window.addEventListener("DOMContentLoaded", () => {
  // Получаем элементы из DOM
  canvasEl = document.querySelector("#drawingArea");
  resultEl = document.querySelector("#result");
  recognizeBtn = document.querySelector("#recognize");
  clearBtn = document.querySelector("#clear");

  // Устанавливаем реальные размеры canvas (если они заданы через CSS)
  canvasEl.width = canvasEl.offsetWidth;
  canvasEl.height = canvasEl.offsetHeight;

  // Обработчик для кнопки "Распознать"
  recognizeBtn.addEventListener("click", async (e) => {
    e.preventDefault();

    // Создаём оффскрин‑canvas размером 28x28
    const offscreen = document.createElement("canvas");
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext("2d");

    // Рисуем основной canvas в оффскрин с масштабированием до 28x28
    offCtx.drawImage(canvasEl, 0, 0, 28, 28);

    // Получаем данные пикселей (формат RGBA)
    const imageData = offCtx.getImageData(0, 0, 28, 28).data;

    // Преобразуем в градации серого: для каждого пикселя возьмём значение красного канала,
    // нормализуем его (делим на 255) и сформируем массив из 784 значений.
    let pixels = [];
    for (let i = 0; i < imageData.length; i += 4) {
      // Можно взять среднее значение из R, G, B – здесь берём R, так как рисуем белым на чёрном фоне.
      let gray = imageData[i] / 255;
      pixels.push(gray);
    }

    // Проверяем, что получено ровно 784 значения (28x28)
    if (pixels.length !== 28 * 28) {
      resultEl.textContent = "Ошибка: неверное количество пикселей";
      return;
    }

    try {
      // Вызываем Tauri-команду "recognize_digit" и передаём массив пикселей
      const digit = await invoke("recognize_digit", { pixelData: pixels });
      resultEl.textContent = `Распознанная цифра: ${digit}`;
    } catch (error) {
      console.error("Ошибка при распознавании:", error);
      resultEl.textContent = "Ошибка распознавания";
    }
  });

  // Обработчик для кнопки "Очистить"
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    const ctx = canvasEl.getContext("2d");
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    resultEl.textContent = "";
  });

  
  let drawing = false;
  canvasEl.addEventListener("mousedown", (e) => {
    drawing = true;
    const ctx = canvasEl.getContext("2d");
    ctx.beginPath();
    const { x, y } = getCursorPosition(e);
    ctx.moveTo(x, y);
  });
  canvasEl.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const ctx = canvasEl.getContext("2d");
    const { x, y } = getCursorPosition(e);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  canvasEl.addEventListener("mouseup", () => drawing = false);
  canvasEl.addEventListener("mouseleave", () => drawing = false);

  function getCursorPosition(e) {
    const rect = canvasEl.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }
});
