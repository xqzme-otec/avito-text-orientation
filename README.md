# Ориентация текстового кропа 0° / 180° (Авито)

**1 − Brier на платформе: 0.9913.**

Решение: компактная антисимметричная CNN (0.47 млн параметров), обученная на синтетических русских кропах и реальных кропах TextOCR, плюс «языковой» сигнал от открытых распознавателей текста PaddleOCR PP-OCRv5 (eslav, cyrillic). Каждый распознаватель читает кроп как есть и повёрнутым на 180°, сигналы объединяет антисимметричная логистическая регрессия.

Подробное описание подхода, валидации и результатов — в **[`solution.ipynb`](solution.ipynb)**.

## Воспроизведение submission.csv

```bash
pip install -r requirements.txt
# выданные данные: test/test/images/*.png и test/sample_submission.csv
python scripts/download_data.py --minimal   # 2 распознавателя PaddleOCR, 16 МБ
python scripts/make_submission.py           # -> submission.csv (~8 мин на CPU)
```

Либо запустить `solution.ipynb` целиком: он делает то же самое и в конце сверяет результат с отправленным файлом `submissions/submission_2_final.csv`.

Для точного воспроизведения в репозитории лежат обученные веса CNN (`models/w24_v1/best.pt`) и коэффициенты стекера (`models/stacker.json`).

## Полное переобучение

```bash
python scripts/download_data.py              # TextOCR, ru-Wikipedia, Google Fonts (~3.5 ГБ)
python scripts/train.py --name w24_v1 --width 24 --steps 16000 --p_synth 0.5
python scripts/fit_stacker.py                # валидация стека и models/stacker.json
python scripts/make_submission.py
```

Все случайные процессы зафиксированы (seed=42, каждый обучающий пример генерируется детерминированно из своего индекса). При обучении на другом железе (GPU/MPS/CPU) веса будут численно немного отличаться.

## Open-source
PaddleOCR PP-OCRv5 recognizers (Apache-2.0, ONNX из RapidOCR), TextOCR (CC-BY 4.0), Wikipedia (CC-BY-SA), Google Fonts (OFL), PyTorch, ONNX Runtime, OpenCV, Pillow, scikit-learn.
