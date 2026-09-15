# Лабораторна робота №1

**Інтелектуальний аналіз даних та моделювання кризових явищ**  
Тема: Web Scraping, Data Collection, Visualization.

Основний результат: [виконаний ноутбук з поясненнями та графіками](assignement-1.ipynb).
Умови взято з [оригінального репозиторію](https://github.com/maxvonlancaster/ida-labs).
ПІБ та групу не надано. Ноутбуки №2–4 у джерелі містять лише заголовки.

## Що виконано

1. Отримання тексту двох реальних вебсторінок без HTML-тегів.
2. Запит до відкритого API GitHub і збереження [result.json](result.json).
3. Пошук погоди у JSON за датою.
4. Пошук погоди у CSV за датою з підтримкою `YYYY-M-D` та `YYYY-MM-DD`.
5. Шість PNG-файлів у [figures](figures): погода, повнота спостережень, порівняння методів заповнення й масштабування.
6. Реальний датасет Kaggle: mean, median, mode, KNN; MinMax, Z-score, Robust; перевірка формули Robust через NumPy.

## Дані та результати

- `resources/weather.json`: чотири дні погоди Токіо у серпні 2024 року.
- `resources/weather.csv`: 6 812 спостережень за 1997–2015 роки. Місто у файлі не зазначено.
- [Wine_Quality.csv](data/kaggle/Wine_Quality.csv): 6 497 рядків, 13 колонок, 38 реальних пропусків. Завантажено з рекомендованого [Kaggle-набору](https://www.kaggle.com/datasets/ilayaraja07/data-cleaning-feature-imputation). У початковому прикладі студентів пропуски лише категоріальні, тому обрано числові дані вина з того самого архіву.
- [Походження і контрольна сума CSV](data/provenance.json).
- [Підсумок експерименту](outputs/summary.json), [похибки по ознаках](outputs/imputation_validation.csv), оброблені CSV у `outputs/`.
- [Джерела вебзапитів і час отримання](outputs/collection_sources.json).

У контрольному експерименті середня стандартизована MAE: KNN — 0.2760, median — 0.7046, mean — 0.7300, mode — 0.7438. Для підсумкового заповнення обрано KNN. У фінальному наборі пропусків немає. Оригінальні відомі значення збережено.

80% рядків використано для навчання, 20% — для валідації (`random_state=42`); у перевірочній копії приховано приблизно 10% відомих числових значень. Масштабування та заповнювачі для оцінювання навчаються лише на train. Це один експеримент, а не гарантія точності на справжніх пропусках. Випадковий поділ рядків може розмістити повторювані профілі в обох частинах; для оцінки узагальнення на нові профілі потрібен поділ за групами.

Основний скейлер — Robust через асиметрію та крайні значення. Порівняння з MinMax та Z-score є в ноутбуці. Ціль `quality` та категорія `type` не масштабуються і не беруть участі у відстанях KNN. Для подальшої прогнозної моделі перетворення потрібно навчати заново лише на її train-вибірці.

## Запуск

Перевірено з Python 3.12. Залежності зафіксовано у `requirements.txt`.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe lab1.py --collect
.\.venv\Scripts\python.exe execute_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python lab1.py --collect
.venv/bin/python execute_notebook.py
.venv/bin/python -m unittest discover -s tests -v
```

`lab1.py` без `--collect` виконує аналіз локальних даних без вебзапитів, якщо CSV Kaggle на місці. `--collect` та виконання всього ноутбука потребують інтернету. Збережений ноутбук можна переглядати без запуску. Для інтерактивного редагування використайте VS Code з Jupyter або встановіть JupyterLab окремо.

`parse_json` і `parse_csv` повертають список словників. Якщо дати немає — порожній список; неправильна календарна дата спричиняє `ValueError`. Мережеві та HTTP-помилки не приховуються. Завантажувач Kaggle використовує публічний endpoint та локальний кеш замість `kagglehub`.

## Перевірки

- Виконано всі 12 комірок коду ноутбука без помилок.
- Пройдено 9 автоматичних тестів: дати, пропуски CSV, очищення HTML, HTTP-помилки, запис JSON, збереження відомих значень, масштаби та незалежна перевірка MAE.
- Перевірено вигляд шести PNG-графіків.

## Навчальні матеріали

- [Візуалізація](https://github.com/maxvonlancaster/intelligent-data-analysis/blob/main/src/103-data-visualization.ipynb).
- [Лінійна алгебра та NumPy](https://github.com/maxvonlancaster/intelligent-data-analysis/blob/main/src/104-linear-algebra.ipynb).

Окремого завдання з прогнозування криз у наданій лабораторній немає.
