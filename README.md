## 1. Установка Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

## 2. Активация Poetry
```mermaid
poetry shell
```

## 3. Установка зависимостей
``
poetry install
``

## 4. Настройка переменных окружения
Скопируйте пример файла .env.example в .env и укажите свои значения:
```mermaid
cp .env.example .env
```
## 5. Запуск скриптов

### Демонстрация цепочки бренда - один тестовый запрос
```
python src/brand_chain.py
```
### Чат-бот в стиле бренда
```
python app_lc.py
```
### Автоматическая оценка стиля (сохраняет результат в reports/style_eval.json)
```
python src/style_eval.py
```