'''
Приложение для запуска на Hugging Face Spaces
'''
import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from geopy.geocoders import Nominatim, Photon
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

# Загрузка модели из репозитория HF
model_path = hf_hub_download(repo_id="Nikgorby/diplom_DS_SF", filename="random_forest_model (1).pkl")
model = joblib.load(model_path)

def predict(features_lst: list, model):
    """Получение предсказаний от модели

    Args:
        features_lst (list): список фичей
        model (model): обученная модель

    Returns:
        list: предсказание
    """
    # Предсказание
    predictions = model.predict(features_lst)

    return predictions
# end def

def addr_to_coords(addr: str):
    """Функция получения координат по адресу

    Args:
        addr (str): Строка с адресом

    Returns:
        float: координаты широта, долгота
    """
    geolocator = Photon(user_agent="my_geocoder")

    # Геокодирование адреса
    try:
        # попытка получения координат
        location = geolocator.geocode(addr)
        if (location):
        # координаты получены
            return location.latitude, location.longitude
        else:
            # координаты не получены
            return 0, 0
    # end if
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Error: {e}. Retrying...")
        time.sleep(2)  # Добавляем задержку перед повторной попыткой
        return addr_to_coords(addr)
    # end try

def predict_out(baths, square, beds, address, pool, property_type, state, year_built, remodeled_year, avg_school_rating, schools_qty, avg_school_dist):
    """Функция получения предсказания от модели

    Args:
        args: фичи

    Returns:
        float: предсказание модели
    """

    # Формирование признака объекта
    lst = [0]*8
    lst[property_type] = 1
    property_type = ", ".join(map(str, lst))

    # Преобразование адреса в координаты
    coords = addr_to_coords(address)
    coords = ", ".join(map(str, coords))

    # Формирование списка фичей
    features_lst = f'{baths}, {square}, {beds}, {coords}, {pool}, {property_type}, {state}, {year_built}, {remodeled_year}, {avg_school_rating}, {schools_qty}, {avg_school_dist}'
    features_lst = features_lst.split(", ")
    features_lst = [float(num) for num in features_lst]
    features_lst = [features_lst]

    # Предсказание
    predictions = predict(features_lst, model)
    
    # Возвращение предсказаний в виде DataFrame
    result = int(predictions[0])
    return features_lst, result

# Создание интерфейса Gradio
title = "Интерактивное демо модели"
description = "Введите фичи."

iface = gr.Interface(
    fn=predict_out,
    inputs=[gr.Textbox(label="Кол-во ванных"),
            gr.Textbox(label="Площадь, кв. фут."),
            gr.Textbox(label="Кол-во спален"),
            gr.Textbox(label="Адрес в виде 32413 Crystal Breeze Ln, Leesburg"),
            gr.Radio(
                ["Нет", "Да"], type="index",
                label = "Бассейн"
            ),
            gr.Dropdown(
                ["Кондоминиум", "Зем. участок", "На неск. семей", "Другое", "Ранчо", "На одну семью", "Таунхаус", "Традиционное"],
                label='Тип объекта',
                type="index"
            ),
            gr.Textbox(label='Штат (номер)'),
            gr.Textbox(label="Год постройки"),
            gr.Textbox(label="Год капитального ремонта (0 если не проводился)"),
            gr.Textbox(label="Средний рейтинг школ рядом (от 0 до 1)"),
            gr.Textbox(label="Количество школ рядом"),
            gr.Textbox(label="Среднее расстояние до школы, миль"),
            ],
    outputs=[gr.Textbox(label="Features"),
             gr.Textbox(label="Предсказание, $")],
    title=title,
    description=description,
    allow_flagging='never'
)

# Запуск приложения
iface.launch()