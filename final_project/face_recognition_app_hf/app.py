from deepface import DeepFace
import pandas as pd
import numpy as np
import os
from pathlib import Path
import datetime as dt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import gradio as gr

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def get_download_btn(inp_file=None, is_raw_file=True):
    if is_raw_file:
        label = 'Скачать полный результат в формате .csv'
    else:
        label = 'Скачать статистику в формате .csv'
    download_btn = gr.DownloadButton(
        label=label,
        value=inp_file,
        visible=inp_file is not None,
        )
    return download_btn

def print_faces(face_objs, image_path):
    # открыть картинку и создать объект для рисования
    pil_image = Image.open(image_path)
    draw = ImageDraw.Draw(pil_image)

    line_widht = int(max(pil_image.size) * 0.003)
    font_size = int(max(pil_image.size) * 0.015)

    # настройки отрисовки
    color = 'red'
    font_path = 'LiberationMono-Regular.ttf'
    font = ImageFont.truetype(str(font_path), size=font_size)
    big_font = ImageFont.truetype(str(font_path), size=2*font_size)

    # итерация по словарям для каждого лица
    for i, res_dict in enumerate(face_objs):
        # извлечение артибутов
        age = res_dict['age']
        x, y, w, h, left_eye, right_eye = res_dict['region'].values()
        gender = res_dict['dominant_gender']
        race = res_dict['dominant_race']
        emotion = res_dict['dominant_emotion']
        text_age = f'Возраст:{age}'
        text_gender = f'Пол:{gender}'
        text_race = f'Раса:{race}'
        text_emo = f'Эмоция:{emotion}'

        # отрисовка боксов и надписей
        draw.rectangle((x, y, x + w, y + h), outline=color, width=line_widht)
        draw.text(xy=(x + 10, y + 2*font_size), text=str(i), font=big_font, fill=color, anchor="lb")
        draw.text(xy=(x, y - font_size), text=text_gender, font=font, fill=color, anchor="lb")
        draw.text(xy=(x, y), text=text_age, font=font, fill=color, anchor="lb")
        draw.text(xy=(x, y + h + font_size), text=text_race, font=font, fill=color, anchor="lb")
        draw.text(xy=(x, y + h + 2*font_size), text=text_emo, font=font, fill=color, anchor="lb")

    return pil_image

def get_stat(images_path):
    '''
    Функция на вход принимает путь к файлам, а возвращает датафрейм
    с результатом обработки изображений
    '''
    # создаем пустой список для запиcи результатов

    result_lst = []
    result_image = np.nan
    # создаем список картинок
    for image in images_path:
        # получим дату из названия
        datetime = image.split('/')[-1].split('.')[0]
        # получим данные из изображений
        try:
            face_objs = DeepFace.analyze(
                img_path = image,
                actions = ['age', 'gender', 'race', 'emotion'],
                detector_backend = 'retinaface',
                silent = True
                )
            if pd.isna(result_image):
                result_image = print_faces(face_objs, image)

        except ValueError:
            face_objs = [{'region':{'x': 0,
                                         'y': 0,
                                         'w': 0,
                                         'h': 0,
                                         'left_eye': 0,
                                         'right_eye': 0},
                          'age': np.nan,
                          'dominant_gender': np.nan,
                          'dominant_race': np.nan,
                          'dominant_emotion': np.nan}]
        res_face_objs = []
        needed_keys = ['region', 'age', 'dominant_gender', 'dominant_race', 'dominant_emotion']
        for res_dict in face_objs:
            new_dict = dict((k, res_dict[k]) for k in needed_keys if k in res_dict)
            new_dict['img_name'] = image.split('/')[-1]
            new_dict['img_path'] = image
            new_dict['datetime'] = datetime
            res_face_objs.append(new_dict)
            del new_dict
        del face_objs
        # добавим результаты в список
        result_lst.extend(res_face_objs)
        del res_face_objs
    df = pd.DataFrame(result_lst)

    df = df.reset_index()
    df = df.rename(columns={
        'dominant_gender': 'gender',
        'dominant_race': 'race',
        'dominant_emotion': 'emotion',
        'index': 'id'
    })

    answer = f'''
    Проанализировано изображений: {len(images_path)}.
    Найдено людей: {len(df.dropna(subset='age'))}.
    '''

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['date'] = df['datetime'].dt.round('h')
    df['age'] = df['age'].astype('Int32')

    df.to_csv('raw_result.csv', index=False)
    df[['id', 'datetime', 'date', 'age', 'gender', 'race', 'emotion']].to_csv('clean_result.csv', index=False)



#=========Графики=======
    data1 = df.groupby('date')['id'].count().reset_index()
    data2 = df.groupby('gender')['id'].count().reset_index()
    data4 = df.groupby('emotion')['id'].count().reset_index()
    data5 = df.groupby('race')['id'].count().reset_index()
    fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"colspan": 2}, None],
           [{}, {}],
           [{}, {}]],
    subplot_titles=('Количество людей',
                    'Гистограмма возраста',
                    'Пол',
                    'Эмоции',
                    'Расы'),
    shared_xaxes=False,
    vertical_spacing=0.1)
    # Количество людей
    fig.add_trace(go.Scatter(x=data1['date'], y=data1['id'],
                        mode='lines+markers',
                        name='Количество людей',
                        marker_color = 'indianred'), row=1, col=1)
    fig.update_xaxes(title_text = "Дата", row=1, col=1)
    fig.update_yaxes(title_text = "Количество", row=1, col=1)
    # Гистограмма возраста
    fig.add_trace(go.Histogram(x=df.loc[df['gender'] == 'Man', 'age'],
                                name='Мужчины',
                                marker_color='lightsalmon'),row=2, col=1)
    fig.add_trace(go.Histogram(x=df.loc[df['gender'] == 'Woman', 'age'],
                                name='Женщины',
                                marker_color='indianred'), row=2, col=1)
    fig.update_xaxes(title_text = "Возраст", row=2, col=1)
    fig.update_yaxes(title_text = "Количество", row=2, col=1)
    # Пол
    fig.add_trace(go.Bar(x=data2['gender'],
                        y=data2['id'],
                        text=data2['id'],
                        textposition='auto',
                        name='Пол',
                        marker_color='lightsalmon'), row=2, col=2)
    fig.update_xaxes(title_text = "Пол", row=2, col=2)
    fig.update_yaxes(title_text = "Количество", row=2, col=2)
    # Эмоции
    fig.add_trace(go.Bar(x=data4['emotion'],
                        y=data4['id'],
                        text=data4['id'],
                        textposition='auto',
                        name='Эмоция',
                        marker_color='lightsalmon'), row=3, col=1)
    fig.update_xaxes(title_text = "Эмоции", row=3, col=1)
    fig.update_yaxes(title_text = "Количество", row=3, col=1)
    # Расы
    fig.add_trace(go.Bar(x=data5['race'],
                        y=data5['id'],
                        text=data5['id'],
                        textposition='auto',
                        name='Раса',
                        marker_color='lightsalmon'), row=3, col=2)

    fig.update_xaxes(title_text = "Расы", row=3, col=2)
    fig.update_yaxes(title_text = "Количество", row=3, col=2)

    fig.update_layout(
        showlegend=False,
        title_text='Графики атрибутов',
        barmode='stack',
        autosize=False,
        width=1000,
        height=1200
        )

    return df, answer, result_image, fig

with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    # состояние с путем до файла
    raw_result_path = gr.State('raw_result.csv')
    clean_result_path = gr.State('clean_result.csv')
    is_raw_file = gr.State(False)
    gr.Markdown(
    """
    # Определение количества людей на изображениях, их пола, возраста, расы и эмоций
    Введите путь до ваших изображений и получите результат.
    """
    )
    with gr.Tab('Обзор'):
        inp = gr.Files(file_count='directory')
        btn = gr.Button("Получить результат")
        res_text = gr.Textbox(label="Результаты")
        with gr.Row():
            with gr.Column():
                res_data = gr.Dataframe()
                raw_download_btn = get_download_btn(inp_file=None)
                clean_download_btn = get_download_btn(inp_file=None)
            res_img = gr.Image(label='Пример изображения')

    with gr.Tab('Графики атрибутов'):
        plot = gr.Plot()

    out = [res_data, res_text, res_img, plot]
    clean_dbtn_inp = [clean_result_path, is_raw_file]
    btn.click(
        fn=get_stat,
        inputs=inp,
        outputs=out,
    ).success(
        fn=get_download_btn,
        inputs=[raw_result_path],
        outputs=raw_download_btn
    ).success(
        fn=get_download_btn,
        inputs=clean_dbtn_inp,
        outputs=clean_download_btn
    )

    raw_download_btn.click(
        lambda path: None,
        inputs=[raw_result_path],
        outputs=None
    )
    clean_download_btn.click(
    lambda path: None,
    inputs=[clean_result_path],
    outputs=None
    )
demo.launch()
