import numpy as np
from scipy.signal.windows import get_window
import plotly.express as px


def make_def_dom(x_start: float, x_stop:float, x_radius: int, y_start: float, y_stop:float, y_radius: int,) -> list:
    """
    Функция формирования матриц области определения
    :param x_start: начало области определения по оси абсцис
    :param x_stop: конец области определения (включительно) по оси абсцис
    :param x_radius: радиус области определения по строкам
    :param y_start: начало области определения по оси ординвт
    :param y_stop: конец области определения (включительно) по оси ординвт
    :param y_radius: радиус области определения по столбцам
    :return : список матриц [x, y, def_dom], 
                х - матрица аргументов по оси абсцис, размером 4 * rows_radius * columns_radius
                y - матрица аргументов по оси ординат, размером 4 * rows_radius * columns_radius
    """
    x = np.linspace(x_start, x_stop, 2 * x_radius)
    x = np.outer(np.ones(2 * y_radius), x)
    y = np.linspace(y_start, y_stop, 2 * y_radius)
    y = np.outer(y, np.ones(2 * x_radius))
    return [x, y]


def func(x: np.ndarray, y:np.ndarray) -> np.ndarray:
    result = 5 * x + 1 * y
    return result



if __name__ == "__main__":
    x = dict()
    y = dict()

    x["start"] = 0
    x["stop"] = 10
    x["radius"] = 64

    y["start"] = 4
    y["stop"] = 15
    y["radius"] = 64

    X, Y = make_def_dom(x["start"], x["stop"], x["radius"], y["start"], y["stop"], y["radius"])
    Z = func(X, Y)
    fft = np.fft.fft2(Z)
    fft = np.roll(fft, y["radius"] - 1, axis=0)
    fft = np.roll(fft, x["radius"] - 1, axis=1)
    # Поверхность функции

    # Тепловая карта функции

    # Поверхность АЧХ

    # Тепловая карта АЧХ

    # Поверхность ФЧХ

    # Тепловая карта ФЧХ