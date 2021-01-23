import sys
import numpy as np
from scipy.signal.windows import get_window
from itertools import product
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cmd_type_to_hub(hub, key, spin, type_, condition):
    """
     Функция занесения значений,
    если они приводимы к типу type и для них condition истинно,
    в массив/словарь hub с индексом/по ключу key
    (для использования в поле command элементы spinbox)
    """
    def vldt():
        entry = spin.get()
        if not entry:
            return False
        try:
            buf = type_(entry)
            if condition(buf):
                hub[key] = buf
            else: 
                spin.set(hub[key])
                spin.configure(validate="key")    
        except ValueError:
            spin.set(hub[key])
            spin.configure(validate="key")
            return False 
        return True
    return vldt


def validate_type_to_hub(hub, key, spin, type_, condition):
    """
         Функция занесения значений,
    если они приводимы к типу type и для них condition истинно,
    в массив/словарь hub с индексом/по ключу key
    (для использования в поле validatecommand элементы spinbox)
    """
    def vldt(entry: str):
        try:
            buf = type_(entry)
            if condition(buf):
                hub[key] = buf
            else: 
                spin.set(hub[key])
                spin.configure(validate="key")    
        except ValueError:
            spin.set(hub[key])
            spin.configure(validate="key")
            return False 
        return True
    return vldt


def find_key(dictionary: dict, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key


def first_phase_der(source: np.ndarray) -> np.ndarray:
    left = -np.diff(source[:source.size // 2])
    right = np.diff(source[source.size // 2 - 1:])
    derivative = np.block([left, 0, right])
    derivative = np.where(derivative < -np.pi, derivative + 2 * np.pi, derivative)
    derivative = np.where(derivative > np.pi, derivative - 2 * np.pi, derivative)
    return derivative


def fft_of_real_from_fft_of_complex(source_fft: np.ndarray) -> np.ndarray:
    """
    Функция выделения Фурье-образа действительной части сигнала из Фурье-образа комплексного сигнла
    :param fft_of_complex: ДПФ сигнала (НЕ ЦЕНТРИРОВАННОЕ)
    :return : ДПФ мнимой части сигнала  (НЕ ЦЕНТРИРОВАННОЕ)
    """
    fft = np.zeros(source_fft.size, dtype=complex)
    fft[0] = np.real(source_fft[0])
    fft[1:] = 0.5 * (np.real(source_fft[1:] + source_fft[1:][::-1]) + np.imag(source_fft[1:] - source_fft[1:][::-1]) * 1j)
    return fft


def fft_of_imag_from_fft_of_complex(source_fft: np.ndarray) -> np.ndarray:
    """
    Функция выделения Фурье-образа мнимой части сигнала из Фурье-образа комплексного сигнла
    :param fft_of_complex: ДПФ сигнала (НЕ ЦЕНТРИРОВАННОЕ)
    :return : ДПФ мнимой части сигнала  (НЕ ЦЕНТРИРОВАННОЕ)
    """
    fft = np.zeros(source_fft.size, dtype=complex)
    fft[0] = np.real(source_fft[0])
    fft[1:] = 0.5 * (np.imag(source_fft[1:] + source_fft[1:][::-1]) + np.real(source_fft[1:][::-1] - source_fft[1:]) * 1j)
    return fft


def fft_of_complex_from_real_and_imag_fft(fft_real: np.ndarray, fft_imag: np.ndarray) -> np.ndarray:
    """
    Функция выделения Фурье-образа комплексного сигнала из Фурье-образов действительной и мнимой частей сигнала
    :param fft_real: ДПФ действтельной части сигнала (НЕ ЦЕНТРИРОВАННОЕ)
    :param fft_imag: ДПФ мнимой части сигнала (НЕ ЦЕНТРИРОВАННОЕ)
    :return : ДПФ комплекснного сигнала  (НЕ ЦЕНТРИРОВАННОЕ)
    """
    fft = np.zeros(fft_real.size, dtype=complex)
    fft[0] = np.real(fft_real[0]) + np.real(fft_imag[0]) * 1j
    fft[1:] = (np.real(fft_real[1:]) - np.imag(fft_imag[1:])) + (np.imag(fft_real[1:]) + np.real(fft_imag[1:])) * 1j
    return fft


class Stend:

    def __init__(self):
        self.type_real = float
        self.type_complex = complex
        self.plots_shape_min = 1
        self.plots_shape_max = 3

        self. signal_func_type_dict = {
                            "действительный": self.type_real,
                            "комплексный": self.type_complex
                        }

        self.signal_func_dict = {
                        "kX + b": lambda x, args: args["k"] * x + args["b"],
                        "s * sin(kX + b) + d": lambda x, args: args["s"] * np.sin(args["k"] * x + args["b"]) + args["d"],
                        "s * ln(kX + b) + d": lambda x, args: args["s"] * np.log(args["k"] * x + args["b"]) + args["d"]
                    }

        self.signal_func_param_dict = {
                                    self.signal_func_dict["kX + b"]: ["k", "b"],
                                    self.signal_func_dict["s * sin(kX + b) + d"]: ["s", "k", "b", "d"],
                                    self.signal_func_dict["s * ln(kX + b) + d"]: ["s", "k", "b", "d"]
                                }
        
        self.signal_param_check_dict = {
                                    "s": {"type": float, "condition": lambda x: True},
                                    "k": {"type": float, "condition": lambda x: True},
                                    "b": {"type": float, "condition": lambda x: True},
                                    "d": {"type": float, "condition": lambda x: True}
                                }

        self.noise_exist_dict = {
                                "отсутствует": False,
                                "присутствует": True
                            }

        self.noise_type_dict = {
                            "аддитивный": lambda signal, noise: signal + noise,
                            "мультипликативный": lambda signal, noise: signal * noise
                        } 

        self.noise_func_dict = {
                            "s * sin(kX + b) + d": lambda x, args: args["s"] * np.sin(args["k"] * x + args["b"]) + args["d"],
                            "Гауссов N(m, std)": lambda x, args: args["m"] + args["std"] * np.random.standard_normal(size=x.shape),
                            "равномерный U[a; b)":lambda x, args: np.abs(args["b"] - args["a"]) * np.random.random_sample(x.shape) + np.min([args["a"], args["b"]])
                        }
        
        self.noise_func_param_dict = {
                                self.noise_func_dict["s * sin(kX + b) + d"]: ["s", "k", "b", "d"],
                                self.noise_func_dict["Гауссов N(m, std)"]: ["m", "std"],
                                self.noise_func_dict["равномерный U[a; b)"]: ["a", "b"]
                                }

        self.noise_param_check_dict = {
                                    "s": {"type": float, "condition": lambda x: True},
                                    "k": {"type": float, "condition": lambda x: True},
                                    "b": {"type": float, "condition": lambda x: True},
                                    "d": {"type": float, "condition": lambda x: True},
                                    "m": {"type": float, "condition": lambda x: True},
                                    "std": {"type": float, "condition": lambda x: x >=0},
                                    "a": {"type": float, "condition": lambda x: True}
                                }

        self.filter_exist_dict = {
                                "не выполняется": False,
                                "выполняется": True
                            }

        self.filter_type_dict = {
                                    "ФНЧ": self.make_lowpass_filter,
                                    "ФВЧ": self.make_highpass_filter
                                    #"Режекторный": self.make_notch_filter,
                                    #"Полосовой": self.make_bandpass_filter,
                                    #"Узкополосный": self.make_narrow_filter
                                }

        self.filter_func_label_dict = {
                                "Хэмминга": "hamming",
                                "прямоугольное": "boxcar",
                                "треугольное": "triang",
                                "Блэкмана": "blackman",
                                "Ханна": "hann",
                                "Бартлетта": "bartlett",
                                "Flat top": "flattop",
                                "Парцена": "parzen",
                                "Бохмана": "bohman",
                                "Блэемана-Харриса": "blackmanharris",
                                "Nuttal": "nuttall",
                                "Бартлетта-Ханна": "barthann",
                                "Кайзера": "kaiser",
                                "Гаусса": "gaussian",
                                "Обобщенная гауссаина": "general_gaussian",
                                "Туки": "tukey"
                            }

        self.filter_func_param_dict = {
                                "hamming": ["width"],
                                "boxcar": ["width"],
                                "triang": ["width"],
                                "blackman": ["width"],
                                "hann": ["width"],
                                "bartlett": ["width"],
                                "flattop": ["width"],
                                "parzen": ["width"],
                                "bohman": ["width"],
                                "blackmanharris": ["width"],
                                "nuttall": ["width"],
                                "barthann": ["width"],
                                "kaiser": ["width", "beta"],
                                "gaussian": ["std"],
                                "general_gaussian": ["p", "std"],
                                "tukey": ["alpha"]
                            }   

        self.filter_param_check_dict = {
                                    "width": {"type": int, "condition": lambda x: (x > 0) and (x % 2 == 1) and (x <= self.signal_size_buf[0])},
                                    "std": {"type": float, "condition": lambda x: x >=0},
                                    "beta": {"type": int, "condition": lambda x: x >0},
                                    "p": {"type": float, "condition": lambda x: x >=0},
                                    "alpha": {"type": float, "condition": lambda x: (x >= 0) and (x <= 1)}
                                }

        self.plots_types_labels = [
                        "действительная часть сигнала",
                        "действительная часть сигнала без наложения шума",
                        "действительная часть сигнала после фильтрации",

                        "мнимая часть сигнала",
                        "мнимая часть сигнала без наложения шума",
                        "мнимая часть сигнала после фильтрации",

                        "амплитуда сигнала",
                        "амплитуда сигнала без наложения шума",
                        "амплитуда сигнала после фильтрации",

                        "фаза сигнала",
                        "фаза сигнала без наложения шума",
                        "фаза сигнала после фильтрации",

                        "АЧХ действительной части сигнала",
                        "АЧХ действительной части сигнала без наложения шума",
                        "АЧХ действительной части сигнала после фильтрации",
                        "окно фильтрации АЧХ действительной части сигнала",

                        "АЧХ мнимой части сигнала",
                        "АЧХ мнимой части сигнала без наложения шума",
                        "АЧХ мнимой части сигнала после фильтрации",
                        "окно фильтрации АЧХ мнимой части сигнала",
                        
                        "вторая производная \n АЧХ действительной части сигнала",
                        "вторая производная \n АЧХ действительной части сигнала без наложения шума",
                        "вторая производная \n АЧХ мнимой части сигнала",
                        "вторая производная \n АЧХ мнимой части сигнала без наложения шума",


                        "ФЧХ действительной части \nсигнала",
                        "ФЧХ действительной части \nсигнала без наложения шума",
                        "ФЧХ действительной части \nсигнала после фильтрации",
                        "окно фильтрации ФЧХ действительной части сигнала",

                        "ФЧХ мнимой части сигнала",
                        "ФЧХ мнимой части сигнала без наложения шума",
                        "ФЧХ мнимой части сигнала после фильтрации",
                        "окно фильтрации ФЧХ мнимой части сигнала",

                        "первая производная \n(односторонняя приведённая) \n ФЧХ действительной части сигнала",
                        "первая производная \n(односторонняя приведённая) \n ФЧХ действительной части сигнала без наложения шума",
                        "первая производная \n(односторонняя приведённая) \n ФЧХ действительной части сигнала после фильтрации",
                        "первая производная \n(односторонняя приведённая) \n ФЧХ мнимой части сигнала",
                        "первая производная \n(односторонняя приведённая) \n ФЧХ мнимой части сигнала без наложения шума",
                        "первая производная \n(односторонняя приведённая) \n ФЧХ мнимой части сигнала после фильтрации"
                        ]

        self.plots_type_dict = dict()

        self.phase_filter_method_dict = {
                                        "правой производной": self.filtering_phase
                                        #"центральной производной": self.filter_phase_first_der_central
                                        }
                        
        self.plots_scale_dict = {
                                    "линейный": lambda x: x,
                                    "логарифмический": np.log10
                                }

        # число отсчётов сигнала
        self.signal_size = [100]

        # буферная переменная числа отсчётов
        self.signal_size_buf = [self.signal_size[0]]

        # тип области значений сигнала
        self.signal_func_type = list(self.signal_func_type_dict.values())[0]

        # массив функций сигнала 
        #   в случае действительного сигнала - 1 элемент
        #   в случае комплексного сигнала - 2 элемента (для амплитудной и фазовой части)
        self.signal_func = [list(self.signal_func_dict.values())[0], list(self.signal_func_dict.values())[0]]

        # границы области определения сигнала
        self.def_dom_borders = [0, 2 * 3.14]

        # массив области определения
        self.def_dom = np.linspace(self.def_dom_borders[0], self.def_dom_borders[1], self.signal_size[0])

        # список словарей дополнительных аргументов для функций сигнала
        #   в случае действительного сигнала - 1 объект
        #   в случае комплексного сигнала - 2 объекта (для амплитудной и фазовой части)
        self.signal_args = [{"k": 0, "b": 1, "s": 1, "d": 0}, {"k": 0, "b": 1, "s": 1, "d": 0}]

        # Словарь массивов значений различных видов сигнала
        self.signal = dict()

        # Словарь массивов значений различных видов Фурье-образов
        self.fft = dict()

        for label in ["source", "clear"]:
            # массив значений сигнала/ сигнала без наложения шума
            self.signal[label] = self.signal_func[0](self.def_dom, self.signal_args[0]).astype(self.type_real)
            # словарь образов действительной и мнимой частей сигнала/ сигнала без наложения шума
            self.fft[label] = dict()
            # Фурье-образо действительной части сигнала/ сигнала без наложения шума
            self.fft[label]["real"] = np.zeros(self.signal_size[0], dtype=complex)
            self.fft[label]["real"] = fft_of_real_from_fft_of_complex(np.fft.fft(np.real(self.signal[label])))
            self.fft[label]["real"] = np.roll(self.fft[label]["real"], self.signal_size[0] // 2 - 1)
            # Фурье-образо мнимой части сигнала/ сигнала без наложения шума
            self.fft[label]["imag"] = np.zeros(self.signal_size[0], dtype=complex)
            self.fft[label]["imag"] = fft_of_imag_from_fft_of_complex(np.fft.fft(np.imag(self.signal[label])))
            self.fft[label]["imag"] = np.roll(self.fft[label]["imag"], self.signal_size[0] // 2 - 1)

        
        # Фурье-образ сигнала после фильтрации 
        self.fft["filtered"] = dict()
        self.fft["filtered"]["real"] = self.fft["source"]["real"]
        self.fft["filtered"]["imag"] = self.fft["source"]["imag"]

        # массив сигнала после фильтрации
        self.signal["filtered"] = np.real(np.fft.ifft(fft_of_complex_from_real_and_imag_fft(np.roll(self.fft["filtered"]["real"], - self.signal_size[0] // 2 + 1),
                                                                                    np.roll(self.fft["filtered"]["imag"], - self.signal_size[0] // 2 + 1))))

        # массив флагов наличия шума
        #   в случае действительного СИГНАЛА - 1 элемент
        #   в случае комплексного СИГНАЛА - 2 элемента (для амплитудной и фазовой части)
        self.noise_exist = [list(self.noise_exist_dict.values())[0], list(self.noise_exist_dict.values())[0]]

        # массив типов шума
        #   в случае действительного СИГНАЛА - 1 элемент
        #   в случае комплексного СИГНАЛА - 2 элемента (для амплитудной и фазовой части)
        self.noise_type = [list(self.noise_type_dict.values())[0], list(self.noise_type_dict.values())[0]]

        # массив функций шума 
        #   в случае действительного СИГНАЛА - 1 элемент
        #   в случае комплексного СИГНАЛА - 2 элемента (для амплитудной и фазовой части)
        self.noise_func = [list(self.noise_func_dict.values())[0], list(self.noise_func_dict.values())[0]]

        # маассив словарей аргументов для функций шума
        self.noise_args = [{"m": 0, "std": 1, "a": 0, "b": 1, "d": 0, "s": 1, "k": 1}, 
                           {"m": 0, "std": 1, "a": 0, "b": 1, "d": 0, "s": 1, "k": 1}]

        # словарь массивов флагов наличия фильтрации АЧХ и ФЧХ
        self.filter_exist = dict()
        for part in ["real", "imag"]:
            self.filter_exist[part] = [list(self.filter_exist_dict.values())[0], list(self.filter_exist_dict.values())[0]]

        # словарь массивов типов фильтров АЧХ и ФЧХ
        self.filter_type =dict()
        for part in ["real", "imag"]:
            self.filter_type[part] = [list(self.filter_type_dict.values())[0], list(self.filter_type_dict.values())[0]]

        # словарь названий функции окна фильтрации
        self.filter_func_label = dict()
        for part in ["real", "imag"]:
            #   первое - окна амплитудной части
            #   второе - окна фазовой части
            self.filter_func_label[part] = [list(self.filter_func_label_dict.values())[0], list(self.filter_func_label_dict.values())[0]]

        # словарь маассивов словарей аргументов для функций шума
        self.filter_args = dict()
        for part in ["real", "imag"]:
            self.filter_args[part] = [{"width": 3, "std": 1, "beta": 0, "p": 1, "alpha": 0.5}, 
                                      {"width": 3, "std": 1, "beta": 0, "p": 1, "alpha": 0.5}]

        # словарь окон фильтрации амплитудной и фазовой составляюей спектра действительной и мнимой частей сигнала
        self.filter = dict()
        for part in ["real", "imag"]: 
            #   первая строка - окно для амплитудной части
            #   вторая строка - окно для фазовой части 
            self.filter[part] = np.zeros((2, self.signal_size[0])) 

        # словарь функций методов фильтрации ФЧХ
        self.phase_filter_method = dict()
        for part in ["real", "imag"]: 
            self.phase_filter_method[part] = list(self.phase_filter_method_dict.values())[0]
        
        # Заполним словарь функций отрисовки графиков
        for label in self.plots_types_labels:
            # Найдем массивы оси абсцисс и ординат
            if "налож" in label:
                source_label = "clear"
            elif "фильтр" in label:
                source_label = "filtered"
            else:
                source_label = "source"

            if "ЧХ" in label:
                x = np.arange(self.signal_size[0])
                if "мним" in label:
                    part = "imag"
                else:
                    part = "real"

                if "окно" in label:
                    y_func = lambda p: p
                    if "А" in label:
                        y_source = self.filter[part][0]
                    else:
                        y_source = self.filter[part][1]
                else:
                    y_source = self.fft[source_label][part]
                    if "А" in label:
                        y_func = np.abs
                    else:
                        y_func = np.angle
                    
                    if "произв" in label:
                        y_source = y_func(y_source)
                        if "втор" in label:
                            y_func = lambda x: np.block([0, np.diff(x, n=2), 0])
                        else:
                            y_func = first_phase_der
            else:
                x = self.def_dom
                y_source = self.signal[source_label]
                if "ампл" in label:
                    y_func = np.abs
                elif "фаз" in label:
                    y_func = np.angle
                elif "мним" in label:
                    y_func = np.imag
                else:
                    y_func = np.real
            self.plots_type_dict[label] = self.draw_plot_wrapper(x, y_source, y_func, label)

        # Определение параметров для графического интерфейса
        # число строк и столбцов сетки графиков 
        self.subplots_shape = [self.plots_shape_min, self.plots_shape_min]  
        # тип функции, отображаемой на соотвествущем графике
        self.subplots_types = dict()
        for r, c in product(range(self.plots_shape_max), range(self.plots_shape_max)):
            self.subplots_types[(r,c)] = list(self.plots_type_dict.keys())[0]

        # масштаб отображаемого графика
        self.subplots_scale = dict()
        for r, c in product(range(self.plots_shape_max), range(self.plots_shape_max)):
            self.subplots_scale[(r,c)] = list(self.plots_scale_dict.keys())[0]

        # ширина окна программы
        self.window_width = 2569
        # высота окна программы 
        self.window_height = 1440

        # параметры размещения ячеек интерфейса
        self.zero_column_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 0, "columnspan": 2,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "N"
                                    }
        self.first_column_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 2, "columnspan": 10,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "n"
                                    }
        self.second_column_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 12, "columnspan": 2,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "n"
                                    }
        self.signal_cell_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 0, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "N"
                                    }
        self.def_dom_cell_options = {
                                    "row": 1, "rowspan": 1,
                                    "column": 0, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "N"
                                    }
        self.noise_cell_options = {
                                    "row": 2, "rowspan": 1,
                                    "column": 0, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "N"
                                    }
        self.signal_calc_button_options = {
                                    "row": 1, "rowspan": 1,
                                    "column": 0, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "WS"
                                    }
        self.quit_button_options = {
                                    "row": 1, "rowspan": 1,
                                    "column": 1, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "SE"
                                    }
        self.plots_cell_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 0, "columnspan": 2,
                                    "ipadx": 0, "ipady": 0, 
                                    "padx": 0, "pady": 0,
                                    "sticky": "NSWE"
                                    }
        self.filter_cell_options = {
                                    "row": 0, "rowspan": 1,
                                    "column": 0, "columnspan": 1,
                                    "ipadx": 0, "ipady": 0,
                                    "padx": 0, "pady": 0,
                                    "sticky": "NS"
                                    }

        # окно грфического интерфейса
        self.window = tk.Tk()
        self.main_window()
        self.window.mainloop()

    # Вёрстка эленементов графического интерфейса
    def main_window(self):
        # Очистим окно
        for child in self.window.winfo_children():
            child.destroy()
        # Зададим размер и заголовок окна
        self.main_size(self.window_height, self.window_width)
        self.window.title("Окно эксперимента")
        self.window.overrideredirect(1)
        #self.window.attributes("-fullscreen", "True")
        # Зададим элементы окна
        zero_column = tk.Frame(self.window, bd=0)
        zero_column.grid(**self.zero_column_options)
        self.signal_frame = tk.Frame(zero_column, bd=0)
        self.signal_frame.grid(**self.signal_cell_options)
        self.signal_cell()
        self.def_dom_frame = tk.Frame(zero_column, bd=0)
        self.def_dom_frame.grid(**self.def_dom_cell_options)
        self.def_dom_cell()
        self.noise_frame = tk.Frame(zero_column, bd=0)
        self.noise_frame.grid(**self.noise_cell_options)
        self.noise_cell()
        first_column = tk.Frame(self.window, bd=0)
        first_column.grid(**self.first_column_options)
        self.plots_frame = tk.Frame(first_column, bd=0)
        self.plots_frame.grid(**self.plots_cell_options)
        self.plots_cell()
        self.signal_calc_button = tk.Button(first_column, height=2, width=16, text="Рассчитать сигнал", command=self.signal_calc_handler, bg="pale green")
        self.signal_calc_button.grid(**self.signal_calc_button_options)
        second_column = tk.Frame(self.window, bd=0)
        second_column.grid(**self.second_column_options)
        self.filter_frame = tk.Frame(second_column, bd=0)
        self.filter_frame.grid(**self.filter_cell_options)
        self.filter_cell()
        def _quit():
            self.plots_display_frame.quit()
            self.plots_display_frame.destroy()
        tk.Button(first_column, pady=5, height=2, width=8, text="Выход", foreground="snow", bg="red", command=_quit).grid(**self.quit_button_options)

        
    def signal_cell(self):
        """
        Функция отрисовки блока управления параметрами сигнала
        :param master: родительский элемент для данного блока
        :param grid_options: аргументы метода grid для размещения в родительском объекте
        """
        # Локальные вспомогательные функции
        def func_type_choose(master, title: str, ind: int):
            """
            Создание и размещение элементов выбора функции
            :param master: родительский объект интерфейса 
            :param title: заголовок блока
            :param ind: индекс в массиве куда будут считываться целевые параметры
            """
            # Создание объектов
            func_label = tk.Label(master, text=title)
            func_cb = ttk.Combobox(master, values=list(self.signal_func_dict.keys()), state="readonly")
            for i, f in enumerate(self.signal_func_dict.items()):
                if f[1] == self.signal_func[ind]:
                    func_cb.current(i)
            args_dict = {"labels": dict(), "spins": dict()}
            for coef in self.signal_args[ind].keys():
                args_dict["labels"][coef] = tk.Label(master, text=coef+" =")
                args_dict["spins"][coef] = ttk.Spinbox(master, from_=-10**6, to=10**6, width=6)
                args_dict["spins"][coef].set(self.signal_args[ind][coef])
                args_dict["spins"][coef].configure(command=cmd_type_to_hub(self.signal_args[ind], coef,\
                                                                            args_dict["spins"][coef],\
                                                                            self.signal_param_check_dict[coef]["type"],\
                                                                            self.signal_param_check_dict[coef]["condition"]),\
                                                   validate="all",\
                                                   validatecommand=(master.register(validate_type_to_hub(self.signal_args[ind], coef,\
                                                                                    args_dict["spins"][coef],\
                                                                                    self.signal_param_check_dict[coef]["type"],\
                                                                                    self.signal_param_check_dict[coef]["condition"])), "%P"))
                
            # Размещение объектов
            func_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="W")
            func_cb.grid(row=2, column=2, columnspan=6, padx=5, pady=5, sticky="WE")
            for i, p in enumerate(self.signal_func_param_dict[self.signal_func[ind]]):
                args_dict["labels"][p].grid(row=3, column=i*2, pady=5, sticky="E")
                args_dict["spins"][p].grid(row=3, column=i*2+1, pady=5, sticky="W")

            # Локальные обработчики событий
            def func_handler(event):
                """
                Обработчик выбора функции сигнала
                """
                buf_func = self.signal_func_dict[event.widget.get()]
                if not self.signal_func[ind] == buf_func:
                    self.signal_func[ind] = buf_func
                    for widget_type in args_dict.values():
                        for widget in widget_type.values():
                            widget.grid_forget()
                    for i, p in enumerate(self.signal_func_param_dict[self.signal_func[ind]]):
                        args_dict["labels"][p].grid(row=3, column=i*2, pady=5, sticky="E")
                        args_dict["spins"][p].grid(row=3, column=i*2+1, pady=5, sticky="W")
                
            # Привязка к локальным обработчикам событий
            func_cb.bind("<<ComboboxSelected>>", func_handler)

        def real_type_choose(target_frame):
            """
            Создание и размещение элементов выбора функции сигнала в случае действительного сигнала
            """
            func_type_choose(target_frame, "Функция сигнала", 0)

        def complex_type_choose(target_frame):
            """
            Создание и размещение элементов выбора функции сигнала в случае комплексного сигнала
            """
            ampl_frame = tk.Frame(target_frame, bd=0)
            ampl_frame.grid(row=0) 
            phase_frame = tk.Frame(target_frame, bd=0)
            phase_frame.grid(row=1)           
            func_type_choose(ampl_frame, "Функция \nамплитуды сигнала", 0)
            func_type_choose(phase_frame, "Функция \nфазы сигнала", 1)
        # Локальные обработчики событий
        def targing_handler(target_frame):
            def signal_type_handler(event):
                buf_type = self.signal_func_type_dict[event.widget.get()]
                if not self.signal_func_type == buf_type:
                    self.signal_func_type = buf_type
                    for child in target_frame.winfo_children():
                        child.destroy()
                    if self.signal_func_type == self.type_real:
                        real_type_choose(target_frame)
                    else:
                        complex_type_choose(target_frame)
                    self.noise_cell()
                    self.filter_cell()
            return signal_type_handler
        # конец области определения локальных функций
        # Очистим блок
        for child in self.signal_frame.winfo_children():
            child.destroy()
        # Создание объектов блока
        cell_label = tk.Label(self.signal_frame, text="Параметры сигнала", bg="black", foreground="snow")
        signal_type_label = tk.Label(self.signal_frame, text="Тип сигнала: ")
        signal_type_cb = ttk.Combobox(self.signal_frame, values=list(self.signal_func_type_dict.keys()), state="readonly")
        for i, f_t in enumerate(self.signal_func_type_dict.items()):
            if f_t[1] == self.signal_func_type:
                signal_type_cb.current(i)
        func_choose_frame = tk.Frame(self.signal_frame, bd=0)
        # Размещение объектов блока
        cell_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="NWE")
        signal_type_label.grid(row=1, column=0, padx=5, pady=5, sticky="W")
        signal_type_cb.grid(row=1, column=1, padx=5, pady=5, sticky="E")
        func_choose_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="SWE")
        if self.signal_func_type == self.type_real:
            real_type_choose(func_choose_frame)
        else:
            complex_type_choose(func_choose_frame)
        # Привязка локальных обработчиков событий
        signal_type_cb.bind("<<ComboboxSelected>>", targing_handler(func_choose_frame))


    def def_dom_cell(self):
        """
        Функция отрисовки блока управления параметрами аргументов сигнала
        :param master: родительский элемент для данного блока
        :param grid_options: аргументы метода grid для размещения в родительском объекте
        """
        # Очистим блок
        for child in self.def_dom_frame.winfo_children():
            child.destroy()
        # Определение объектов блока
        cell_label = tk.Label(self.def_dom_frame, text=" Параметры области определения сигнала", bg="black", foreground="snow")
        borders_label = tk.Label(self.def_dom_frame, text="Границы отрезка")
        borders_choose_frame = tk.Frame(self.def_dom_frame, bd=0)
        left_bracket = tk.Label(borders_choose_frame, text="[")
        range_start_spin = ttk.Spinbox(borders_choose_frame, from_=0, to=10**6, width=6)
        range_start_spin.set(self.def_dom_borders[0])
        range_start_spin.configure(command=cmd_type_to_hub( self.def_dom_borders, 0,\
                                                            range_start_spin,\
                                                            float, lambda x: x >= 0),\
                                   validate="key",\
                                   validatecommand=(self.def_dom_frame.register(validate_type_to_hub(self.def_dom_borders, 0,\
                                                                                                    range_start_spin,\
                                                                                                    float, lambda x: x >= 0)), "%P"))
        separator = tk.Label(borders_choose_frame, text="; ")
        range_stop_spin = ttk.Spinbox(borders_choose_frame, from_=0, to=10**6, width=8)
        range_stop_spin.set(self.def_dom_borders[1])
        range_stop_spin.configure(command=cmd_type_to_hub(self.def_dom_borders, 1,\
                                                          range_stop_spin,\
                                                          float, lambda x: x >= 0),\
                                  validate="key",\
                                  validatecommand=(self.def_dom_frame.register(validate_type_to_hub(self.def_dom_borders, 1,\
                                                                                                    range_stop_spin,\
                                                                                                    float, lambda x: x >= 0)), "%P"))
        right_bracket = tk.Label(borders_choose_frame, text="]")
        range_size_label = tk.Label(borders_choose_frame, text="Число отсчётов")
        range_size_spin = ttk.Spinbox(borders_choose_frame, from_=0, to=10**6, width=6, increment=2)
        range_size_spin.set(self.signal_size_buf[0])
        range_size_spin.configure(command=cmd_type_to_hub(self.signal_size_buf, 0,\
                                                          range_size_spin,\
                                                          int, lambda x: (x > 0) and (x % 2 == 0)),\
                                  validate="key",\
                                  validatecommand=(self.def_dom_frame.register(validate_type_to_hub(self.signal_size_buf, 0,\
                                                                                                    range_size_spin,\
                                                                                                    int, lambda x: (x > 0) and (x % 2 == 0))), "%P"))
        # Размещение объектов блока
        cell_label.grid(row=0, padx=5, pady=5)
        borders_label.grid(row=1, padx=5, pady=5)
        borders_choose_frame.grid(row=2, padx=5, pady=5)
        left_bracket.grid(row=0, column=0, padx=5, pady=5)
        range_start_spin.grid(row=0, column=1, pady=5, sticky="w")
        separator.grid(row=0, column=2, pady=5)
        range_stop_spin.grid(row=0, column=3, pady=5)
        right_bracket.grid(row=0, column=4, padx=5, pady=5)
        range_size_label.grid(row=1, column=0, columnspan=2, pady=5)
        range_size_spin.grid(row=1, column=2, columnspan=2, pady=5)


    def noise_cell(self):
        """
        Функция отрисовки блока управления параметрами шума
        :param master: родительский элемент для данного блока
        :param grid_options: аргументы метода grid для размещения в родительском объекте
        """
        # Локальные вспомогательные функции
        def noise_choose(master, title:str, ind: int):
            """
            Создание и размещение элементов выбора параметров шума
            :param master: родительский объект интерфейса 
            :param title: заголовок блока
            :param ind: индекс в массиве куда будут считываться целевые параметры
            """
            r = 0  # счётчик строк для размещения объектов
            # Создание объектов
            frame_label = tk.Label(master, text=title, bg="lemon chiffon")
            noise_exist_cb = ttk.Combobox(master, values=list(self.noise_exist_dict.keys()), state="readonly")
            for i, f in enumerate(self.noise_exist_dict.items()):
                if f[1] == self.noise_exist[ind]:
                    noise_exist_cb.current(i)
            param_frame = tk.Frame(master, bd=0)
            noise_type_label = tk.Label(param_frame, text="Тип шума")
            noise_type_cb = ttk.Combobox(param_frame, values=list(self.noise_type_dict.keys()), state="readonly")
            for i, f in enumerate(self.noise_type_dict.items()):
                if f[1] == self.noise_type[ind]:
                    noise_type_cb.current(i)
            noise_func_label = tk.Label(param_frame, text="Модель шума")
            noise_func_cb = ttk.Combobox(param_frame, values=list(self.noise_func_dict.keys()), state="readonly")
            for i, f in enumerate(self.noise_func_dict.items()):
                if f[1] == self.noise_func[ind]:
                    noise_func_cb.current(i)
            args_dict = {"labels": dict(), "spins": dict()}
            for coef in self.noise_args[ind].keys():
                args_dict["labels"][coef] = tk.Label(param_frame, text=coef+" =")
                args_dict["spins"][coef] = ttk.Spinbox(param_frame, from_=-10**6, to=10**6, width=6)
                args_dict["spins"][coef].set(self.noise_args[ind][coef])
                args_dict["spins"][coef].configure(command=cmd_type_to_hub(self.noise_args[ind], coef,\
                                                                            args_dict["spins"][coef],\
                                                                            self.noise_param_check_dict[coef]["type"],\
                                                                            self.noise_param_check_dict[coef]["condition"] ),\
                                                   validate="all",\
                                                   validatecommand=(master.register(validate_type_to_hub(self.noise_args[ind], coef,\
                                                                                    args_dict["spins"][coef],\
                                                                                    self.noise_param_check_dict[coef]["type"],\
                                                                                    self.noise_param_check_dict[coef]["condition"])), "%P"))
                
            # Размещение объектов 
            if title:
                frame_label.grid(row=r, padx=5, pady=5)
                r += 1
            noise_exist_cb.grid(row=r, padx=5, pady=5)
            r += 1
            param_frame.grid(row=r, padx=5, pady=5, sticky="WE")

            def param_frame_objects_placement():
                if self.noise_exist[ind]:
                    noise_type_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="WE")
                    noise_type_cb.grid(row=0, column=2, columnspan=4, padx=5, pady=5)
                    noise_func_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
                    noise_func_cb.grid(row=1, column=2, columnspan=4, padx=5, pady=5)
                    for i, p in enumerate(self.noise_func_param_dict[self.noise_func[ind]]):
                        args_dict["labels"][p].grid(row=2, column=i*2, pady=5)
                        args_dict["spins"][p].grid(row=2, column=i*2+1, pady=5)
                else:
                    noise_type_label.grid_forget()
                    noise_type_cb.grid_forget()
                    noise_func_label.grid_forget()
                    noise_func_cb.grid_forget()
                    for widget_type in args_dict.values():
                        for widget in widget_type.values():
                            widget.grid_forget()
            
            param_frame_objects_placement()
            # Локальные обработчики событий
            def noise_exist_handler(event):
                buf_exist = self.noise_exist_dict[event.widget.get()]
                if not self.noise_exist[ind] == buf_exist:
                    self.noise_exist[ind] = buf_exist
                    param_frame_objects_placement()

            def noise_type_handler(event):
                buf_type = self.noise_type_dict[event.widget.get()]
                if not self.noise_type[ind] == buf_type:
                    self.noise_type[ind] = buf_type

            def noise_func_handler(event):
                buf_func = self.noise_func_dict[event.widget.get()]
                if not self.noise_func[ind] == buf_func:
                    self.noise_func[ind] = buf_func
                    for widget_type in args_dict.values():
                        for widget in widget_type.values():
                            widget.grid_forget()
                    for i, p in enumerate(self.noise_func_param_dict[self.noise_func[ind]]):
                        args_dict["labels"][p].grid(row=2, column=i*2, pady=5)
                        args_dict["spins"][p].grid(row=2, column=i*2+1, pady=5)
            # Привязка к локальным обработчикам событий
            noise_exist_cb.bind("<<ComboboxSelected>>", noise_exist_handler)
            noise_type_cb.bind("<<ComboboxSelected>>", noise_type_handler)
            noise_func_cb.bind("<<ComboboxSelected>>", noise_func_handler)

        def real_choose(master):
            """
            Создание и размещение элементов выбора при действительном типе сигнала
            """
            noise_choose(master, "", 0)

        def complex_choose(master):
            """
            Создание и размещение элементов выбора при комплексном типе сигнала
            """
            ampl_frame = tk.Frame(master, bd=0)
            ampl_frame.grid(row=0) 
            phase_frame = tk.Frame(master, bd=0)
            phase_frame.grid(row=1)           
            noise_choose(ampl_frame, "Шум амплитуды сигнала", 0)
            noise_choose(phase_frame, "Шум фазы сигнала", 1)

        # Очистим блок
        for child in self.noise_frame.winfo_children():
            child.destroy()
        # Создание объектов блока
        cell_label = tk.Label(self.noise_frame, text="Параметры шума", bg="black", foreground="snow")
        noise_choose_frame = tk.Frame(self.noise_frame, bd=0)
        # Размещение объектов блока
        cell_label.grid(row=0, padx=5, pady=5, sticky="NWE")
        noise_choose_frame.grid(row=1, padx=5, pady=5)
        if self.signal_func_type == self.type_real:
            real_choose(noise_choose_frame)
        else:
            complex_choose(noise_choose_frame)


    def filter_cell(self):
        """
        Функция отрисовки блока управления параметрами фильтрации АЧХ и ФЧХ
        :param master: родительский элемент для данного блока
        :param grid_options: аргументы метода grid для размещения в родительском объекте
        """
        # Локальные вспомогательные функции
        def filter_choose(master, title:str, part: str, ind: int):
            """
            Создание и размещение элементов выбора параметров фильтрации
            :param master: родительский объект интерфейса 
            :param title: заголовок блока
            :param part: ключ для дотсупа к соответствующей части сигнла 
            :param ind: индекс в массиве куда будут считываться целевые параметры
            """
            # Создание объектов
            frame_label = tk.Label(master, text=title, bg="lemon chiffon")
            filter_exist_cb = ttk.Combobox(master, values=list(self.filter_exist_dict.keys()), state="readonly")
            for i, f in enumerate(self.filter_exist_dict.values()):
                if f == self.filter_exist[part][ind]:
                    filter_exist_cb.current(i)
            param_frame = tk.Frame(master, bd=0)
            filter_type_label = tk.Label(param_frame, text="Тип фильтра")
            filter_type_cb = ttk.Combobox(param_frame, values=list(self.filter_type_dict.keys()), state="readonly")
            for i, f in enumerate(self.filter_type_dict.values()):
                if f == self.filter_type[part][ind]:
                    filter_type_cb.current(i)
            filter_func_label = tk.Label(param_frame, text="Функция окна")
            filter_func_cb = ttk.Combobox(param_frame, values=list(self.filter_func_label_dict.keys()), state="readonly")
            for i, f in enumerate(self.filter_func_label_dict.values()):
                if f == self.filter_func_label[part][ind]:
                    filter_func_cb.current(i)
            args_dict = {"labels": dict(), "spins": dict()}
            for coef in self.filter_args[part][ind].keys():
                args_dict["labels"][coef] = tk.Label(param_frame, text=coef+" =")
                args_dict["spins"][coef] = ttk.Spinbox(param_frame, from_=-10**6, to=10**6, width=6)
                if coef == "width":
                    args_dict["spins"][coef].configure(increment=2)
                args_dict["spins"][coef].set(self.filter_args[part][ind][coef])
                args_dict["spins"][coef].configure(command=cmd_type_to_hub(self.filter_args[part][ind], coef,\
                                                                            args_dict["spins"][coef],\
                                                                            self.filter_param_check_dict[coef]["type"],\
                                                                            self.filter_param_check_dict[coef]["condition"] ),\
                                                   validate="all",\
                                                   validatecommand=(master.register(validate_type_to_hub(self.filter_args[part][ind], coef,\
                                                                                    args_dict["spins"][coef],\
                                                                                    self.filter_param_check_dict[coef]["type"],\
                                                                                    self.filter_param_check_dict[coef]["condition"])), "%P"))
            phase_filter_method_label = tk.Label(param_frame, text="Метод \nфильтрации фазы")
            phase_filter_method_cb = ttk.Combobox(param_frame, values=list(self.phase_filter_method_dict.keys()), state="readonly")
            for i, f in enumerate(self.phase_filter_method_dict.values()):
                if f == self.phase_filter_method[part]:
                    phase_filter_method_cb.current(i)   
            # Размещение объектов 
            frame_label.grid(row=0, padx=5, pady=5)
            filter_exist_cb.grid(row=1, padx=5, pady=5)
            param_frame.grid(row=2, padx=5, pady=5, sticky="WE")
            def param_frame_objects_placement():
                if self.filter_exist[part][ind]:
                    filter_type_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="WE")
                    filter_type_cb.grid(row=0, column=2, columnspan=4, padx=5, pady=5)
                    filter_func_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
                    filter_func_cb.grid(row=1, column=2, columnspan=4, padx=5, pady=5)
                    for i, p in enumerate(self.filter_func_param_dict[self.filter_func_label[part][ind]]):
                        args_dict["labels"][p].grid(row=2, column=i*2, pady=5)
                        args_dict["spins"][p].grid(row=2, column=i*2+1, pady=5)
                    if ind == 1:
                        phase_filter_method_label.grid(row=3, column=0, columnspan=2, pady=5)
                        phase_filter_method_cb.grid(row=3, column=2, columnspan=4, pady=5)
                else:
                    filter_type_label.grid_forget()
                    filter_type_cb.grid_forget()
                    filter_func_label.grid_forget()
                    filter_func_cb.grid_forget()
                    for widget_type in args_dict.values():
                        for widget in widget_type.values():
                            widget.grid_forget()
                    phase_filter_method_label.grid_forget()
                    phase_filter_method_cb.grid_forget()
            param_frame_objects_placement()

            # Локальные обработчики событий
            def filter_exist_handler(event):
                buf_exist = self.filter_exist_dict[event.widget.get()]
                if not self.filter_exist[part][ind] == buf_exist:
                    self.filter_exist[part][ind] = buf_exist
                    param_frame_objects_placement()

            def filter_type_handler(event):
                buf_type = self.filter_type_dict[event.widget.get()]
                if not self.filter_type[part][ind] == buf_type:
                    self.filter_type[part][ind] = buf_type

            def filter_func_handler(event):
                buf_func = self.filter_func_label_dict[event.widget.get()]
                if not self.filter_func_label[part][ind] == buf_func:
                    self.filter_func_label[part][ind] = buf_func
                    for widget_type in args_dict.values():
                        for widget in widget_type.values():
                            widget.grid_forget()
                    for i, p in enumerate(self.filter_func_param_dict[self.filter_func_label[part][ind]]):
                        args_dict["labels"][p].grid(row=2, column=i*2, pady=5)
                        args_dict["spins"][p].grid(row=2, column=i*2+1, pady=5)

            def phase_filter_method_handler(event):
                buf_method = self.phase_filter_method_dict[event.widget.get()]
                if not self.phase_filter_method[part] == buf_method:
                    self.phase_filter_method[part] = buf_method

            # Привязка к локальным обработчикам событий
            filter_exist_cb.bind("<<ComboboxSelected>>", filter_exist_handler)
            filter_type_cb.bind("<<ComboboxSelected>>", filter_type_handler)
            filter_func_cb.bind("<<ComboboxSelected>>", filter_func_handler)
            phase_filter_method_cb.bind("<<ComboboxSelected>>", phase_filter_method_handler)

        # Очистим блок
        for child in self.filter_frame.winfo_children():
            child.destroy()
        # Создание объектов блока
        cell_label = tk.Label(self.filter_frame, text="Параметры фильтрации \nв частотной области", bg="black", foreground="snow")
        filter_choose_frame = tk.Frame(self.filter_frame, bd=0)
        # Размещение объектов блока
        cell_label.grid(row=0, padx=5, pady=5, sticky="WE")
        filter_choose_frame.grid(row=1, padx=5, pady=5)
        if self.signal_func_type == self.type_real:
            ampl_frame = tk.Frame(filter_choose_frame, bd=0)
            ampl_frame.grid(row=0) 
            phase_frame = tk.Frame(filter_choose_frame, bd=0)
            phase_frame.grid(row=1)           
            filter_choose(ampl_frame, "Фильтрация АЧХ", "real", 0)
            filter_choose(phase_frame, "Фильтрация ФЧХ", "real", 1)
        else:
            # Создание объектов
            real_frame = tk.Frame(filter_choose_frame, bd=0)
            real_frame_label = tk.Label(real_frame, text="Спектр действительной части сигнала", bg="goldenrod1")
            real_ampl_frame = tk.Frame(real_frame, bd=0)
            real_phase_frame = tk.Frame(real_frame, bd=0)       
            filter_choose(real_ampl_frame, "Фильтрация АЧХ", "real", 0)
            filter_choose(real_phase_frame, "Фильтрация ФЧХ", "real", 1)
            imag_frame = tk.Frame(filter_choose_frame, bd=0)
            imag_frame_label = tk.Label(imag_frame, text="Спектр мнимой части сигнала", bg="goldenrod1")
            imag_ampl_frame = tk.Frame(imag_frame, bd=0)
            imag_phase_frame = tk.Frame(imag_frame, bd=0)       
            filter_choose(imag_ampl_frame, "Фильтрация АЧХ", "imag", 0)
            filter_choose(imag_phase_frame, "Фильтрация ФЧХ", "imag", 1)
            # Размещение объектов
            real_frame.grid(row=0)
            real_frame_label.grid(row=0)
            real_ampl_frame.grid(row=1)
            real_phase_frame.grid(row=2)
            imag_frame.grid(row=1)
            imag_frame_label.grid(row=0)
            imag_ampl_frame.grid(row=1)
            imag_phase_frame.grid(row=2)


    def plots_cell(self):
        """
        Функция отрисовки блока отображения графиков
        :param master: родительский элемент для данного блока
        :param grid_options: аргументы метода grid для размещения в родительском объекте
        """
        # Локальные вспомогательные функции
        def draw_some_dict(some_dict: dict):
            for r, c in product(range(self.subplots_shape[0]), range(self.subplots_shape[1])):
                some_dict[(r,c)].grid(row=1+r, column=c)

        def hide_some_dict(some_dict: dict):
            for r, c in product(range(self.subplots_shape[0]), range(self.subplots_shape[1])):
                some_dict[(r,c)].grid_forget()

        # Локальные обработчики событий
        def num_plots_spin_handler(ind: int, spin, types_dict, scale_dict):
            def func():
                buf = int(spin.get())
                if not self.subplots_shape[ind] == buf:
                    hide_some_dict(types_dict)
                    hide_some_dict(scale_dict)
                    self.subplots_shape[ind] = buf
                    draw_some_dict(types_dict)
                    draw_some_dict(scale_dict)
                    self.plots_area()
            return func

        def choose_plot_type(row: int, column: int, types_dict: dict, scale_dict: dict):
            def func(event):
                types_dict[(row, column)].configure(state="disable")
                buf_type = event.widget.get()
                if not self.subplots_types[(row, column)] == buf_type:
                    scale_dict[(row, column)].configure(state="disable")
                    self.subplots_types[(row, column)] = buf_type
                    if ("произв" in buf_type) or ("окно" in buf_type) or (not "АЧХ" in buf_type):
                        self.subplots_scale[(row, column)] = list(self.plots_scale_dict.keys())[0]
                        for i, l in enumerate(self.plots_scale_dict.keys()):
                            if l == self.subplots_scale[(row,column)]:
                                scale_plots_dict[(row,column)].current(i)
                        scale_state="disable"
                    else:
                        scale_state="readonly"
                    self.axes_dict[(row, column)].remove()
                    del self.axes_dict[(row, column)]
                    self.plots_type_dict[self.subplots_types[(row, column)]](row, column)
                    scale_dict[(row, column)].configure(state=scale_state)
                types_dict[(row, column)].configure(state="readonly")
            return func

        def choose_plot_scale(row: int, column: int, scale_dict: dict):
            def func(event):
                scale_dict[(row, column)].configure(state="disable")
                buf_type = event.widget.get()
                if not self.subplots_scale[(row, column)] == buf_type:
                    self.subplots_scale[(row, column)] = buf_type
                    self.axes_dict[(row, column)].remove()
                    del self.axes_dict[(row, column)]
                    self.plots_type_dict[self.subplots_types[(row, column)]](row, column)
                scale_dict[(row, column)].configure(state="readonly")
            return func

        # Очистим блок
        for child in self.plots_frame.winfo_children():
            child.destroy()
        # Объекты отрисовки графиков
        self.plots_display_frame = tk.Frame(self.plots_frame, bd=0)
        self.plots_figure = plt.figure(figsize=(19.5, 11))
        plt.subplots_adjust(top=0.94, bottom=0.07, left=0.048, right=0.97, hspace=0.3, wspace=0.12)
        self.axes_dict = dict()
        self.canvas = FigureCanvasTkAgg(self.plots_figure, self.plots_display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0)
        self.toolbar_frame = tk.Frame(self.plots_display_frame, bd=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        #self.toolbar.update()
        self.canvas.get_tk_widget().grid(row=0)
        def key_handler(event):
            key_press_handler(event, self.canvas, self.toolbar)
        self.canvas.mpl_connect("key_press_event", key_handler)
        self.plots_area()
        # Определение объектов блока
        plots_param_frame = tk.Frame(self.plots_frame, bd=0)
        plots_param_label = tk.Label(plots_param_frame, text="Параметры отображения графиков", bg="black", foreground="snow")
        number_plots_frame = tk.Frame(plots_param_frame)
        number_plots_label = tk.Label(number_plots_frame, text="Количество отображаемых графиков")
        number_plots_rows_label = tk.Label(number_plots_frame, text="строк")
        number_plots_rows_spin = ttk.Spinbox(number_plots_frame, values=list(range(self.plots_shape_min, self.plots_shape_max + 1)), width=4, state="readonly")
        number_plots_rows_spin.set(self.subplots_shape[0])
        number_plots_cols_label = tk.Label(number_plots_frame, text="столбцов")
        number_plots_cols_spin = ttk.Spinbox(number_plots_frame, values=list(range(self.plots_shape_min, self.plots_shape_max + 1)), width=4, state="readonly") 
        number_plots_cols_spin.set(self.subplots_shape[1])  
        types_plots_frame = tk.Frame(plots_param_frame, bd=0)
        types_plots_label = tk.Label(types_plots_frame, text="Отображаемые графики")
        scale_plots_frame = tk.Frame(plots_param_frame, bd=0)
        scale_plots_label = tk.Label(scale_plots_frame, text="Масштаб")
        types_plots_dict = dict()
        scale_plots_dict = dict()
        for r, c in product(range(self.plots_shape_max), range(self.plots_shape_max)):
            types_plots_dict[(r,c)] = ttk.Combobox(types_plots_frame, values=list(self.plots_type_dict.keys()), state="readonly", width=60)
            for i, l in enumerate(self.plots_type_dict.keys()):
                if l == self.subplots_types[(r,c)]:
                    types_plots_dict[(r,c)].current(i)
            types_plots_dict[(r,c)].bind("<<ComboboxSelected>>", choose_plot_type(r,c, types_plots_dict, scale_plots_dict))
            scale_plots_dict[(r,c)] = ttk.Combobox(scale_plots_frame, values=["линейный", "логарифмический"], state="disable")
            for i, l in enumerate(self.plots_scale_dict.keys()):
                if l == self.subplots_scale[(r,c)]:
                    scale_plots_dict[(r,c)].current(i)
            scale_plots_dict[(r,c)].bind("<<ComboboxSelected>>", choose_plot_scale(r,c, scale_plots_dict))
        number_plots_rows_spin.configure(command=num_plots_spin_handler(0, number_plots_rows_spin, types_plots_dict, scale_plots_dict))
        number_plots_cols_spin.configure(command=num_plots_spin_handler(1, number_plots_cols_spin, types_plots_dict, scale_plots_dict))        

        # Размещение объектов блока
        plots_param_frame.grid(row=0, pady=5)
        plots_param_label.grid(row=0, column=0, columnspan=2, pady=5)
        number_plots_frame.grid(row=1, column=0, sticky="w")
        number_plots_label.grid(row=0, column=0, columnspan=4, pady=5)
        number_plots_rows_label.grid(row=1, column=0, pady=5, sticky="e")
        number_plots_rows_spin.grid(row=1, column=1, pady=5, sticky="w")
        number_plots_cols_label.grid(row=1, column=2, pady=5, sticky="e")
        number_plots_cols_spin.grid(row=1, column=3, pady=5, sticky="w")
        types_plots_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="e")
        types_plots_label.grid(row=0, column=0, columnspan=self.plots_shape_max, pady=5)
        scale_plots_frame.grid(row=1, column=1, pady=5, sticky="e")
        scale_plots_label.grid(row=0, column=0, columnspan=self.plots_shape_max, pady=5)
        draw_some_dict(types_plots_dict)
        draw_some_dict(scale_plots_dict)
        self.plots_display_frame.grid(row=1, pady=5)
        self.toolbar_frame.grid(row=2)
        

    # Обработчики событий элементов графического интерфейса
    def plots_area(self):
        keys = list(self.axes_dict.keys())
        for k in keys:
            self.axes_dict[k].remove()
        self.axes_dict.clear()
        for r, c in product(range(self.subplots_shape[0]), range(self.subplots_shape[1])):
            self.plots_type_dict[self.subplots_types[(r,c)]](r,c)


    def signal_calc_handler(self):
        """
        Обработчик инициации пересчёта сигнала
        """
        self.signal_calc_button.configure(state="disable")
        # пересчитаем значение сигнала
        with warnings.catch_warnings():
            np.seterr(all='raise')
            try:
                self.signal_size[0] = self.signal_size_buf[0]
                self.def_dom = np.linspace(self.def_dom_borders[0], self.def_dom_borders[1], self.signal_size[0]).astype(self.type_real)  # область определения
                if self.signal_func_type == self.type_real: # действиельный сигнал
                    self.signal["clear"] = self.signal_func[0](self.def_dom, self.signal_args[0]).astype(self.signal_func_type)
                    if self.noise_exist[0]: # с шумом           
                        self.signal["source"] = self.noise_type[0](self.signal["clear"],\
                                                        self.noise_func[0](self.def_dom, self.noise_args[0]).astype(self.signal_func_type))
                    else:  # без наложения шума
                        self.signal["source"] = self.signal["clear"]
                else: # комплексный сигнал
                    signal_clear = np.zeros((2, self.signal_size[0]))
                    signal = np.zeros((2, self.signal_size[0]))
                    for ind in range(2):
                        signal_clear[ind] = self.signal_func[ind](self.def_dom, self.signal_args[ind]).astype(self.type_real)
                        if self.noise_exist[ind]:  # с шумом
                            signal[ind] = self.noise_type[ind](signal_clear[ind],\
                                                            self.noise_func[ind](self.def_dom, self.noise_args[ind]).astype(self.type_real))                    
                        else: # без шума
                            signal[ind] = signal_clear[ind]
                    self.signal["clear"] = signal_clear[0] * np.exp(signal_clear[1] * 1j)
                    self.signal["source"] = signal[0] * np.exp(signal[1] * 1j)

                # пересчитаем Фурье-образ
                for label in ["source", "clear"]:
                    s_fft = np.fft.fft(self.signal[label])
                    # Фурье-образов действительной части сигнала/ сигнала без наложения шума
                    self.fft[label]["real"] = fft_of_real_from_fft_of_complex(s_fft)
                    self.fft[label]["real"] = np.roll(self.fft[label]["real"], self.signal_size[0] // 2 - 1)
                    # Фурье-образов мнимой части сигнала/ сигнала без наложения шума
                    self.fft[label]["imag"] = fft_of_imag_from_fft_of_complex(s_fft)
                    self.fft[label]["imag"] = np.roll(self.fft[label]["imag"], self.signal_size[0] // 2 - 1)

                fft = dict()
                # Пересчитаем параметры окон фильтрации и проведём фильтрацию
                for part in ["real", "imag"]:
                    self.filter[part] = np.zeros((2, self.signal_size[0]))
                    self.filter[part][0] = self.filter_type[part][0](part, 0) if self.filter_exist[part][0] else np.ones(self.signal_size[0]) 
                    self.filter[part][1] = self.filter_type[part][1](part, 1) if self.filter_exist[part][1] else np.ones(self.signal_size[0]) 
                    fft["ampl"] = np.abs(self.fft["source"][part]) * self.filter[part][0] 
                    fft["phase"] = self.phase_filter_method[part](part) 
                    self.fft["filtered"][part] = fft["ampl"] * np.exp(fft["phase"] * 1j)
                
                # Получим сигнал из отфильтрованного спектра    
                self.signal["filtered"] = np.fft.ifft(fft_of_complex_from_real_and_imag_fft(np.roll(self.fft["filtered"]["real"], - self.signal_size[0] // 2 + 1),
                                                                                            np.roll(self.fft["filtered"]["imag"], - self.signal_size[0] // 2+ 1)))

                if self.signal_func_type == self.type_real:
                    self.signal["filtered"] = np.real(self.signal["filtered"]) 
            except Warning:
                messagebox.showwarning("Ошибка вычисления", "Выбранная функция сигнала или шума не определена \nв точках выбранной области определения.")
                self.signal_calc_button.configure(state="normal")
                return   
            except FloatingPointError:
                messagebox.showwarning("Ошибка вычисления", "Выбранная функция сигнала или шума не определена \nв точках выбранной области определения.")
                self.signal_calc_button.configure(state="normal")
                return   

        self.plots_type_dict.clear()
        # Заполним словарь функций отрисовки графиков
        for label in self.plots_types_labels:
            # Найдем массивы оси абсцисс и ординат
            if "налож" in label:
                source_label = "clear"
            elif "фильтр" in label:
                source_label = "filtered"
            else:
                source_label = "source"

            if "ЧХ" in label:
                x = np.arange(self.signal_size[0])
                if "мним" in label:
                    part = "imag"
                else:
                    part = "real"

                if "окно" in label:
                    y_func = lambda p: p
                    if "А" in label:
                        y_source = self.filter[part][0]
                    else:
                        y_source = self.filter[part][1]
                else:
                    y_source = self.fft[source_label][part]
                    if "А" in label:
                        y_func = np.abs
                    else:
                        y_func = np.angle
                    
                    if "произв" in label:
                        y_source = y_func(y_source)
                        if "втор" in label:
                            y_func = lambda x: np.block([0, np.diff(x, n=2), 0])
                        else:
                            y_func = first_phase_der


            else:
                x = self.def_dom
                y_source = self.signal[source_label]
                if "ампл" in label:
                    y_func = np.abs
                elif "фаз" in label:
                    y_func = np.angle
                elif "мним" in label:
                    y_func = np.imag
                else:
                    y_func = np.real

            self.plots_type_dict[label] = self.draw_plot_wrapper(x, y_source, y_func, label)

        # Отобразим изменения
        self.plots_area()
        self.signal_calc_button.configure(state="normal")    
        

    # Вспомогательные функции
    def main_size(self, height: int, width: int):
        self.window.geometry(str(width) + 'x' + str(height))
        self.window.minsize(width=width, height=height)
        self.window.maxsize(width=width, height=height)

    # Фунции фильтрации ФЧХ
    def filtering_phase(self, part: str):
        """
        Функция фильтрации фазы по первой (правой) производной
        """
        phfch = np.angle(self.fft["source"][part]) # ФЧХ сигнала
        size = self.signal_size[0]
        mean = np.median(np.diff(phfch[size // 2 - 1:]))
        # Фильтрация правой половины массива ФЧХ центрированного сигнала
        derivative = np.diff(phfch[size // 2 - 1:])
        derivative -= mean
        # Получим значения производной абсолютной фазы (если бы фаза не ограничивалась полуинтервалом [-pi; pi))
        derivative = np.where(derivative < -np.pi, derivative + 2 * np.pi, derivative)
        derivative = np.where(derivative > np.pi, derivative - 2 * np.pi, derivative)
        # Домножим на правую половину массива фильтра фазы
        derivative *= self.filter[part][1, size // 2:] 
        # Проинтегрируем полученную производную, приняв начальным условием значение стартового элемента правой половины
        derivative += mean
        phfch[size // 2 :] = phfch[size // 2 - 1] + np.cumsum(derivative)

        # Отражаем отфильрованную половину ФЧХ, используя свойство нечётности
        phfch[: size // 2 - 1] = -phfch[size // 2 :-1][::-1]
        return phfch


    # Функции рассчёта массивов окон фильтрации

    def make_lowpass_filter(self, part: str, ind: int) -> np.ndarray:
        """
        Функция создания фильтра низких частот
        """
        with warnings.catch_warnings():
            np.seterr(all='raise')
            try:
                if not "width" in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                    win_arg = [self.filter_func_label[part][ind]]
                    for a in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                        win_arg.append(self.filter_args[part][ind][a])
                    filter = get_window(tuple(win_arg), self.signal_size[0] + 1, fftbins=False)[1:]
                else:
                    filter = np.zeros(self.signal_size[0])
                    start = self.signal_size[0] // 2 - self.filter_args[part][ind]["width"] // 2 - 1
                    stop = self.signal_size[0] // 2 + self.filter_args[part][ind]["width"] // 2
                    win_arg = [self.filter_func_label[part][ind]]
                    for a in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                        if not a == "width":
                            win_arg.append(self.filter_args[part][ind][a])
                    filter[start: stop] = get_window(tuple(win_arg), self.filter_args[part][ind]["width"], fftbins=False)
                    c = np.min(filter[start: stop])
                    filter[:start] = c
                    filter[stop:] = c
            except FloatingPointError:
                messagebox.showwarning("Ошибка вычисления", "Выбранная функция окна фильтрации не может быть рассчитана при заданных параметрах. \nИзмените параметры и повторите попытку.")
                self.signal_calc_button.configure(state="normal")
                return self.filter[part][ind]
        return filter


    def make_highpass_filter(self, part: str, ind: int) -> np.ndarray:
        """
        Функция создания фильтра высоких частот
        """
        with warnings.catch_warnings():
            np.seterr(all='raise')
            try:
                if not "width" in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                    win_arg = [self.filter_func_label[part][ind]]
                    for a in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                        win_arg.append(self.filter_args[part][ind][a])
                    filter = get_window(tuple(win_arg), self.signal_size[0] + 1, fftbins=False)[:1]
                    min_val = filter[0]
                else:
                    filter = np.zeros(self.signal_size[0])
                    start = self.signal_size[0] // 2  - self.filter_args[part][ind]["width"] // 2 - 1
                    stop = self.signal_size[0] // 2 + self.filter_args[part][ind]["width"] // 2
                    win_arg = [self.filter_func_label[part][ind]]
                    for a in self.filter_func_param_dict[self.filter_func_label[part][ind]]:
                        if not a == "width":
                            win_arg.append(self.filter_args[part][ind][a])
                    filter[start: stop] = get_window(tuple(win_arg), self.filter_args[part][ind]["width"] + 1, fftbins=False)
                    min_val = np.min(filter[start: stop]) 
                    filter[:start] = min_val
                    filter[stop:] = min_val
                filter = 1 + min_val - filter
            except FloatingPointError:
                messagebox.showwarning("Ошибка вычисления", "Выбранная функция окна фильтрации не может быть рассчитана при заданных параметрах. \nИзмените параметры и повторите попытку.")
                self.signal_calc_button.configure(state="normal")
                return self.filter[part][ind]
        return filter


    def make_notch_filter(self, part: str, ind: int) -> np.ndarray:
        """
        Функция создания режекторного фильтра
        """
        return np.ones(self.signal_size[0])

    def make_narrow_filter(self, part: str, ind: int) -> np.ndarray:
        """
        Функция создания узкополосного фильтра
        """
        return np.ones(self.signal_size[0])

    def make_bandpass_filter(self, part: str, ind: int) -> np.ndarray:
        """
        Функция создания полосового фильтра
        """
        return np.ones(self.signal_size[0])


    # Функции отрисовки графиков
    def draw_plot_wrapper(self, x: np.ndarray, y_source: np.ndarray, y_func, title: str):
        """
        Декоратор функций отрисовщиков графиков
        :param x: массив значений фргументов
        :param y_source: массив значений источника отображаемой функции
        :param y_func: функция модификации источника значений функции
        :param title: заголовок графика
        """
        def func(row: int, column: int):
            y = y_func(y_source)
            self.axes_dict[(row,column)] = self.plots_figure.add_subplot(self.subplots_shape[0],\
                                                                self.subplots_shape[1],\
                                                                row * self.subplots_shape[1] + column + 1)

            if self.subplots_scale[(row, column)] == list(self.plots_scale_dict.keys())[0]:
                self.axes_dict[(row,column)].plot(x, y)
            else:
                y = np.clip(y, sys.float_info.min, None)
                self.axes_dict[(row,column)].semilogy(x, y)
            self.axes_dict[(row,column)].set_title(title)
            self.axes_dict[(row,column)].grid()
            self.canvas.draw()
        return func

            
if __name__ == "__main__":
    Stend()