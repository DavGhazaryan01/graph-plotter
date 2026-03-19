import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st

st.title("📊 Graph Plotter with Critical Points")


x_range = st.slider("Ընտրեք միջակայքը", -50, 50, (-10, 10))

user_input = st.text_input("Ներմուծեք ֆունկցիան (Օրինակpip freeze > requirements.txt: x**3 - 3*x):")

if user_input:

    try:

        x = sp.symbols('x')
        f_sym = sp.sympify(user_input)

        f_prime = sp.diff(f_sym, x)

        critical_points = sp.solve(f_prime, x)

        x_intercepts = sp.solve(f_sym, x)

        f = sp.lambdify(x, f_sym, "numpy")

        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = f(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")

        for point in critical_points:
            if point.is_real:
                px = float(point)


                if x_range[0] <= px <= x_range[1]:

                    y_point = f(px)
                    second_derivative = sp.diff(f_prime, x).subs(x, point)

                    if second_derivative > 0:
                        ax.scatter(px, y_point, label="Minimum")
                        ax.text(px, y_point, "  Min")
                    elif second_derivative < 0:
                        ax.scatter(px, y_point, label="Maximum")
                        ax.text(px, y_point, "  Max")

        for root in x_intercepts:
            if root.is_real:
                rx = float(root)

                # 🔥 ДОБАВИЛИ: проверка диапазона
                if x_range[0] <= rx <= x_range[1]:
                    ax.scatter(rx, 0, label = "Root")
                    ax.text(rx, 0, "  Root")

        ax.axhline(0)
        ax.axvline(0)
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("Ошибка в функции. Проверьте ввод.")