import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st

st.title("📊 Graph Plotter with Critical Points")

st.info("""
Ֆունկցիաներ:

- x**2  (բարձրացնել աստիճան)
- sqrt(x)  (արմատ)
- Abs(x)  (մոդուլ)
- sin(x), cos(x), tan(x)
""")

x_range = st.slider("Ընտրեք միջակայքը", -50, 50, (-10, 10))

user_input = st.text_input("Ներմուծեք ֆունկցիան (օրինակ: x**3 - 3*x):")

if user_input:
    try:
        x = sp.Symbol('x', real=True)

        safe_dict = {
            "x": x,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
            "log": sp.log,
            "exp": sp.exp,
        }

        f_sym = sp.sympify(user_input, locals=safe_dict)
        f_prime = sp.diff(f_sym, x)

        try:
            critical_points = sp.solve(f_prime, x, domain=sp.S.Reals)
        except NotImplementedError:
            critical_points = []

        try:
            x_intercepts = sp.solve(f_sym, x)
        except NotImplementedError:
            x_intercepts = []

        f = sp.lambdify(
            x,
            f_sym,
            modules=[
                {
                    "cot": lambda t: 1 / np.tan(t),
                    "Abs": np.abs,
                    "sign": np.sign,
                    "log": np.log,
                    "exp": np.exp,
                },
                "numpy"
            ]
        )

        x_vals = np.linspace(x_range[0], x_range[1], 2000)

        with np.errstate(divide='ignore', invalid='ignore'):
            y_vals = np.array(f(x_vals), dtype=float)

        y_vals[~np.isfinite(y_vals)] = np.nan

        dy = np.abs(np.diff(y_vals, prepend=np.nan))
        finite_vals = y_vals[np.isfinite(y_vals)]

        if len(finite_vals) > 0:
            threshold = np.nanpercentile(np.abs(finite_vals), 95) * 2 + 5
            y_vals[dy > threshold] = np.nan

        fig, ax = plt.subplots(figsize=(10, 6))

        # График функции
        ax.plot(x_vals, y_vals, color='blue', linewidth=1.5, label='f(x)')

        # Легенда для точек
        ax.plot([], [], color='green', marker='o', linestyle='None', label='Minimum')
        ax.plot([], [], color='red', marker='o', linestyle='None', label='Maximum')
        ax.plot([], [], color='orange', marker='o', linestyle='None', label='Root')

        # Точки max/min
        for point in critical_points:
            try:
                if not point.is_real:
                    continue

                px = float(point)

                if not (x_range[0] <= px <= x_range[1]):
                    continue

                y_point = float(f(px))

                if not np.isfinite(y_point):
                    continue

                second_derivative = sp.diff(f_prime, x).subs(x, point)

                if second_derivative > 0:
                    ax.scatter(px, y_point, color='green', zorder=5)
                    ax.text(
                        px,
                        y_point,
                        f"Min\n({px:.2f}, {y_point:.2f})",
                        color='green',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6)
                    )

                elif second_derivative < 0:
                    ax.scatter(px, y_point, color='red', zorder=5)
                    ax.text(
                        px,
                        y_point,
                        f"Max\n({px:.2f}, {y_point:.2f})",
                        color='red',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6)
                    )

            except Exception:
                continue

        # Корни
        for root in x_intercepts:
            try:
                if not root.is_real:
                    continue

                rx = float(root)

                if not (x_range[0] <= rx <= x_range[1]):
                    continue

                ax.scatter(rx, 0, color='orange', zorder=5)

                ax.text(
                    rx,
                    0,
                    f"Root\n({rx:.2f}, 0)",
                    color='orange',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6)
                )

            except Exception:
                continue

        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"f(x) = {sp.pretty(f_sym)}")

        finite_vals = y_vals[np.isfinite(y_vals)]

        if len(finite_vals) > 0:
            y_min = np.nanpercentile(finite_vals, 2)
            y_max = np.nanpercentile(finite_vals, 98)
            margin = (y_max - y_min) * 0.1 + 1
            ax.set_ylim(y_min - margin, y_max + margin)

        # Легенда
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ошибка: {e}")
