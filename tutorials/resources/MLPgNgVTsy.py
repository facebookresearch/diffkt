from manim import *
import math

class ExponentialDistribution(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 5], y_range=[0, 2], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="f(x) = \\lambda e^{-\\lambda x}")

        l = ValueTracker(2.0)

        def func(x):
            return l.get_value() * math.exp(-l.get_value() * x)


        graph = always_redraw(lambda: ax.plot(func, color=MAROON))
        lambda_label = always_redraw(lambda: MathTex("\\lambda = {:.2f}".format(l.get_value())))

        self.add(ax, labels, graph, lambda_label, rea)
        self.play(l.animate.set_value(.01), run_time=4.0)
        self.play(l.animate.set_value(5.0), run_time=8.0)
        self.play(l.animate.set_value(2.0), run_time=4.0)
