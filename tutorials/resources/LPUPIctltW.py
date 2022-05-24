from manim import *
import math

data = [1.8929,
       6.3687,
       3.228,
       1.2192,
       0.2585,
       0.4404,
       3.0278,
       1.9918,
       3.4013,
       3.0343,
       1.0201,
       2.436,
       1.8981,
       2.9764,
       1.3621
]
class ExponentialDistributionMLE(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 5], y_range=[0, 2], axis_config={"include_tip": False}
        )
        axis_labels = ax.get_axis_labels(x_label="x", y_label="f(x) = \\lambda e^{-\\lambda x}")

        l = ValueTracker(5.0)

        def func(x):
            return l.get_value() * math.exp(-l.get_value() * x)

        graph = always_redraw(lambda: ax.plot(func, color=MAROON))

        self.add(ax, axis_labels, graph)

        for d in data:
            self.add(Circle(stroke_color=BLUE,fill_color=BLUE, radius=.08, fill_opacity=.95).move_to(ax.c2p(d,0)))

        for d in data:
            likelihood_line = always_redraw(lambda d=d: Line(start=ax.c2p(d,0), end=ax.c2p(d,func(d)), color=BLUE))
            self.add(likelihood_line)

        def dl_f(l, x):
            return (-l*x*math.exp(-l*x) + math.exp(-l*x))*math.exp(l*x)/l

        lr = .01
        iterations = 25

        lambda_label = MathTex("\\lambda = ").move_to(RIGHT * 2.0)
        value_label = always_redraw(lambda: Text("{:.2f}".format(l.get_value())).next_to(lambda_label, RIGHT))

        sum_label = MathTex("\sum log(f(x_i)) = ").next_to(lambda_label, DOWN)
        sum_value_label = always_redraw(lambda: Text("{:.2f}".format(l.get_value() + sum(dl_f(l.get_value(), x) for x in data))) \
                                        .next_to(sum_label, RIGHT))

        self.add(lambda_label, value_label, sum_label, sum_value_label)

        for i in range(iterations):
            new_l = l.get_value() + sum(dl_f(l.get_value(), x) for x in data) * lr
            self.play(l.animate.set_value(new_l), run_time=.5)
