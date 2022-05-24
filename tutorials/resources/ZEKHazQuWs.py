from manim import *
import math

class ExponentialDistributionArea(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 5], y_range=[0, 2], axis_config={"include_tip": False}
        )
        axis_labels = ax.get_axis_labels(x_label="x", y_label="f(x) = 2 e^{-2 x}")

        l = ValueTracker(2.0)

        def func(x):
            return l.get_value() * math.exp(-l.get_value() * x)

        x_label = MathTex("x = 1")

        x_line = Line(start=ax.c2p(1,0), end=ax.c2p(1,func(1)), color=BLUE)
        x_label2 = x_label.copy().move_to(ax.c2p(1,-.10,0))

        graph = ax.plot(func, color=MAROON)
        self.add(ax, axis_labels, graph, x_label)

        self.play(Transform(x_label, x_label2))
        self.play(Write(x_line))

        rects = ax.get_riemann_rectangles(graph, (0,1),color=BLUE, fill_opacity=0.7)
        self.play(Write(rects), duration=2)

        rea = ax.get_area(graph, (0, 1), color=BLUE, opacity=0.7)
        self.play(Transform(rects, rea))

        area_label = MathTex(.8647)
        area_label2 = area_label.copy().move_to(ax.c2p(.5,.25))
        self.play(Write(area_label))
        self.play(Transform(area_label, area_label2))

        self.wait(5)
        self.play(Uncreate(area_label),
                  Uncreate(rects),
                  Uncreate(x_line),
                  Uncreate(x_label))

