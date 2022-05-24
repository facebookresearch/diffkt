from manim import *

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
class NumberLine(Scene):
    def construct(self):
        nl = NumberLine( x_range=[0, 7])
        self.add(nl)
        for d in data:
            self.add(Circle(stroke_color=BLUE,fill_color=BLUE, radius=.08, fill_opacity=.95).move_to(nl.n2p(d)))


