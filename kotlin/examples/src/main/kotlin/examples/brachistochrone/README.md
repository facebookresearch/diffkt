### Brachistochrome Curve

According to the Wikipedia, a Brachistochrome curve is "is the one lying on 
the plane between a point A and a lower point B, where B is not directly below A, 
on which a bead slides frictionlessly under the influence of a uniform gravitational 
field to a given end point in the shortest time."

[Brachistochrome Curve](https://en.wikipedia.org/wiki/Brachistochrone_curve)

This example of a Brachistochrome curve has two examples using **DiffKt**. One uses the `DTensor` api and 
the other uses the `DScalar` api. The examples call the `reverseDerivative()` function in the **DiffKt** api to 
use automatic differentiation to calculate the derivatives in the model used in the calculation. 