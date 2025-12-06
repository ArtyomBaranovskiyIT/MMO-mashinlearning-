import math

class Shape:
   
    def area(self):
       
        raise NotImplementedError("Метод area() должен быть переопределен")

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        """Площадь круга"""
        return round(math.pi * self.radius ** 2, 2)

class Square(Shape):
    def __init__(self, side):
        self.side = side
    
    def area(self):
        """Площадь квадрата"""
        return self.side ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        """Площадь прямоугольника"""
        return self.width * self.height


print("\n=== ЗАДАНИЕ 3: ГЕОМЕТРИЧЕСКИЕ ФИГУРЫ ===")

shapes = [
    Circle(5),
    Square(4),
    Rectangle(3, 6)
]

for shape in shapes:
    print(f"{shape.__class__.__name__}: площадь = {shape.area()}")