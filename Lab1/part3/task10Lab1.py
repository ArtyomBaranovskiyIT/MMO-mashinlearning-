import math

class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def is_valid(self):
        """Проверка на существование треугольника"""
        return (self.a + self.b > self.c and 
                self.a + self.c > self.b and 
                self.b + self.c > self.a and
                self.a > 0 and self.b > 0 and self.c > 0)
    
    def area(self):
        """Нахождение площади треугольника по формуле Герона"""
        if not self.is_valid():
            return "Треугольник не существует"
        
        p = self.perimeter() / 2  # Полупериметр
        area = math.sqrt(p * (p - self.a) * (p - self.b) * (p - self.c))
        return round(area, 2)
    
    def perimeter(self):
        """Нахождение периметра треугольника"""
        if not self.is_valid():
            return "Треугольник не существует"
        return self.a + self.b + self.c

# Демонстрация
print("=== ЗАДАНИЕ 1: КЛАСС TRIANGLE ===")
t1 = Triangle(3, 4, 5)
print(f"Треугольник 3,4,5: существует={t1.is_valid()}, площадь={t1.area()}, периметр={t1.perimeter()}")

t2 = Triangle(1, 1, 3)
print(f"Треугольник 1,1,3: существует={t2.is_valid()}, площадь={t2.area()}, периметр={t2.perimeter()}")