if __name__ == "__main__":
    # Запускаем все демонстрации
    print("=" * 60)
    print("РЕШЕНИЕ ВСЕХ ЗАДАНИЙ")
    print("=" * 60)
    
    # Задание 1
    import math
    
    class Triangle:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
        
        def is_valid(self):
            return (self.a + self.b > self.c and 
                    self.a + self.c > self.b and 
                    self.b + self.c > self.a and
                    self.a > 0 and self.b > 0 and self.c > 0)
        
        def area(self):
            if not self.is_valid():
                return "Не существует"
            p = self.perimeter() / 2
            return round(math.sqrt(p * (p - self.a) * (p - self.b) * (p - self.c)), 2)
        
        def perimeter(self):
            if not self.is_valid():
                return "Не существует"
            return self.a + self.b + self.c
    
    print("1. Triangle(3,4,5):", f"существует={Triangle(3,4,5).is_valid()}, площадь={Triangle(3,4,5).area()}")
    
    # Задание 2
    class House:
        def __init__(self, area, price):
            self._area = area
            self._price = price
        
        def final_price(self, discount):
            return self._price * (1 - discount / 100)
    
    class SmallHouse(House):
        def __init__(self, price):
            super().__init__(40, price)
    
    class Human:
        def __init__(self, name, money):
            self.name = name
            self.money = money
            self.house = None
        
        def __make_deal(self, house, price):
            self.money -= price
            self.house = house
        
        def buy_house(self, house, discount=0):
            final_price = house.final_price(discount)
            if self.money >= final_price:
                self.__make_deal(house, final_price)
                return True
            return False
    
    print("2. Созданы классы House, SmallHouse, Human")
    
    # Задание 3
    class Shape:
        def area(self):
            pass
    
    class Circle(Shape):
        def __init__(self, radius):
            self.radius = radius
        
        def area(self):
            return round(3.14 * self.radius ** 2, 2)
    
    class Square(Shape):
        def __init__(self, side):
            self.side = side
        
        def area(self):
            return self.side ** 2
    
    class Rectangle(Shape):
        def __init__(self, width, height):
            self.width = width
            self.height = height
        
        def area(self):
            return self.width * self.height
    
    print("3. Созданы геометрические фигуры")
    
    # Задание 4
    class BankAccount:
        bank_name = "Банк"
        total_accounts = 0
        
        def __init__(self, owner, balance=0):
            self.owner = owner
            self.__balance = balance
            BankAccount.total_accounts += 1
        
        def deposit(self, amount):
            if amount > 0:
                self.__balance += amount
        
        @classmethod
        def change_bank_name(cls, new_name):
            cls.bank_name = new_name
        
        @staticmethod
        def validate_amount(amount):
            return amount > 0
    
    print("4. Создан класс BankAccount с разными методами")
    
    print("\nВсе задания выполнены успешно! ✅")