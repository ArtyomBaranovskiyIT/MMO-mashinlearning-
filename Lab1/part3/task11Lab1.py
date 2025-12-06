class House:
    def __init__(self, area, price):
        self._area = area  # Площадь в м²
        self._price = price  # Цена
    
    def final_price(self, discount):
        """Возвращает цену с учетом скидки"""
        return self._price * (1 - discount / 100)

class SmallHouse(House):
    def __init__(self, price):
        # конструктор с фиксированной площадью 40м²
        super().__init__(40, price)

class Human:
    def __init__(self, name, money):
        self.name = name
        self.money = money
        self.house = None
    
    def __make_deal(self, house, price):
        """Приватный метод для совершения сделки"""
        self.money -= price
        self.house = house
        print(f"{self.name} купил дом за {price}!")
    print(__make_deal.__doc__)
    def buy_house(self, house, discount=0):
        """Покупка дома с проверкой средств"""
        final_price = house.final_price(discount)
        
        print(f"\n{self.name} хочет купить дом:")
        print(f"Исходная цена: {house._price}")
        print(f"Скидка: {discount}%")
        print(f"Финальная цена: {final_price}")
        print(f"Денег у {self.name}: {self.money}")
        
        if self.money >= final_price:
            self.__make_deal(house, final_price)
        else:
            print(f"Недостаточно денег! Нужно еще {final_price - self.money}")
    
    def info(self):
        """Информация о человеке"""
        print(f"\n{self.name}:")
        print(f"Деньги: {self.money}")
        if self.house:
            print(f"Дом: площадь {self.house._area}м², куплен за {self.house._price}")
        else:
            print("Дома нет")


print("\n=== ЗАДАНИЕ 2: ДОМА И ЛЮДИ ===")

big_house = House(100, 500000)
small_house = SmallHouse(200000)

ivan = Human("Иван", 300000)
maria = Human("Мария", 100000)

# Пытаемся купить дома
ivan.buy_house(small_house, 10)  # Со скидкой 
ivan.info()

maria.buy_house(big_house)  # Без скидки
maria.info()

