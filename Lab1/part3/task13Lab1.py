class BankAccount:
    # Атрибут класса
    bank_name = "Национальный Банк"
    total_accounts = 0
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Приватный атрибут
        BankAccount.total_accounts += 1
        self.account_number = BankAccount.total_accounts
    
    # Метод экземпляра
    def deposit(self, amount):
        """Пополнение счета"""
        if amount > 0:
            self.__balance += amount
            print(f"Пополнение на {amount}. Новый баланс: {self.__balance}")
        else:
            print("Сумма должна быть положительной")
    
    # Метод экземпляра
    def withdraw(self, amount):
        """Снятие со счета"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Снятие {amount}. Новый баланс: {self.__balance}")
            return amount
        else:
            print("Недостаточно средств или неверная сумма")
            return 0
    
    # Метод экземпляра
    def get_balance(self):
        """Получение баланса"""
        return self.__balance
    
    # Метод класса
    @classmethod
    def change_bank_name(cls, new_name):
        """Изменение названия банка"""
        cls.bank_name = new_name
        print(f"Название банка изменено на: {new_name}")
    
    # Статический метод
    @staticmethod
    def validate_amount(amount):
        """Проверка корректности суммы"""
        return isinstance(amount, (int, float)) and amount > 0
    
    # Специальный метод
    def __str__(self):
        return f"Счет #{self.account_number} ({self.owner}) в {self.bank_name}"


print("\n=== ЗАДАНИЕ 4: БАНКОВСКИЙ СЧЕТ ===")

# Создаем счета
account1 = BankAccount("Иван Петров", 1000)
account2 = BankAccount("Мария Сидорова", 500)

print(account1)
print(account2)


account1.deposit(300)
account1.withdraw(200)
account2.withdraw(600)  # Недостаточно средств


BankAccount.change_bank_name("Международный Банк")


print(f"Проверка суммы 100: {BankAccount.validate_amount(100)}")
print(f"Проверка суммы -50: {BankAccount.validate_amount(-50)}")

print(f"Всего счетов: {BankAccount.total_accounts}")