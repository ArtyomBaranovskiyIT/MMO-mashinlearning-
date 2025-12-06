# Напишите функцию, которая будет принимать один аргумент. Если
# в функцию передаётся словарь, то вывести ключ с максимальным значением.
# Если список, то найти количество элементов, расположенных до
# первого отрицательного элемента.
# Число – проверить простое, или нет.
# Строка – вывести в обратном порядке. Найти сумму цифр в строке.
# Сделать проверку со всеми этими случаями. – 4 балла
def process_argument(arg):
   
    
    print(f"\nОбрабатываем: {arg} (тип: {type(arg).__name__})")
    
    if isinstance(arg, dict):
        # СЛУЧАЙ 1: СЛОВАРЬ
        if not arg:
            print("Словарь пуст!")
            return None
        
        # Находим ключ с максимальным значением
        max_key = max(arg, key=arg.get)
        print(f"Ключ с максимальным значением: '{max_key}' = {arg[max_key]}")
        return max_key
    
    elif isinstance(arg, list):
        # СЛУЧАЙ 2: СПИСОК
        if not arg:
            print("Список пуст!")
            return 0
        
        # Ищем первый отрицательный элемент
        for i, item in enumerate(arg):
            if item < 0:
                print(f"Элементов до первого отрицательного: {i}")
                return i
        
        print("Отрицательных элементов нет")
        return len(arg)
    
    elif isinstance(arg, int):
        # СЛУЧАЙ 3: ЧИСЛО
        if arg < 2:
            print(f"Число {arg} не является простым")
            return False
        
        # Проверяем на простоту
        is_prime = True
        for i in range(2, int(arg**0.5) + 1):
            if arg % i == 0:
                is_prime = False
                break
        
        print(f"Число {arg} {'простое' if is_prime else 'не простое'}")
        return is_prime
    
    elif isinstance(arg, str):
        # СЛУЧАЙ 4: СТРОКА
        # Выводим в обратном порядке
        reversed_str = arg[::-1]
        print(f"Строка в обратном порядке: '{reversed_str}'")
        
        # Находим сумму цифр в строке
        digit_sum = 0
        digits_found = []
        for char in arg:
            if char.isdigit():
                digit_sum += int(char)
                digits_found.append(char)
        
        if digits_found:
            print(f"Цифры в строке: {', '.join(digits_found)}")
            print(f"Сумма цифр: {digit_sum}")
        else:
            print("В строке нет цифр")
            digit_sum = 0
        
        return reversed_str, digit_sum
    
    else:
        print(f"Неподдерживаемый тип: {type(arg)}")
        return None

def main():
    """Тестирование функции со всеми случаями"""
    print("=== ТЕСТИРОВАНИЕ ФУНКЦИИ ===")
    
    # Тест 1: Словарь
    test_dict = {'a': 10, 'b': 25, 'c': 15, 'd': 30}
    process_argument(test_dict)
    
    # Тест 2: Список
    test_list1 = [1, 2, 3, -4, 5, 6]
    process_argument(test_list1)
    
    test_list2 = [1, 2, 3, 4, 5]  # Без отрицательных
    process_argument(test_list2)
    
    # Тест 3: Число
    process_argument(17)  # Простое
    process_argument(15)  # Не простое
    process_argument(1)   # Граничный случай
    
    # Тест 4: Строка
    test_string1 = "Hello123World45"
    process_argument(test_string1)
    
    test_string2 = "NoDigitsHere"
    process_argument(test_string2)
    
    # Дополнительные тесты
    print("\n" + "="*50)
    print("ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ:")
    
    # Пустые коллекции
    process_argument({})
    process_argument([])
    
    # Специальные случаи
    process_argument([-1, 2, 3])  # Отрицательный первый
    process_argument("54321")     # Только цифры
    process_argument(2)           # Наименьшее простое

main()
