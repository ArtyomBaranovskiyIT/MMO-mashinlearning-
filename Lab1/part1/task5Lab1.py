#5. Реализуйте программу «Магазин автозапчастей», которая будет
# включать в себя шесть пунктов меню. У вас есть словарь, где ключ –
# название продукции. Значение – список, который содержит состав
# продукции, цену и кол-во (шт),которое есть в магазине.
# 1. Просмотр описания: название – описание
# 2. Просмотр цены: название – цена.
# 3. Просмотр количества: название – количество.
# 4. Всю информацию.
# 5. Покупка
# В пункте «Покупка» необходимо совершить покупку, с
# клавиатуры вводите название продукции и его кол-во, n – выход из
# программы. Посчитать цену выбранных товаров и сколько товаров
# осталось в изначальном списке
# 6. До свидания

def main():
    # Инициализация базы данных товаров
    # Ключ: название, Значение: [описание, цена, количество]
    products = {
        "Масло моторное": ["Синтетическое масло 5W-30", 2500, 15],
        "Воздушный фильтр": ["Фильтр воздушный оригинал", 1200, 8],
        "Тормозные колодки": ["Колодки передние керамические", 4500, 5],
        "Аккумулятор": ["Аккумулятор 60Ah", 8000, 3],
        "Свечи зажигания": ["Иридиевые свечи", 1500, 12],
        "Щетки стеклоочистителя": ["Щетки 60см каркасные", 1800, 10]
    }
    
    cart = []  # Корзина покупок: список кортежей (товар, количество)
    
    while True:
        print("\n" + "="*50)
        print("МАГАЗИН АВТОЗАПЧАСТЕЙ")
        print("="*50)
        print("1. Просмотр описания")
        print("2. Просмотр цены")
        print("3. Просмотр количества")
        print("4. Вся информация")
        print("5. Покупка")
        print("6. До свидания")
        print("="*50)
        
        choice = input("Выберите пункт меню (1-6): ")
        
        if choice == "1":
            view_descriptions(products)
        elif choice == "2":
            view_prices(products)
        elif choice == "3":
            view_quantities(products)
        elif choice == "4":
            view_all_info(products)
        elif choice == "5":
            shopping(products, cart)
        elif choice == "6":
            print("Спасибо за посещение! До свидания!")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

def view_descriptions(products):
    """Просмотр описаний товаров"""
    print("\n--- ОПИСАНИЯ ТОВАРОВ ---")
    for product, info in products.items():
        print(f"{product} – {info[0]}")

def view_prices(products):
    """Просмотр цен товаров"""
    print("\n--- ЦЕНЫ ТОВАРОВ ---")
    for product, info in products.items():
        print(f"{product} – {info[1]} руб.")

def view_quantities(products):
    """Просмотр количества товаров"""
    print("\n--- КОЛИЧЕСТВО ТОВАРОВ ---")
    for product, info in products.items():
        print(f"{product} – {info[2]} шт.")

def view_all_info(products):
    """Просмотр всей информации о товарах"""
    print("\n--- ВСЯ ИНФОРМАЦИЯ О ТОВАРАХ ---")
    print(f"{'Товар':<25} {'Описание':<30} {'Цена':<8} {'Кол-во':<6}")
    print("-" * 75)
    for product, info in products.items():
        print(f"{product:<25} {info[0]:<30} {info[1]:<8} {info[2]:<6}")

def shopping(products, cart):
    """Процесс покупки"""
    print("\n--- ПРОЦЕСС ПОКУПКИ ---")
    print("Доступные товары:")
    for product in products.keys():
        print(f"- {product}")
    
    total_cost = 0
    
    while True:
        print("\nВведите название товара для покупки или 'n' для выхода:")
        product_name = input().strip()
        
        if product_name.lower() == 'n':
            break
        
        if product_name not in products:
            print("Такого товара нет в магазине!")
            continue
        
        # Получаем информацию о товаре
        description, price, quantity = products[product_name]
        
        print(f"Товар: {product_name}")
        print(f"Описание: {description}")
        print(f"Цена: {price} руб.")
        print(f"В наличии: {quantity} шт.")
        
        try:
            buy_quantity = int(input("Введите количество для покупки: "))
            
            if buy_quantity <= 0:
                print("Количество должно быть положительным!")
                continue
                
            if buy_quantity > quantity:
                print(f"Недостаточно товара! В наличии только {quantity} шт.")
                continue
                
            # Добавляем в корзину
            cart.append((product_name, buy_quantity, price))
            
            # Обновляем количество в магазине
            products[product_name][2] = quantity - buy_quantity
            
            cost = buy_quantity * price
            total_cost += cost
            print(f"Добавлено в корзину: {buy_quantity} шт. {product_name}")
            print(f"Стоимость: {cost} руб.")
            print(f"Общая стоимость: {total_cost} руб.")
            
        except ValueError:
            print("Пожалуйста, введите число!")
    
    # Показываем итог покупки
    if cart:
        print("\n" + "="*50)
        print("ИТОГ ПОКУПКИ:")
        print("="*50)
        for item in cart:
            product, quantity, price = item
            print(f"{product}: {quantity} шт. × {price} руб. = {quantity * price} руб.")
        print(f"ОБЩАЯ СУММА: {total_cost} руб.")
        print("Спасибо за покупку!")
    else:
        print("Корзина пуста.")

main()
#Done