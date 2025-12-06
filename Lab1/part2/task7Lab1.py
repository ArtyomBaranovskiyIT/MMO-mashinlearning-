# 1. Напишите функцию is_password_good(password), которая принимает
# в качестве аргумента строковое значение пароля password и возвращает
# значение True если пароль является надежным и False в противном случае.
# Пароль является надежным если:
# – его длина не менее 8 символов;
# – он содержит как минимум одну заглавную букву (верхний регистр);
# – он содержит хотя бы одну цифру

def is_password_good_detailed(password):
    
    if len(password) < 8:
        print("Пароль слишком короткий (минимум 8 символов)")
        return False
    
    conditions = {
        'length': len(password) >= 8,
        'uppercase': any(char.isupper() for char in password),
        'digit': any(char.isdigit() for char in password)
    }
    
    
    if not conditions['uppercase']:
        print("Отсутствуют заглавные буквы")
    if not conditions['digit']:
        print("Отсутствуют цифры")
    
    return all(conditions.values())

def main():
    print("=== ПРОВЕРКА НАДЕЖНОСТИ ПАРОЛЯ ===")

while True:
    try:
        password = input("\nВведите пароль для проверки (или 'q' для выхода): ")
        
        if password.lower() == 'q':
            print("До свидания!")
            break
        
        if not password:
            raise ValueError("Вы не ввели пароль!")
        
        print(f"\nПроверка пароля: '{password}'")
        
        # Проверяем пароль
        if is_password_good_detailed(password):
            print("Пароль надежный!")
        else:
            print(" Пароль ненадежный. Рекомендации:")
            print("   - Минимум 8 символов")
            print("   - Хотя бы одна заглавная буква")
            print("   - Хотя бы одна цифра")
    
    except ValueError as e:
        print(f"{e}")
    
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
    
    finally:
        print("-" * 50)  # Разделитель после каждой проверки


main()
#Done