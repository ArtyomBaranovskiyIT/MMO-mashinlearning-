# 6. В кортеже целых чисел найдите максимальный и минимальный
# элементы.
def find_min_max(numbers):
    if not numbers:  
        return None, None
    
    min_num = numbers[0]
    max_num = numbers[0]
    
    for num in numbers:
        if num < min_num:
            min_num = num
        if num > max_num:
            max_num = num
    
    return min_num, max_num

numbers_tuple = (5, 2, 8, 1, 9, 3, 7)
min_val, max_val = find_min_max(numbers_tuple)

print(f"Кортеж: {numbers_tuple}")
print(f"Минимальный элемент: {min_val}")
print(f"Максимальный элемент: {max_val}")
#Done