#3. 
#Найдите сумму отрицательных элементов списка.
#Найдите сумму элементов списка между двумя первыми нулями. Если
#двух нулей нет в списке, то выведите ноль

numbers = [9,-3,1,-5,0,1,5,-12,3,0,1,3]
#numbers = list(map(int,input("Введите элементы списка через пробел: ").split()))
negative_sum = sum(x for x in numbers if x < 0)

try:
    first_zero = numbers.index(0)
    second_zero = numbers.index(0, first_zero + 1)

    sum_btw_zeros = sum(numbers[first_zero + 1:second_zero])#btw-between
    print(f"Сумма между двумя первыми нулями: {sum_btw_zeros}")

except ValueError:
    print("Сумма между двумя первыми нулями: 0")

print(f"Сумма отрицательных элементов: {negative_sum}")


#Done
