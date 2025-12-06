#1
#Определить, сколько в числе четных цифр, а сколько нечетных.
#Число вводится с клавиатуры.
number = int(input("Введите число: ")) #сделать проверку на ввод
even = 0
odd = 0
#Проверка на ноль
if number == 0:
    even = 1
else:
    temp = number
    while temp > 0:
        last_digit = temp % 10  
        if last_digit % 2 == 0: 
            even += 1
        else:
            odd += 1
            
        temp //= 10   

print(f"В числе {number}: четных цифр - {even}")
print(f"В числе {number}: нечетных цифр - {odd}")


#Done