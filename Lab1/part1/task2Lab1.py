#2
#Посчитать, сколько пар (стоят рядом) верхнего и нижнего
#регистра находится в веденном с клавиатуры слове. (Пример HjkLM- 1
#пара нижнего, 1 пара верхнего), а также сколько всего букв в слове.

word = str(input("Введите слово: "))

upper = 0
lower = 0


for i in range(len(word) - 1) :
    char1 = word[i]
    char2 = word[i + 1]
    # Если оба символа - буквы
    if char1.isalpha() and char2.isalpha():
        # Если оба символа в верхнем регистре
        if char1.isupper() and char2.isupper():
            upper += 1
        # Если оба символа в нижнем регистре
        elif char1.islower() and char2.islower():
            lower += 1

total = sum(1 for char in word if char.isalpha())

print(f"Пары верхнего {upper}")
print(f"Пары нижнего {lower}")
print(f"Всего {total}")
#Done

