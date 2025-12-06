# Даны два числа n и m. Создайте двумерный массив размером n×m и
# заполните его символами "." и "*" в шахматном порядке. В левом верхнем
# углу должна стоять точка

def create_chess_board(n, m):
    """ list comprehension"""
    return [['.' if (i + j) % 2 == 0 else '*' for j in range(m)] for i in range(n)]

print(create_chess_board(1,1))