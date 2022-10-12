import numpy as np

# Zad 1
arr1 = np.ones(50) * 5
print(arr1)

# Zad 2
arr2 = np.arange(1,26).reshape((5,5))
print(arr2)

# Zad 3
arr3 = np.arange(10, 51, 2).reshape((3,7))

# Zad 4
arr4 = np.identity(3) * 8
print(arr4)

# Zad 5
arr5 = np.arange(0, 1, 0.01).reshape((10,10))
print(arr5)

# Zad 6
lin = np.linspace(0, 1, 50)
print(lin)

# Zad 7
arr7 = arr2.flatten()[12:24]
print(arr7)

# Zad 8
arr8 = arr2[:, 4].reshape((5,1)) # wszystkie wiersze (:) z ostatniej kolumny (4)
print(arr8)

# Zad 9
suma = np.sum(arr2[3:5,:])
print(suma) # 205

# Zad 10
# Przygotować skrypt, który stworzy tensor (tablica wielowymiarowa) zawierający losowe wartości całkowite, losowym wymiarze i losowym rozmiarze każdego z wymiarów
row = np.random.randint(10)
col = np.random.randint(10)
tensor = np.random.randint(0, 10, size = (row, col))
print(tensor)