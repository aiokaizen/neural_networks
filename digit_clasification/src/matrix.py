import random


class Matrix:

    def __init__(self, rows, cols, default=0) -> None:
        self.rows = rows
        self.cols = cols

        self.data = [[] for _ in range(self.rows)]
        for i in range(self.rows):
            self.data[i] = [default for _ in range(self.cols)]
    
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = (random.random() - 0.5) * 2
    
    def print(self):
        print('======================\n')
        for row in self.data:
            print(row)
        print('\n======================\n')
    
    def add(self, value):
        if type(value) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] + value.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] + value

    def multiply(self, multiplier):
        if type(multiplier) == Matrix:
            if self.rows != multiplier.rows or self.cols != multiplier.cols:
                raise Exception("Cant multiply two maticies with different rows or cols.")
            # Hadamard product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] * multiplier.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] * multiplier

    def devide(self, devider):
        if type(devider) == Matrix:
            if self.rows != devider.rows or self.cols != devider.cols:
                raise Exception("Cant execute devision between two maticies with different rows or cols.")
            # Hadamard product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] / devider.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = self.data[i][j] / devider
    
    def map(self, f):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = f(self.data[i][j])
    
    def copy(self):
        return Matrix.from_2d_array(self.data)

    def flatten(self):
        array = []
        for i in range(self.rows):
            for j in range(self.cols):
                array.append(self.data[i][j])
        return array

    @classmethod
    def subtract(cls, m1, m2):
        result = Matrix(m1.rows, m1.cols)
        for i in range(m1.rows):
            for j in range(m1.cols):
                result.data[i][j] = m1.data[i][j] - m2.data[i][j]
        return result
    
    @classmethod
    def transpose(cls, m):
        result = Matrix(m.cols, m.rows)

        # Transpose matrix
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[j][i] = m.data[i][j]

        return result
    

    @classmethod
    def matrix_product(cls, a, b):
        if b.rows != a.cols:
            raise Exception("Number of rows and cols don't match.")
        
        output = Matrix(a.rows, b.cols)
        for i in range(output.rows):
            row = a.data[i]
            for j in range(output.cols):
                col = [b.data[k][j] for k in range(b.rows)]
                products = [a * b for a, b in zip(row, col)]
                output.data[i][j] = sum(products)
        return output
    
    @classmethod
    def from_array(cls, array):
        matrix = cls(len(array), 1)

        for i in range(len(array)):
            matrix.data[i][0] = array[i]
        
        return matrix

    @classmethod
    def from_2d_array(cls, array_2d):
        rows = len(array_2d)
        cols = len(array_2d[0])
        matrix = cls(rows, cols)

        for i in range(matrix.rows):
            for j in range(matrix.cols):
                matrix.data[i][j] = array_2d[i][j]
        
        return matrix
