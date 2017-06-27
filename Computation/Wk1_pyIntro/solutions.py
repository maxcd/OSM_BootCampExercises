# solutions.py
"""Volume IB: Testing.
<Name>
<Date>
"""
import math

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.


def addition(a, b):
    return a + b


def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a + b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other):
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here
'''
    Each car has four attributes:
        - color:  red, green, purple (pos1)
        - shape: diamond, oval, sqiggly  (pos2)
        - quantity: one, two, or three (pos3)
        - pattern: solid, stripped, outline (pos4)

    of which every one can have one of three values
    each being represented by an integer between 1 and 3
    at the corresponding position

'''

def iscard(card):
    iscard = True
    if len(card) != 4:
        iscard = False
    for i in range(0, len(card)):
        if int(card[i]) < 0 or int(card[i])>2:
            iscard = False
    return iscard

def isset(card1, card2, card3):
    def allsame(a,b,c):
        return a==b and a==c
    def alldifferent(a,b,c):
        return len(set((a,b,c)))==3
    attr = [False, False, False, False]
    for i in range(0,min(len(card1), len(card2), len(card3))):
        if allsame(card1[i],card2[i],card3[i]) == True or alldifferent(card1[i],card2[i],card3[i]) == True:
            attr[i] = True
    if attr == [True, True, True, True]:
        isset = True
    else:
        isset = False
    return isset

def count_sets(filename):
    if os.path.exists(os.path.join(os.getcwd(),'hands', filename)) == False:
        raise ValueError("File does not exist")

    with open(os.path.join(os.getcwd(),'hands', filename)) as myfile:
        cards = myfile.read().splitlines()
        if len(cards) != 12:
            raise ValueError("Should have 12 cards")

        if all(iscard(card)==True for card in cards)==False:
            raise ValueError("Input is invalid")

        for i in range(0, 12):
            for j in range(i+1, 12):
                if cards[i] == cards[j]:
                    raise ValueError("Should not have duplicate cards")

        sets = 0
        for i in range(0, 12):
            for j in range(i+1, 12):
                for k in range(j+1,12):
                    if isset(cards[i], cards[j], cards[k]) == True:
                        sets = sets + 1

    return sets
