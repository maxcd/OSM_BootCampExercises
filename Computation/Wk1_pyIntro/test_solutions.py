# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""

import solutions as soln
import pytest

# Problem 1: Test the addition and fibonacci functions from solutions.py
def test_addition():
    assert soln.addition(2, 3) == 5, "Failed on positive integers"
    assert soln.addition(-2, 1) == -1; "Failed on negatiove and positive integers"
    assert soln.addition(-3, -5) == -8, "Failed on negative integers"

def test_smallest_factor():
    assert soln.smallest_factor(1) == 1, "Failed on 1"
    assert soln.smallest_factor(17) == 17, "Failed on a prime"
    assert soln.smallest_factor(22) == 2, "Failed on even number"
    assert soln.smallest_factor(21) == 3, "Failed on uneven number"


# Problem 2: Test the operator function from solutions.py
def test_operator():
    assert soln.operator(1, 2, "*") == 2, "Failed on multuplication"
    assert soln.operator(1, 2, "-") == -1, "Failed on subtraction"
    assert soln.operator(1, 2, "/") == .5, "Failed on dividing"
    assert soln.operator(1, 2, "+") == 3, "Failed on addition"
    pytest.raises(ValueError, soln.operator, a=1, b=0, oper="/")
    pytest.raises(ValueError, soln.operator, a=1, b=2, oper="plus")
    pytest.raises(ValueError, soln.operator, a=1, b=2, oper=7)
    pytest.raises(ValueError, soln.operator, a=1, b=2, oper="%")


# Problem 3: Finish testing the complex number class
@pytest.fixture
def set_up_complex_nums():
    number_1 = soln.ComplexNumber(1, 2)
    number_2 = soln.ComplexNumber(5, 5)
    number_3 = soln.ComplexNumber(2, 9)
    return number_1, number_2, number_3

def set_up_complex_zero():
    number_4 = soln.ComplexNumbers(0, 0)
    return number_4

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 + number_2 == soln.ComplexNumber(6, 7)
    assert number_1 + number_3 == soln.ComplexNumber(3, 11)
    assert number_2 + number_3 == soln.ComplexNumber(7, 14)
    assert number_3 + number_3 == soln.ComplexNumber(4, 18)

def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 * number_2 == soln.ComplexNumber(-5, 15)
    assert number_1 * number_3 == soln.ComplexNumber(-16, 13)
    assert number_2 * number_3 == soln.ComplexNumber(-35, 55)
    assert number_3 * number_3 == soln.ComplexNumber(-77, 36)

def test_complex_subtraction(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 - number_2 == soln.ComplexNumber(-4, -3)
    assert number_1 - number_3 == soln.ComplexNumber(-1, -7)
    assert number_2 - number_3 == soln.ComplexNumber(3, -4)
    assert number_3 - number_3 == soln.ComplexNumber(0, 0)

def test_coomplex_division(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 / number_2 == soln.ComplexNumber(3/10, 1/10)
    assert number_1 / number_3 == soln.ComplexNumber(4/17, -1/17)
    assert number_2 / number_3 == soln.ComplexNumber(11/17, -7/17)
    assert number_3 / number_3 == soln.ComplexNumber(1, 0)

    with pytest.raises(Exception) as excinfo:
        number_1 / soln.ComplexNumber(0, 0)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Cannot divide by zero"

def test_complex_equality(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert (number_1 == number_2) == False
    assert (number_2 == number_2) == True

def test_complex_string(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.__str__() == "1+2i"

def test_complex_norm(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.norm() == 2.23606797749979

def test_complex_conjugate(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.conjugate() == soln.ComplexNumber(1, -2)

# Problem 4: Write test cases for the Set game.

def test_iscard():
    assert soln.iscard("0123")==False
def test_isset():
    assert soln.isset("0111" ,"01111","01111")==True
@pytest.fixture
def set_up_sets():
    set1 = "set1.txt"
    set2 = "set2.txt"
    set3 = "set3.txt"
    set4 = "set4.txt"
    set5 = "wrongfile.txt"
    return set1, set2, set3, set4, set5

def test_set(set_up_sets):
    set1, set2, set3, set4, set5 = set_up_sets
    assert soln.count_sets(set1) == 6

    with pytest.raises(Exception) as excinfo:
        soln.count_sets(set2)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Should have 12 cards"

    with pytest.raises(Exception) as excinfo:
        soln.count_sets(set3)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Should not have duplicate cards"

    with pytest.raises(Exception) as excinfo:
        soln.count_sets(set4)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Input is invalid"

    with pytest.raises(Exception) as excinfo:
        soln.count_sets(set5)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "File does not exist"
