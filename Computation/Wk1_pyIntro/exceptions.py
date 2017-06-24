'''
This works because the difference of any such number entered in step_1 and the reverse
has the same absolute value 198.
'''
def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
                   "digits differ by 2 or more: ")
    first, last = int(list(step_1)[0]), int(list(step_1)[-1])

    if (abs(first - last)) < 2:
        raise ValueError('Difference between leading and last digit must be'
                         'at least 2')
    elif len(list(step_1)) != 3:
        raise ValueError('The number ou entered does not hava not three digits ')

    step_2 = input("Enter the reverse of the first number, obtained "
                   "by reading it backwards: ")
    if not list(step_2)[::-1] == list(step_1):
        raise ValueError('Entered number is not the reversed of the first number')

    step_3 =input("Enter the positive difference of these numbers: ")

    diff = abs(int(step_2) - int(step_1))
    if int(step_3) != diff:
        raise ValueError('Entered number is not the absolute difference')

    step_4 = input("Enter the reverse of the previous result: ")
    if not list(step_4)[::-1] == list(step_3):
        raise ValueError('Number you entered is not the reverse of the previous')

    print(str(step_3) + " + " + str(step_4) + " = 1089 (ta-da!)")

# Problem 2 try except else - Block
from random import choice

def random_walk(max_iters=1e12):
    walk = 0
    direction = [1, -1]
    for i in range(int(max_iters)):
        try:
            walk += choice(direction)
        except KeyboardInterrupt as k:
            print('process interrupted at iteration {}'.format(i))
        else:
            print('process completed')
        finally:
            return walk

rw = random_walk()
