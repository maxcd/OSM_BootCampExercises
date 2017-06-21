import box as bx

remaining = list(range(1, 10))
pl_input = input("Numbers to eliminate: ")
eliminate = bx.parse_input(pl_input, remaining)

for i in range(len(eliminate)):
    remaining = remaining.remove(eliminate[i])
