import random

length = int(input("Write range of random array:"))
res = [random.randrange(-2000, 5000) for i in range(length)]
print(res)
shift = int(input("Write shift of random array:"))


def slicearray(res,shift):
    lst = res[shift:] + res[:shift]
    print(lst)

slicearray(res,shift)