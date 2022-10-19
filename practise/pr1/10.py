import random

b = int(input("Write range of random array: "))
res = [random.randrange(-2000, 5000) for i in range(b)]


def search(res):
    print ("Array: " +  str(res))
    a = max(res)
    b = min(res)

    output =(a,b)
    print(tuple(output))

search(res)
    