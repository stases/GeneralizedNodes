

# Make a function that accepts 3 arguments and some kwargs, simple new function that adds numbers
def my_func(a, b, c, **kwargs):
    return a + b + c

# Now make a config file that will have a,b,c and something else
config = {
    "b": 2,
    "c": 3,
    "d": 4,
}
a = 1
result = my_func(a, **config)
print(result)