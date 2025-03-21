x = [i for i in range(10)]
for idx, i in enumerate(x):
    if i < 5:
        x[idx] = idx + 10
        
print(x)