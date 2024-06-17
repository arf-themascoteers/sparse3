def determine_value(size):
    if size <= 5:
        return 0.3
    elif size >= 30:
        return 0.1
    else:
        return 0.3 - ((size - 5) / 25) * 0.2
    
for i in range(0, 36, 5):
    print(f"Size: {i}, Value: {determine_value(i)}")