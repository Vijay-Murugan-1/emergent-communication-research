states = [16, 100, 1000, 1000000]
actions = [4, 6, 10]

for s in states:
    for a in actions:
        size = s * a
        print(f"States: {s}, Actions: {a} -> Q-table entries: {size}")
