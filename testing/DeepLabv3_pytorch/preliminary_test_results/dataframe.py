import pandas as pd

def reshape(array):
    result = []
    count = 0
    temp = []

    for e in array:    
        temp.append(e)
        if count == 5:
            result.append(temp)
            count = 0
            temp = []
        else:
            count += 1

    return result

# nums = list(range(0, 120))
# reshaped = reshape(nums)


# nums = ["20", "19", "17-32", "61"]
# series = pd.Series(nums, dtype="string")
# print(series)


# for num in series:
#     print(num)

def parse_str_to_int(elem):
    try:
        return int(elem)
    except ValueError:
        lower, higher = list(map(lambda x: int(x), elem.split("-")))

        return int((lower + higher) / 2)

nums = ["20", "19", "17-32", "61"]
nums = pd.Series(nums, dtype="string")
print(nums)
nums = nums.transform(parse_str_to_int)
print(nums)
