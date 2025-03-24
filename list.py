list = ["shree","shruti","harshvi"]
input = input("enter name:")
for i in range(0,len(list)):
    flag = 1
    if list[i] == input:
        flag = 0
        break
if flag == 1:
    print("not found")
if flag == 0:
    print("found")
    