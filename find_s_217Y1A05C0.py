import csv
a = []
print("217Y1A05C0 - Find-S")
with open("enjoysport.csv", "r") as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
print(*a, sep = "\n")
print("The total no. of instances are:", len(a))
num_attributes = len(a[0]) - 1
hypothesis = ['0'] * num_attributes
print("The initial hypothesis is:", hypothesis)
print("The hypothesis for the training instances:")
for i in range(len(a)): 
    if(a[i][num_attributes]=="yes"):
        for j in range(num_attributes):
            if hypothesis[j]=="0" or hypothesis[j]==a[i][j]:
                hypothesis[j]=a[i][j]
            else:
                hypothesis[j]="?"
    print(i+1, hypothesis)
print("\nMaximally specific hypothesis for given instances is:")
print(hypothesis)
