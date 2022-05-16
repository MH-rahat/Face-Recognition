import csv
import pandas as pd
reader0 = csv.reader(open("csvfiles/forhad_test.csv"))
reader1 = csv.reader(open("csvfiles/forhad_train.csv"))
reader2 = csv.reader(open("csvfiles/rahat_train.csv"))
reader3 = csv.reader(open("csvfiles/rahat_test.csv"))
reader4 = csv.reader(open("csvfiles/rifat_train.csv"))
reader5 = csv.reader(open("csvfiles/rifat_test.csv"))
reader6 = csv.reader(open("csvfiles/riyadh_train.csv"))
reader7 = csv.reader(open("csvfiles/riyadh_test.csv"))
f = open("csvfiles/data_combined.csv", "w")
writer = csv.writer(f)

for row in reader0:
    writer.writerow(row)
for row in reader1:
    writer.writerow(row)
for row in reader2:
    writer.writerow(row)
for row in reader3:
    writer.writerow(row)
for row in reader4:
    writer.writerow(row)
for row in reader5:
    writer.writerow(row)
for row in reader6:
    writer.writerow(row)
for row in reader7:
    writer.writerow(row)
f.close()

csv3= pd.read_csv('csvfiles/data_combined.csv')
keep=['pixels','person','usage']
csv3[keep].to_csv("csvfiles/data_combined_final.csv")  
csv4= pd.read_csv('csvfiles/data_combined_final.csv') 