import mysql.connector
myconn=mysql.connector.connect(host="localhost",user="root",passwd="srinu@1009",database="sampleDB")
cur=myconn.cursor()
cur.execute("select * from Students")
res=cur.fetchall()
print("Student details are:")
for x in res:
    print(x)
myconn.commit()
myconn.close()
