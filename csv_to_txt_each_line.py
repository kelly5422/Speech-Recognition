import csv
import numpy as np

with open('sample.csv', newline='', errors='ignore') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		a=row[0]
		b=row[1]
		a = '%03d' % int(a)
		txt_path='./test/txt/p225/' + 'p225_' + a + '.txt'
		print(txt_path)
		f=open(txt_path, 'w')
		f.write(b)
		f.close()