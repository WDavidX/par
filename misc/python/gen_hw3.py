import string, os
from random import randint

num=100000000
maxr=100



filename='n'+str(num)+'.mydat'
print "max num =",num
print "filename=",filename
fout=open(filename,'w')

fout.write(str(num)+"\n")

for k in xrange(0,num,1):
	r=randint(1,maxr)
	fout.write(str(r)+"\n")

fout.close()
print "======================== " + filename +" ========================"
#os.system('cat '+filename)

