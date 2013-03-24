import os, re, string, operator, datetime, sys, time,pprint
from datetime import date
os.system("make")
nplistmpi=[2,3,4,5,6,7,8,9,10,16]
nlistmpi=[1, 10, 100, 1000]

nplistmpi=[2,3]
nlistmpi=[1,10]

timestring=datetime.datetime.now().strftime('%d-%H%M')
mpilogfile="zlogmpi"+timestring+".txt"
print "MPI logname: "+mpilogfile

os.system("rm -f "+mpilogfile+" && touch "+mpilogfile)
counter=0
for n in nlistmpi:
	for np in nplistmpi:
		cmd="mpirun -np "+str(np)+" ./mpiscan "+str(n) +" >> " +mpilogfile
		counter=counter+1
		cmd_counter="echo "+str(counter) + " >> "+mpilogfile
		os.system(cmd_counter) 
		print cmd
		os.system(cmd)
	

os.system("cat "+mpilogfile)
