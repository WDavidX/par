import os, re, string, operator, datetime, sys, time,pprint
from datetime import date
os.system("make")
nplistseg=[2,3,4,5,6,7,8]
nlistseg=['n1.dat','n10.dat','n100.dat', 'n1000.dat', 'n10000.dat','n100000.dat']

nplistseg=[2,3]
nlistseg=['n1.dat','n10.dat']

timestring=datetime.datetime.now().strftime('%d-%H%M')
seglogfile="zlogseg"+timestring+".txt"
print "seg logname: "+seglogfile

os.system("rm -f "+seglogfile+" && touch "+seglogfile)
counter=0
for n in nlistseg:
	for np in nplistseg:
		cmd="./ompsscan "+str(np)+" ./"+(n) +" outputseg.txt >> " +seglogfile
		counter=counter+1
		cmd_counter="echo "+str(counter) + " >> "+seglogfile
		os.system(cmd_counter) 
		print cmd
		os.system(cmd)
	

os.system("cat "+seglogfile)
