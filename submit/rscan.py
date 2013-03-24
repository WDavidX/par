import os, re, string, operator, datetime, sys, time,pprint
from datetime import date
os.system("make")
nplistscan=[2,3,4,5,6,7,8]
nlistscan=['n1.dat','n10.dat','n100.dat', 'n1000.dat', 'n10000.dat','n100000.dat']

nplistscan=[2,3]
nlistscan=['n1.dat','n10.dat']

timestring=datetime.datetime.now().strftime('%d-%H%M')
scanlogfile="zlogscan"+timestring+".txt"
print "Scan logname: "+scanlogfile

os.system("rm -f "+scanlogfile+" && touch "+scanlogfile)
counter=0
for n in nlistscan:
	for np in nplistscan:
		cmd="./ompscan "+str(np)+" ./"+(n) +" outputscan.txt >> " +scanlogfile
		counter=counter+1
		cmd_counter="echo "+str(counter) + " >> "+scanlogfile
		os.system(cmd_counter) 
		print cmd
		os.system(cmd)
	

os.system("cat "+scanlogfile)
