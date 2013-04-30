import sys, os, time, platform, getpass

class info:
	import sys, os, time, platform, getpass
	def __init__(self):
		self.infouser=getpass.getuser()
		self.infostart=time.time()
		self.infostart_local=time.localtime(self.infostart)
		self.infostart_local_str= time.strftime("%m%d-%H-%M-%S", self.infostart_local)
		self.infopid=os.getpid()
		self.infohost=platform.uname()[1]
		self.infoend=None
		self.infoedn_local=None
		self.infoend_local_str=None
		self.tdiff=None
		
	def Tstart(self):
		x=time.time()
		self.infostart=x
		self.infostart_local=time.localtime(self.infostart)
		self.infostart_local_str= time.strftime("%m%d-%H-%M-%S", self.infostart_local)
		
		
	def Tend(self):
		time.sleep(0.1)
		self.infoend=time.time()
		self.infoend_local=time.localtime(self.infoend)
		self.infoend_local_str= time.strftime("%m%d-%H-%M-%S", self.infoend_local)
		self.tdiff=self.infoend-self.infostart
	
	def startstr(self):
		return  os.path.realpath(__file__)+"\n"+self.infouser+'@'+self.infohost+'\tPID='+str(self.infopid)+'\t Start on '+self.infostart_local_str
		
	def endstr(self):
		return 'Start on '+self.infostart_local_str	+'\tTime elapses (sec): \t'+"%.6f"%(self.tdiff)
	
	def __del__(self):
		class_name = self.__class__.__name__
		#print class_name, "destroyed"

def runonce(np,matname,vecname,outdir='/tmp/xx/',indir='/export/scratch/csci5451_s13/hw4_data/'):
	global zlogname
	cmd='mpirun -np '+str(np)+' --hostfile ~/hosts ~/f/matvec_mpi '+indir+matname+ ' '+indir+vecname+' '+outdir +  ' >> '+zlogname
	print cmd
	os.system(cmd)

matlistA=['m50000-A.ij','m100000-A.ij','m200000-A.ij','m400000-A.ij']
matlistB=['m50000-B.ij','m100000-B.ij','m200000-B.ij','m400000-B.ij']
veclist=['m50000.vec','m100000.vec','m200000.vec','m400000.vec']
#nplist=[1,2,4,8,16]
nplist=range(1,57,1)


matlistA= [matlistA[-1] ]
matlistB= [matlistB[-1] ]
veclist= [veclist[-1]]
#nplist=nplist[2]
nplist=nplist[ 1 : 2]


###########################################################################
stat=info()
stat.Tstart()
zlogname='zlogA'+stat.infostart_local_str+'.txt';
print 'The log name is ',zlogname

fh=open(zlogname,'w')
fh.write('%s\n'%(stat.startstr()))
fh.close()

counter=0
for k in xrange(len(matlistA)):
	for np in nplist:
		print "%d \tFile order %d (%s), NP %d"%(counter,k,  matlistA[k],np)
		fh=open(zlogname,'a')
		fh.write('%d\n'%(counter))
		fh.close()
		#runonce(np, matlistA[k],veclist[k])
		counter+=1

stat.Tend()

fh=open(zlogname,'a')
fh.write('%s\n'%(stat.endstr()))
fh.close()

os.system('tail -10 '+zlogname)

###########################################################################
stat=info()
stat.Tstart()
zlogname='zlogB'+stat.infostart_local_str+'.txt';
print 'The log name is ',zlogname

fh=open(zlogname,'w')
fh.write('%s\n'%(stat.startstr()))
fh.close()

counter=0
for k in xrange(len(matlistB)):
	for np in nplist:
		print k, np
		fh=open(zlogname,'a')
		fh.write('%d\n'%(counter))
		fh.close()
		runonce(np, matlistB[k],veclist[k])
		counter+=1

stat.Tend()

fh=open(zlogname,'a')
fh.write('%s\n'%(stat.endstr()))
fh.close()
os.system('tail -10 '+zlogname)
