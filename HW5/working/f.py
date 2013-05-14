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

def runonce(fin,fout,nnbrs,outdir='/tmp/xx/',indir='/export/scratch/csci5451_s13/hw2_data/'):
	global zlogname
	cmd='./cssd -nthreads=8 -nnbrs='+str(nnbrs)+' '+indir+fin+' '+outdir+fout+' >>  '+zlogname
	#cmd='./cssd -nthreads=8 -nnbrs='+str(nnbrs)+' '+indir+fin+' '+outdir+fout
	print cmd
	os.system(cmd)

print "++++++++++++++++++++++++++++++"
runinfo=info()
zlogname='./zlog'+runinfo.infostart_local_str+'.txt'
os.system('touch '+zlogname)
print 'zlogname',zlogname, '  runinfo ',runinfo.infostart_local_str
#nnbrlist=[100,50,20,10]
nnbrlist=[10,25,50,100]
#nnbrlist=[100]
#filelist=['sport10']
filelist=['sports','sports10']
#filelist=['sports']
os.system('make')
###############################################################
for fname in filelist:
	for nnbr in nnbrlist:
		runonce(fname+'.mat',fname+'.txt',nnbr,'/tmp/xx/')

