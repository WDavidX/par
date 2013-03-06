import os

print "python scan ssh key"

for ct in range(1,10,10):
	cmd="ssh-keyscan parallel-"+str(ct)+".cselabs.umn.edu"
	print '\n',cmd
	os.system(cmd)

for ct in range(1,30,10):
	cmd="ssh-keyscan kh2170-"+"{0:02d}".format(ct)+".cselabs.umn.edu"
	print '\n',cmd
	os.system(cmd)

for ct in range(31,40,1):
	cmd="ssh-keyscan kh4250-"+"{0:02d}".format(ct)+".cselabs.umn.edu"
	print '\n',cmd
	os.system(cmd)
