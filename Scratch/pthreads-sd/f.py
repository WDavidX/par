import os, re


numthread=8
source_sdname='sd.c'
inputfilename='../hw2_data/sports.mat'

numthread_w=r'#define NTHREADS '+str(numthread)
print numthread_w

find_nthreads=re.compile('#define NTHREADS .*')
print find_nthreads
source_code_fin=open('sd.c','r')
m=re.sub(find_nthreads,numthread_w,source_code_fin.read())
source_code_fin.close()
source_code_fout=open('sd.c','w')
source_code_fout.write(m)
source_code_fout.close()



os.system('make clean')
os.system('make')


inputdatarawname=os.path.splitext(os.path.basename(inputfilename))[0]

outputfilename_serial=inputdatarawname+'_s.nbrs'
outputfilename_parallel=inputdatarawname+'_p.nbrs'

outputfilename_serial_reorder=inputdatarawname+'_rs.nbrs'
outputfilename_parallel_reorder=inputdatarawname+'_rp.nbrs' 

arglist=' -nnbrs=20 -minsim=.3 '+inputfilename

cmd_s='./sd_serial '+arglist+' '+outputfilename_serial 
cmd_p='./sd '+arglist+' '+outputfilename_parallel

print cmd_s
os.system(cmd_s)
print cmd_p
os.system(cmd_p)

os.system('touch '+outputfilename_serial_reorder)
os.system('touch '+outputfilename_parallel_reorder)

cmd_sort_s='sort -n -k 1 -k 2 '+outputfilename_serial+ '> '+outputfilename_serial_reorder
cmd_sort_p='sort -n -k 1 -k 2 '+outputfilename_parallel+ '> '+outputfilename_parallel_reorder

print cmd_sort_s
os.system(cmd_sort_s)

print cmd_sort_p
os.system(cmd_sort_p)

cmd_diff='diff '+outputfilename_serial_reorder+' '+ outputfilename_parallel_reorder
print cmd_diff
os.system(cmd_diff)
