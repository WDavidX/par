import os, re, string, operator, datetime, sys, time,pprint
from datetime import date

datapath='/export/scratch/csci5451_s13/hw2_data/'
# datapath='/home/d/hw2_data/'
runoption=' -nnbrs=50 -minsim=.3 '
nonbrs=True
threadlist=[8,4,2]
outdir_prefix='outp'
timestring=datetime.datetime.now().strftime('%d-%H%M')

matfilelist=[datapath+file for file in os.listdir(datapath) if file.endswith(".mat")]
matfilelist=sorted(matfilelist,key=os.path.getsize)
matfilelist=matfilelist[:]
pprint.pprint( matfilelist)
time.sleep(1)

def printf(format, *args):
    sys.stdout.write(format % args)

def modifynthreads(nthreads,srcname='sd.c'):
	print "Modify src file: %s; NTHREADS %d"%(srcname, nthreads)
	find_nthreads=re.compile('#define NTHREADS .*')
	source_code_fin=open(srcname,'r')
	numthread_w=r'#define NTHREADS '+str(nthreads)
	m=re.sub(find_nthreads,numthread_w,source_code_fin.read())
	source_code_fin.close()
	source_code_fout=open('srcname','w')
	source_code_fout.write(m)
	source_code_fout.close()
	print
	os.system('make clean')
	os.system('make')
	print

def runparallel(fin,outdir,exename='sd',comparedir='./sernbrs/'):
	global runoption,nonbrs
	fin_base=os.path.splitext(os.path.basename(fin))[0]
	fout_nbrs=outdir+fin_base+'.nbrs'
	fout_log=outdir+fin_base+'.log'
	fout_sort=outdir+fin_base+'.snbrs'
	fcompare_sort=comparedir+'s_'+fout_log
	f_diff_sort='er_'+fin_base+'.dif'
	os.system('rm -f '+fout_log)

	run_p='./'+exename+runoption+' '+fin+' '+fout_nbrs+' > '+fout_log
	sort_p='sort -n -k 1 -k 2 '+fout_nbrs + ' > '+fout_sort
	if nonbrs:
		os.system('rm -f '+fout_nbrs)
	print run_p
	print sort_p

	os.system(run_p)
	os.system(sort_p)

	if os.path.exists(fcompare_sort):
		os.system('rm -f '+f_diff_sort)
		os.system('diff '+	fcompare_sort + ' '+fout_sort+' > ' +f_diff_sort)
	os.system('cat '+fout_log)
	return None


def runserial(fin,outdir='./sernbrs/',exename='sd'):
	global runoption,nonbrs
	fin_base=os.path.splitext(os.path.basename(fin))[0]
	fout_nbrs='s_'+outdir+fin_base+'.nbrs'
	fout_log='s_'+outdir+fin_base+'.log'
	fout_sort='s_'+outdir+fin_base+'.snbrs'
	os.system('rm -f '+fout_log)
	run_s='./'+exename+runoption+' '+fin+' '+fout_nbrs+' > '+fout_log
	sort_s='sort -n -k 1 -k 2 '+fout_nbrs + ' > '+fout_sort
	if nonbrs:
		os.system('rm -f '+fout_nbrs)
	print run_s
	print sort_s

	os.system('cat '+fout_log)



for thread in threadlist:
	modifynthreads(thread)
	outdir=outdir_prefix+timestring+'-th'+str(thread)+'/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	for infile in matfilelist:
		runparallel(infile,outdir)
