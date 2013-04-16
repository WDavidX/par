# FOR VLSI2 HW1 

import os
#define glable constant
#initial vth value and vth change steps
vth0_val=0.471
vth_begin=vth0_val*0.85
vt_step=0.471*0.03
#parameter width of keeper
width_val=180
width_begin=width_val*0.5 #the first width will fail
w_step=width_val*0.1


#define file handle
cir_fi = open ('domino.sp','r') #the original circuit file
cir_fo = open ('domino.out','w') #the revised circuit file
model_fi=open('NMOS_VTL.inc','r') #the original model file
model_fo=open ('NMOS_VTL.out','w') #the revised model file
result=open ('result.txt','w') #the final result file

# write title
title='''
SNM(Normalized to Vdd)	NMOS Threshold voltage	Keeper width \n
______________________________________________________________ \n
'''
result.write(title)


#start 
content_m=model_fi.read()
content_c=cir_fi.read()
######## define the re object##########################
import re
find_vt=re.compile(r'vth0 = 0.471') #key word to be substitude in model
find_w=re.compile(r'PARAM kw=80e-9') #key word to be substitude in circuit

w_cur=width_begin #initialize the width
for i in xrange(12):
	w_pre=str(w_cur-w_step)+'e-9'
	w_str=str(w_cur) #convert data to string
	w_exprnew='PARAM kw='+w_str+'e-9' #generate the new value
	c=re.sub(find_w,w_exprnew,content_c) #substitude
	cir_fo=open('domino.out','w') #new circuit file
	cir_fo.write(c)
	w_cur=w_cur+w_step #increase the width
############# modify the model file########################	
	vth_cur=vth_begin #initialize the vth
	for j in xrange(12):
		vth_pre=str(vth_cur-vt_step)
		vth_str=str(vth_cur)
		vt_exprnew='vth0 = '+vth_str
		m=re.sub(find_vt,vt_exprnew,content_m)
		model_fo=open ('NMOS_VTL.out','w') #the revised model file
		model_fo.write(m)
		vth_cur=vth_cur+vt_step
############ simulation with hspice########################
		os.system('hspice testrun_domino.sp')
############ get value from .mt0 file######################
		if (i!=0 and j!=0): 
			mt0_fi=open('testrun_domino.mt0','r')
			mt0_fi=mt0_fi.readlines() #read .mt0 in the form of lines
			data=mt0_fi[3].split() #split the line into each data value
			data_nomarl=float(data[0])/1.1
			voltage=str(data_nomarl) #find the value of voltage
			result.write(voltage+'\t'+'\t')
			result.write(vth_pre+'\t'+'\t'+'\t')
			result.write(w_pre+'\n')
