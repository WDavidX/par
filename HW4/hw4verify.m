% verify
clear ;clc; format short
c=clock;
disp( sprintf( '%d-%d-%d %2d:%2d:%2d',c(1),c(2),c(3),c(4),c(5),c(6)))
tic
matname='m1000-B.ij';vecname='m1000.vec';
wcmat=load(matname);wcvec=load(vecname);
toc
nnz=length(wcmat);
n=length(wcvec);

i=1+wcmat(:,1);
j=1+wcmat(:,2);
val=wcmat(:,3);
b=wcvec;

% A=zeros(n,n);
for k=1:nnz
    A(1+wcmat(k,1),1+wcmat(k,2))=wcmat(k,3);
end

C=A*b;

outname=sprintf('ro%d.vec',n);

fh=fopen(outname,'w');
for k=1:n
    fprintf(fh,'%.15f\n',C(k));
end
fclose(fh);

cmd=sprintf('diff o%d.vec %s',n,outname)
system(cmd)
toc
