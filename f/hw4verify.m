% verify
clear all;clc; format short
matname='m1000-A.ij';vecname='m1000.vec';
wcmat=load(matname);wcvec=load(vecname);
% [nnz,~]=size(wcmat);
nnz=length(wcmat);
n=length(wcvec);

i=1+wcmat(:,1);
j=1+wcmat(:,2);
val=wcmat(:,3);
b=wcvec;

A=zeros(n,n);
for k=1:nnz
    A(1+wcmat(k,1),1+wcmat(k,2))=wcmat(k,3);
end

C=A*b


 