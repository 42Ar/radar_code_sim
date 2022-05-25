dl1=127;
dl2=15;
if site<5
 if site<3
  nb=64;
  nl=128;
  sint=4e-6;
  nsp=127;
  range0=405e-6;
  ndg=150;
  ut=62;
  ipp=1250e-6;
  lt1=ut;
  scomb=[1 0 0 0];
  o2=ones(1,2);
  o4=ones(1,4);
  p2=2*ncal+2*(nsp+lt1+ut+1)+nl*(nsp+lt1+ut)+2*(nsp+(nb-1))+(dl1+dl2+1)*ndg;
 elseif site<5
  nb=61;
  nl=120;
  sint=2.4e-6;
  nsp=411;
  range0=270e-6;
  ndg=250;
  ut=59;
  ipp=1500e-6;
  lt1=ut;
  p2=[];
  o2=ones(1,1);
  o4=ones(1,3);
 end
 ncal=10;
else
 nb=61;
 nl=60;
 sint=2.4e-6;
 nsp=61;
 nsp=53;
 range0=-60.6e-6;
 ndg=nsp;
 lt1=0;
 ut=0;
 ipp=1500e-6;
 ncal=500;
 p2=[];
 o2=1;
 o4=ones(1,3);
end
 ld=length(dd_data);
 if site<5
  r0d=sum(reshape(rd(2*(nsp+ncal+lt1+ut+1)+[1:(nsp+lt1+ut)*nl]),nsp+lt1+ut,nl),2);
  rd(ld+(1:ndg))=r0d(1:ndg)*nb/(nb-1)*2;
  r0d=sum(reshape(rd(2*(nsp+ncal+lt1+ut+1)+nsp*2+(nl-2)+nl*(nsp+lt1+ut)+[1:ndg*dl1]),ndg,dl1),2);
 else
  r0d=sum(reshape(rd(nsp+2*ncal+lt1+ut+(1:(nsp+lt1+ut)*nl)),nsp+lt1+ut,nl),2);
  rd(ld+(1:ndg))=r0d(1:ndg)*nb/(nb-1)*2;
  r0d=sum(reshape(rd(2*nsp+2*ncal+lt1+ut+1+nl*(nsp+lt1+ut)+(1:ndg*(nl-1))),ndg,nl-1),2);
 end
 rd(ld+ndg+(1:ndg))=rd(ld+(1:ndg))+r0d;
%pp
 psig=[2*ncal+(nsp+lt1+ut+1)+nl*(nsp+lt1+ut) 2*ncal p2 ld];
 psamp=[nsp+[nb-1 lt1+ut+1*o2] ndg];
 plen=[nb o4]*sint;
 pdt=o4*sint;
 prange0=range0+[0 0 ipp 0]+[0 -lt1 -ut -lt1]*sint;
 if site>2, prange0(3)=[]; end
 bfac=1./[1 o2*6.3 63];
 pbac=[2 0 0*o2];
%noise
 back=0; cal=ncal; bsamp=ncal; csamp=ncal;
%sigspec
 sig=nsp+[2*ncal+lt1+ut+1 p2+1 2*ncal+(nsp+lt1+ut+1)+nl*(nsp+lt1+ut)+[0 dl1*ndg]];
 siglen=o4*nb*sint;
 sigdt=pdt;
 slagincr=[sint*o2 ipp (dl1+1)*ipp];
 swlag=o4/128;
 npulses=o4*640;
 maxlag=[nl*o2 dl1 dl2];
 srange0=range0+o4*sint*(1-lt1);
 sigsamp=[nsp+lt1+ut ndg ndg];
 nbits=[nb*o2 NaN NaN];
 if site>4
  sigtyp={'fdalt' 'puls2' 'puls2'};
  sgates=[nsp NaN NaN];
  sig0=[2*ncal NaN NaN];
 elseif site<5
  if site<3
   txs=0;
   txsam=134;
   ntx=256;
   txdt=pdt/2;
   if length(d_raw)>ntx*txsam
    amplim=ntx*txsam+[0 25*41984];
   end
  elseif site==3
   txs=[0 length(d_raw)/2];
   txsam=[128 128];
   ntx=[128*2 128*2];
   txdt=[pdt/2 pdt/2];
   if length(d_raw)>sum(ntx.*txsam)
    amplim=[ntx(1)*txsam(1) length(d_raw)/2;ntx(1)*txsam(1)+length(d_raw)/2 length(d_raw)];
   end
  elseif site==4
   txs=0;
   txsam=128;
   ntx=128*2;
   txdt=pdt/2;
   if length(d_raw)>ntx*txsam
    amplim=[ntx*txsam length(d_raw)];
   end
  end
  psig(1)=2*(nsp+ncal+lt1+ut+1)+nl*(nsp+lt1+ut);
  psamp(1)=2*nsp+2*nb-2;
  pdt(1)=sint/2;
  if site<3
   psamp(3)=nsp+2*ut+1;
   sig=[2*ncal+2*(nsp+lt1+ut+1) 2*ncal+2*(nsp+lt1+ut+1)+nl*(nsp+lt1+ut)+2*nsp+nl-2+[(dl1+dl2+1)*ndg+2*(nsp+2*ut+1) 0 dl1*ndg]];
   sigtyp={'fdalt' 'fdalt' 'puls2' 'puls2'};
   sgates=[nsp+lt1+ut nsp+2*ut NaN NaN];
   sig0=[2*ncal sig(2)-2*(nsp+2*ut+1) NaN NaN];
   sigdt(1:2)=sint/2;
   maxlag(1:2)=nl;
   srange0(2)=range0+ipp+sint*(1-ut);
   sigsamp=[nsp+lt1+ut NaN ndg ndg];
  else
   psamp(2)=nsp+2*ut+1;
   sig=[2*ncal+2*(nsp+lt1+ut+1) 2*ncal+2*(nsp+lt1+ut+1)+nl*(nsp+lt1+ut)+2*nsp+nl-2+[0 dl1*ndg]];
   sigtyp={'fdalt' 'puls2' 'puls2'};
   sgates=[nsp+lt1+ut NaN NaN];
   sig0=[2*ncal NaN NaN];
   sigdt(1)=sint/2;
   maxlag(1)=nl;
  end
 end
 if site==3 || (site==5 && length(d_raw)>0)
  ld2=ld/2;
  if site==3
   %txs=0;
   %txsam=942;
   %ntx=128*25;
   %txdt=pdt;
   r0d=sum(reshape(rd(ld2+2*(nsp+ncal+lt1+ut+1)+[1:(nsp+lt1+ut)*nl]),nsp+lt1+ut,nl),2);
   rd(ld+2*ndg+(1:ndg))=r0d(1:ndg)*nb/(nb-1)*2;
   r0d=sum(reshape(rd(ld2+2*(nsp+ncal+lt1+ut+1)+nsp*2+(nl-2)+nl*(nsp+lt1+ut)+[1:ndg*dl1]),ndg,dl1),2);
  else
   r0d=sum(reshape(rd(ld2+(nsp+2*ncal+lt1+ut+1)+(1:(nsp+lt1+ut)*nl)),nsp+lt1+ut,nl),2);
   rd(ld+2*ndg+(1:ndg))=r0d(1:ndg)*nb/(nb-1)*2;
   r0d=sum(reshape(rd(ld2+2*nsp+2*ncal+lt1+ut+1+nl*(nsp+lt1+ut)+(1:ndg*(nl-1))),ndg,nl-1),2);
  end
  rd(ld+3*ndg+(1:ndg))=rd(ld+(1:ndg))+r0d;
  lag0=ld+[NaN 0 ndg];
  w_lag0=[NaN 3200 3200];
  [psig,psamp,plen,pdt,prange0,back,cal,bsamp,csamp,sig,sigtyp,siglen,maxlag,sigdt,srange0,bacspec,pbac,sigsamp,bfac,npulses,slagincr,swlag,sig0,nbits,sgates,w_lag0]=...
  def_rep(2,1,ld2,psig,psamp,plen,pdt,prange0,back,cal,bsamp,csamp,sig,sigtyp,siglen,maxlag,sigdt,srange0,bacspec,pbac,sigsamp,bfac,npulses,slagincr,swlag,sig0,nbits,sgates,w_lag0);
  psig(2,3)=psig(1,3)+2*ndg;
  lag0=[lag0;lag0+2*ndg];
  %nraw=psamp(1)*128*25; amplim=[0 nraw-1;nraw 2*nraw-1];
 end
