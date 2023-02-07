 
from cmath import phase
from math import ceil
import sys
from time import time
import numpy
import cv2
from scipy import stats,ndimage
from copy import deepcopy
import gc 
import os
import matplotlib.pyplot as plt 
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skimage.restoration import unwrap_phase

 

from keys import *
from plots import SnrPlot, PhasePlot, CoherencePlot, DopplerPlot

import h5py
import collections

from filter import KalmanLinearFiltering 

#from LAD import LAD

FLAG_ANALYSIS=False
MAX_SNR=None
 

class RD2():

    flagConfig=False
    flagCreateNPY=True

    def transpose_data(self,Dat,mode='T'):
        Data=deepcopy(Dat)
        ndim    =   Data.ndim
        data   =   deepcopy(Data)
        buff_array    =   []

        if ndim==3:
            
            (z,_,_) =   data.shape
 

            for nn in range(z):
                
                buff    =   []

                dBlock  =   data[nn]
                
                if mode=='T': 
                    dBlock  =   dBlock.T

                (x,y)   =   dBlock.shape
                
                for nx in range(x):

                    line=dBlock[nx]
                    buff.insert(0,line.tolist())

                buff  = numpy.array(buff)

                if mode=='B':
                    buff=buff.T

                buff_array.append(buff)
            
            buff_array=numpy.array(buff_array)
            return buff_array


        elif ndim==2:

            buff=[]
            (x,y)=data.shape
            
            if mode=='T':
                data=data.T

            for nx in range(x):

                line=data[nx]
                buff.insert(0,line.tolist())
            
            buff=numpy.array(buff)

            if mode=='B':
                buff=buff.T

            return buff
        
        elif ndim==1:

            return data[::-1]

            

       

        

        

            



    def __load_data(self, path, nFile ):
        mode_load='npy'

        #nFile=str(nFile).zfill(4)


        if mode_load=='h5':
        

            block=path+'/sfiles-{}.h5'.format(nFile)


            with h5py.File(block,'r') as hf:
                self.dataCoh=hf['coh'][:] 
                self.dataDop=hf['dop'][:]
                self.dataPha=hf['pha'][:]
                self.dataSNR=hf['snr'][:]
                self.originPha=hf['pha'][:]
                
                # self.ry =self.height= hf['yrange'][:]
            
            self.ry=self.height=numpy.load(path+'/yrange.npy')

                
        

        else:
            coh=path+'/scoh-{}.npy'.format(nFile)
            snr=path+'/ssnr-{}.npy'.format(nFile)
            dop=path+'/sdop-{}.npy'.format(nFile)
            pha=path+'/spha-{}.npy'.format(nFile)
            xti=path+'/sxti-{}.npy'.format(nFile)


            self.dataCoh    =   numpy.load(coh) 
            self.dataDop    =   numpy.load(dop)
            self.dataPha    =   numpy.load(pha)
            self.dataSNR    =   numpy.load(snr)
            
            self.originPha  =   deepcopy(self.dataPha)
            self.ry=self.height=numpy.load(path+'/range_y.npy')
 


        
 
        #self.xti             =   numpy.load(xti)
        ry=path+'/range_y.npy'
        
    
        # if FLAG_ANALYSIS:
        #     rrr=7000
        #     self.dataCoh=self.dataCoh[:,rrr:,:]
        #     self.dataDop=self.dataDop[:,rrr:,:]
        #     self.dataPha=self.dataPha[:,rrr:,:]
        #     self.dataSNR=self.dataSNR[:,rrr:,:]
        return  self.ry
         

    def process_filter_phase(self,):

        mask=(5,3)

        z,_,_ = self.dataPha.shape

        for n in range(z):

            block=self.dataPha[n]*numpy.pi/180

            for _ in range(20):

                block=numpy.exp(1j*block)
                print(self.dataPha.shape)
                real,imag=block.real,block.imag
                real,imag=ndimage.uniform_filter(real,size=mask),ndimage.uniform_filter(imag,size=mask)

                block=numpy.arctan2(imag,real)
            

            self.dataPha[n]=block*180/numpy.pi




    def operation_set_MAX_SNR(self,pairs):

        snr=deepcopy(self.dataSNR)
        z,x,y=snr.shape

        limit=int((CRITERION_AVG_SNR/100)*x*y)



        #Distinct limit= 

        self.dv=[]

        for n,pair in enumerate(pairs):

            ch0=pair[0]
            ch1=pair[1]

            dv=abs(self.dataSNR[ch0]-self.dataSNR[ch1])

            dv=numpy.mean(dv)

            self.dv.append(dv)

        self.dv=numpy.array(self.dv)
            
        snr=numpy.mean(snr,axis=0)
        snr=numpy.ravel(snr)
        snr=numpy.sort(snr)

        self.MAX_SNR=numpy.mean(snr[-2:])
        print('set_max_SNR',self.MAX_SNR,snr[0],snr[-2:])




    def operation_thresh(self,thresh,pairs,SNRVar):

        #---------------------- Parameters -----------------------------
        Cohthresh,DerPhaThresh,SNRThresh=thresh[0],thresh[1],thresh[2]

        #--------------------- Aliasing velocity -----------------------

        vDopplerMax =   200


        #-------------------- Median Filter ---------------------------
 
        self.dataCoh=ndimage.median_filter(self.dataCoh,size=(1,3,3))
        self.dataCoh=ndimage.median_filter(self.dataCoh,size=(1,3,3))

        self.dataSNR=ndimage.median_filter(self.dataSNR,size=(1,3,3))
        self.dataSNR=ndimage.median_filter(self.dataSNR,size=(1,3,3))


        #(3, 9000, 62)
        #(nC, time, heights)
        #Grados sexagecimales 


        # self.dataPha=ndimage.median_filter(self.dataPha,size=(1,5,1)) 
        # print(self.dataPha.shape,'formnat')

        self.process_filter_phase()

        self.operation_set_MAX_SNR(pairs)

        SNR_MASK=True

        if SNR_MASK:


            if SNRVar ==False: 
                (z,_,_) =   self.dataSNR.shape

                maskSNR =   deepcopy(self.dataSNR)

                for nn in range(z):

                    maskSNR[nn] =   deepcopy(self.dataSNR[nn]>SNRThresh)
    
    
            else:

                (z,_,_)=self.dataSNR.shape

                maskSNR    =   deepcopy(self.dataSNR)


                for n in range(z):

                    
                    dSNR    =   deepcopy(maskSNR[n])
                    dSNR    =   dSNR.astype('float32')
                    dSNR    =   numpy.where(dSNR<0,numpy.nan,dSNR)

                    uSNR    =   deepcopy(dSNR)
                    uSNR    =   uSNR.astype('float32')

                    (x,y)   =   dSNR.shape

                    for nr in range(y):

                        line    =   dSNR[:,nr]
                        vline   =   line[~numpy.isnan(line)]
                        vline     =   numpy.array_split(numpy.sort(vline),5)
                        tH      =   stats.trim_mean(vline[-3],0.4)

                        #------------------ Criterion ------------------------

                        if tH<SNRThresh:
                            tH=SNRThresh
                        
                        line    =   line>tH
                        dSNR[:,nr]  =   line

                    dSNR    =   dSNR.astype('bool')
                    maskSNR[n] =   dSNR
                    indVal  =   numpy.where(~dSNR)
                    
                    uSNR[indVal]  =   numpy.nan
                    self.dataSNR[n]  =   uSNR
                    
            maskSNR =   maskSNR.astype('bool')

        maskCoh=    self.dataCoh>Cohthresh
 

        maskDop =   numpy.abs(self.dataDop)<vDopplerMax

        #------------- Mask Phase---------------------------

        

        maskPha=numpy.abs(self.dataPha[:,:,1:]-self.dataPha[:,:,:-1])
        maskPha=ndimage.median_filter(maskPha,size=(1,5,5))
        
        #maskPha=ndimage.median_filter(maskPha,size=5)
        maskPha=maskPha<30

        maskPha=numpy.dstack((maskPha,numpy.full(( maskPha.shape[0],maskPha.shape[1],1), True, dtype=bool)))



        
        for n,pair in enumerate(pairs):

            #Coherence y fase tiene el numero de pares
            #Snr y Dop tienen el numero de canales

            #------------------ Select Channels --------------------------

            ch0=pair[0]
            ch1=pair[1]

            #------------------- Buffering Data ------------------------

            dataPhase=deepcopy(self.dataPha[n])
            dataPhase=dataPhase.astype('float32')

            dataCoh=deepcopy(self.dataCoh[n])
            dataCoh=dataCoh.astype('float32')


            #-------------------- Make mask SNR -----------------------------

            if SNR_MASK: mask_SNR =   maskSNR[ch0] | maskSNR[ch1]

            #------------- Throw away this part -------------------------
            if FLAG_ANALYSIS:

                iSNR=numpy.where(~mask_SNR)

                copySNR=deepcopy(self.dataSNR[pair]) 

                b1=copySNR[0]
                b1[iSNR]=numpy.nan
                copySNR[0]=b1

                b2=copySNR[1]
                b2[iSNR]=numpy.nan
                copySNR[1]=b2
            

                SnrPlot(copySNR,        name="SNR_filter_pair:{}".format(n))
                SnrPlot(self.dataSNR[pair],   name="SNR_origin_pair:{}".format(n))

            #------------------------------------------------------------                
                
            mask_Coh =   maskCoh[n]

            if FLAG_ANALYSIS:

                #------------------------------------------------------------
                ############################################################

                copyCoh=deepcopy(self.dataCoh[n])
                iCoh= numpy.where(~mask_Coh)

                copyCoh[iCoh]=numpy.nan
                (x,y)=copyCoh.shape

                CoherencePlot(copyCoh.reshape((1,x,y)),name='Coh_Filter_Pair:{}'.format(n))
                CoherencePlot(self.dataCoh[n].reshape((1,x,y)),name='Coh_Origin_Pair:{}'.format(n))
                #############################################################


            mask_dop =   maskDop[ch0] | maskDop[ch1]

            if FLAG_ANALYSIS:
                #------------- Throw away this part -------------------------
                #------------------------------------------------------------

                iDop=numpy.where(~mask_dop)

                copyDop=deepcopy(self.dataDop[pair]) 

                b1=copyDop[0]
                b1[iDop]=numpy.nan
                copyDop[0]=b1

                b2=copyDop[1]
                b2[iDop]=numpy.nan
                copyDop[1]=b2
            

                DopplerPlot(copyDop,        name="Dop_filter_pair:{}".format(n))
                DopplerPlot(self.dataDop[pair],   name="Dop_origin_pair:{}".format(n))

                #------------------------------------------------------------   



            #mask_res =    mask_Coh & maskPha[n]
            # mask_res =   mask_SNR & mask_Coh & maskPha[n]
            mask_res =   mask_SNR & mask_Coh 

            
            #-------------------- Save Value Mask -----------------------------

            indNan=numpy.where(~mask_res)


            dataPhase[indNan]=numpy.nan
            dataCoh[indNan]=numpy.nan

            #-------------------- Save Data --------------------------------

            self.dataPha[n]=dataPhase
            self.dataCoh[n]=dataCoh

            (x,y)=dataCoh.shape

            if FLAG_ANALYSIS:


                CoherencePlot(dataCoh.reshape((1,x,y)),name='Coh_Resulting_Pair:{}'.format(n))
                PhasePlot(dataPhase.reshape((1,x,y)),name='Pha_Resulting_Pair:{}'.format(n))


        

        

    
    def difference_gaussian(self,img,k1,s1,k2,s2,k3,s3):

        b1 = cv2.GaussianBlur(img,(k1, k1), s1)
        b2 = cv2.GaussianBlur(img,(k2, k2), s2)
        b3 = cv2.GaussianBlur(img,(k3, k3), s3)

        diff1   =   b1  -   b2
        diff2   =   b2  -   b3
    
        if FLAG_ANALYSIS: cv2.imwrite('gaussian_blur.png', diff1 & diff2 )
        return diff1  


    
    def get_data(self,dataOut):

        buff            =   dataOut.Buff

        self.dataCoh    =   deepcopy(buff[0])
        self.dataPha    =   deepcopy(buff[1])
        self.dataSNR    =   deepcopy(buff[2])
        self.dataDop    =   deepcopy(buff[3])


        self.dataSNR    =   10*numpy.log(self.dataSNR)
        self.dataDop    =   10*numpy.log(self.dataDop)


        dataOut.Buff    =   None

        gc.collect()



    def __operationOC(self,mask):


        #----------- Set values ----------------

        dilate      =   cv2.getStructuringElement(cv2.MORPH_RECT, (4,3))
        erode       =   cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        #----------- Process -------------------
 
        mask        =   cv2.dilate(mask,dilate,iterations=1)


        mask        =   cv2.erode(mask,erode,iterations=1)
        mask        =   cv2.dilate(mask,dilate,iterations=1)
        mask        =   cv2.erode(mask,erode,iterations=2)
        
        # mask        =   cv2.erode(mask,erode,iterations=1)

        

        return mask 


    def filter__contours(self,mask):

        mask        = self.__operationOC(mask)

        if FLAG_ANALYSIS:    cv2.imwrite('openclose.png',mask)

        mask        =   self.difference_gaussian(mask,3, 5, 9, 11,13,15)

        contours, _ =   cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return contours 



    def detect_events(self,pairs):

        buff_contours   =   []
        #-------------- Transpose data -----------------

        self.dataCoh =  self.transpose_data(self.dataCoh)
        self.dataPha =  self.transpose_data(self.dataPha) 
        self.dataSNR =  self.transpose_data(self.dataSNR)
        self.originPha  =   self.transpose_data(self.originPha)

        for n,pair in enumerate(pairs):
 
            ch0,ch1     =   pair[0],pair[1]



            #-----------------------------------------------

            dataCoh     =   self.dataCoh[n]

            #--------- Generate mask ------------------

            maskCoh     =   numpy.where(dataCoh>0,0,1)
            maskCoh    *=   255
            maskCoh     =   maskCoh.astype(numpy.uint8)

            if FLAG_ANALYSIS: cv2.imwrite('binarizacion_pair{}.png'.format(n),maskCoh)
            
            #---------- Detect Contours ---------------

            contours    =   self.filter__contours(maskCoh)
        
            
            #----------- Store Contours ---------------
            #Change this line 
            buff_contours.append([maskCoh])
            #buff_contours.append(contours)

            #------------ Return data origin ----------------

            #self.dataCoh    =   self.transpose_data(self.dataCoh,mode='B')

        
        return buff_contours

    def select_events(self,contours, **kwargs):

        #------------- Automatic Threshold ------------

        MIN_BINS_PER_HEIGHT     =   round(MIN_SPANHEIGHT_METEORS/dH)
        MIN_SEC_TRAIL           =   2.5

        THextent                =   ceil((MIN_SEC_TRAIL/self.tsamp)**2)
        minArea                 =   ceil(MIN_SEC_TRAIL/self.tsamp)*MIN_BINS_PER_HEIGHT


        #---------------- Get atributtes --------------

        THarea        =   kwargs.get('area',minArea)
        THextent      =   kwargs.get('extent',THextent)
        THden         =   kwargs.get('THden',30)

        heightsList   =   kwargs.get('heights')
        timestamp     =   kwargs.get('timestamp')
        pairs         =   kwargs.get('pairs')
        tsamp         =   kwargs.get('tsamp',TSAMP)
        mode_unwrap   =   '1D'
        
        
        Buff          =   []

        (z,x,y)       =   self.dataCoh.shape
        timestamp     =   numpy.arange(0,(y+1)*tsamp,tsamp)
        #------------- Transpose Data -----------------
        
        #self.dataCoh    =   self.transpose_data(self.dataCoh.T)
        #self.dataPha    =   self.transpose_data(self.dataPha.T)   

        #---------------- Process ---------------------
        #self.height=self.transpose_data(self.height)



        for n,pair in enumerate(pairs):

            image_number=0
        
            BuffPerPair     =   []
            contour         =   contours[n]


            iBinary =   deepcopy(self.dataCoh[n])
            iBinary =   numpy.where(iBinary>0,0,255)
            iBinary =   iBinary.astype(numpy.uint8)

         

           

            for nx,c in enumerate(contour): 
                #area    =   cv2.contourArea(c)

                if 1:
                #if area > THarea:
           
                        x,y,h,w =   cv2.boundingRect(c)
                    # extent  =   h*h
                    # den     =   (int(area)/(w*h))*100

                    #if extent >THextent and den>THden :
                   
                        
                        #------------- Select Events ------------
                        
                        #print(image_number," ",den,extent)
                        BCoh    =   self.dataCoh[n,y:y+w,x:x+h]


                        cBinary =   deepcopy(self.dataCoh)
                        cBinary =   numpy.where(BCoh>0,0,255)
                        cBinary =   cBinary.astype(numpy.uint8)
                        
                        
                        #-------- Add mask filter --------------

                        maskBinary  =   self.__operationOC(cBinary)
                        maskBinary  =   maskBinary  -   cBinary

                        indNan      =   numpy.where(maskBinary==255)

                        BCoh[indNan]    =   numpy.nan
                        
                        cBinary[indNan] =   255 


                        BPha    =   deepcopy(self.dataPha[n,y:y+w,x:x+h])
                        
                        BOriginPha  =   deepcopy(self.originPha[n,y:y+w,x:x+h])
      
                        BSnr    =   deepcopy(self.dataSNR[pair][:,y:y+w,x:x+h])

                        BSnr    =   BSnr.astype('float32') 
 

                        BPha    =   BPha.astype('float32')
                        BOriginPha  = BOriginPha.astype('float32')
                        BCoh    =   BCoh.astype('float32')

                        #------ Realizamos el unwrapp -------
                    
                        #BOriginPha=unwrap_phase(BOriginPha*numpy.pi/180)*180/numpy.pi

                        #-----------------------------------
                        BPha[indNan]    =   numpy.nan
                        BSnr[0][indNan]    =   numpy.nan
                        BSnr[1][indNan]    =   numpy.nan

                        if mode_unwrap=='2D':

                            iNan=numpy.where(~numpy.isnan(BPha))

                            BPha[iNan]=BOriginPha[iNan]

                        


                        hm      =   self.height[y:y+w]
                        tm      =   timestamp[x:x+h]

                        #--------- Add section COMPONENT ANALYSIS FUNCTION --------------

                        coppyBinary   =   deepcopy(cBinary)

                        coppyBinary   =   255 -   coppyBinary
                        
                        coppyBinary   =   coppyBinary.astype(numpy.uint8)
 
 
                        (tLabels, idLabel, value, centroids)=cv2.connectedComponentsWithStats(coppyBinary,
                                                                                            8,
                                                                                            cv2.CV_32S)

                        for i in range(1,tLabels):


                            zerosBlock    =   numpy.empty((coppyBinary.shape))

                            zerosBlock.fill(255)

                            zerosBlock    =   zerosBlock.astype(numpy.uint8)
                            #  Area del componente:
                            area    =   value[i,cv2.CC_STAT_AREA]

                       
                            
                            if area>=minArea:
                                 

                                indWhere=numpy.where(idLabel==i)
                                indNoWhere=numpy.where(~(idLabel==i))

                                zerosBlock[indWhere]=0

                                #zerosBlock  =   255 - zerosBlock

                                chain_base ="img/{}-ROI{}_{}/".format(self.nFile,n,image_number )
                                
                                chain=chain_base+"{}-ROI{}_{}--{}".format(self.nFile,n,image_number,i)

                                if not os.path.isdir(os.path.dirname(chain_base)):
                                    os.makedirs(os.path.dirname(chain_base))

                                if self.FlagPlot:cv2.imwrite(chain+"_bn.png", zerosBlock )

                                coppyBPha       =   deepcopy(BPha)
                                hm_             =   deepcopy(hm)
                                coppyBSnr       =   deepcopy(BSnr)
                                coppyEnterelySnr    = deepcopy(BSnr)

                                coppyBCoh       =   deepcopy(BCoh)
                                BOriginPha_     =   deepcopy(BOriginPha)

                                coppyBPha   =   coppyBPha.astype('float32')

                                #---
                                coppyBPha[indNoWhere]   =   numpy.nan
                                coppyBSnr[0][indNoWhere]   =   numpy.nan
                                coppyBSnr[1][indNoWhere]   =   numpy.nan
                                coppyBCoh[indNoWhere]   =   numpy.nan
 
                                
                                coppyBPha   =   self.transpose_data(coppyBPha,'B').T
                                BOriginPha_  =  self.transpose_data(BOriginPha_,'B').T
                                
                                coppyBSnr[0]   =   self.transpose_data(coppyBSnr[0],'B').T
                                coppyBSnr[1]   =   self.transpose_data(coppyBSnr[1],'B').T


                                coppyEnterelySnr[0]   =   self.transpose_data(coppyEnterelySnr[0],'B').T
                                coppyEnterelySnr[1]   =   self.transpose_data(coppyEnterelySnr[1],'B').T

                                coppyBCoh   =   self.transpose_data(coppyBCoh,'B').T

 

                                arx0=~numpy.all(numpy.isnan(coppyBPha), axis=0)
                                 
                                
                                coppyBPha   =   coppyBPha[:,arx0]
                                BOriginPha_  =   BOriginPha_[:,arx0]
                                coppyBSnr   =   coppyBSnr[:,:,arx0]
                                coppyBCoh   =   coppyBCoh[:,arx0]

                                coppyEnterelySnr=coppyEnterelySnr[:,:,arx0]

                                arx1=~numpy.all(numpy.isnan(coppyBPha), axis=1)



                                coppyBPha   =   coppyBPha[arx1,:]
                                BOriginPha_ = BOriginPha_[arx1,:]
                                coppyBSnr   =   coppyBSnr[:,arx1,:]
                                coppyEnterelySnr   =   coppyEnterelySnr[:,arx1,:]
                                coppyBCoh   =   coppyBCoh[arx1,:]
                                

                                
                                #Seleccionamos el valor snr más grande
                        
                                coppyEnterelySnr=numpy.mean(coppyEnterelySnr,axis=0)

                                SnrMean=numpy.mean(coppyBSnr,axis=0)
 


                                indMax=numpy.unravel_index(numpy.nanargmax(SnrMean), SnrMean.shape)

                                #_----------------------------------------------------

                                uu,hh=indMax[0],indMax[1]
                  
                                xx,yy=SnrMean.shape
                                xx=xx-1;yy=yy-1; 

                                
                                if self.FlagPlot:
                                    buffSnr=coppyEnterelySnr[uu-1:uu+2,hh-1:hh+2] 
                                    if uu==0:
                                        buffSnr[:,0]=numpy.nan
                                    elif uu==xx:
                                        buffSnr[:,-1]=numpy.nan

                                    if hh==0:
                                        buffSnr[0,:]=numpy.nan
                                    
                                    elif yy==0:
                                        buffSnr[-1,:]=numpy.nan
                                    
                                    buffSnr=buffSnr.ravel()
                                    #Normalizamos
                                    buffSnr=buffSnr/self.MAX_SNR
                                    

                                    ratio_point=numpy.count_nonzero(~numpy.isnan(coppyBSnr))/((xx+1)*(yy+1))

                                    #duration

                                    svm=numpy.array([ratio_point,xx+1,yy+1,self.MAX_SNR] ).astype(numpy.float64) 


                                


                                #_---------------------------------------------------



                                hm_=hm_[arx1]

                                if self.FlagPlot:


                                    blocks={'pha':coppyBPha,
                                            'snr':coppyBSnr,
                                            'coh':coppyBCoh,
                                            'orisnr':coppyBSnr,
                                            'oripha':BOriginPha_,
                                            'svm':svm,
                                            'buffsnr':buffSnr.astype(numpy.float64) }



                                    with h5py.File(chain+'.h5py','w') as f:

                                        for code in ['pha','snr','coh','oripha','orisnr','svm','buffsnr']:

                                            f[code]=blocks[code]


                                aux     =   [tm,hm_,coppyBPha,n,chain,BOriginPha_]

                                #numpy.save(chain+'.npy',coppyBPha)

                                BuffPerPair.append(aux)

                                 

                        #----------------------------------------------------------------

                        if self.FlagPlot: cv2.rectangle(iBinary, (x, y), (x + h, y + w), (0,255,0), 2)
       
            

                        image_number+=1

        

            if self.FlagPlot: cv2.imwrite('{}--test{}.png'.format(self.nFile,n),iBinary)

            Buff.append(BuffPerPair)
 
                        
        
        
        #------------- Return data format ------------------
        self.dataCoh    =   self.transpose_data(self.dataCoh,'B').T        
        self.dataPha    =   self.transpose_data(self.dataPha,'B').T 
        self.originPha  =   self.transpose_data(self.originPha,'B').T 

        #self.height =  self.transpose_data(self.height)         
        
        

        return Buff



                                        
    def select_heights(self,Hmin,Hmax):

        iMin,    =   numpy.where(self.ry<=Hmin)
        iMin    =   iMin[-1]

        iMax,    =   numpy.where(self.ry<=Hmax)
        iMax    =   iMax[-1]
 

        self.dataCoh    =   self.dataCoh[:,:,iMin:iMax+1]
        self.dataDop    =   self.dataDop[:,:,iMin:iMax+1]
        self.dataPha    =   self.dataPha[:,:,iMin:iMax+1]
        self.originPha  =   self.originPha[:,:,iMin:iMax+1]
        self.dataSNR    =   self.dataSNR[:,:,iMin:iMax+1]

        self.ry         =   self.ry[iMin:iMax+1]

     
        self.height     =   self.ry
 


    def get_winds(self):
        return self.output

    def get_heights(self):
        return self.ry

    def run (self,dataOut=None,tsamp=TSAMP,**kwargs):
        
          
        #ippSeconds = dataOut.ippSeconds*dataOut.nCohInt*dataOut.nAvg
        ippSeconds=1
        timeList    =   numpy.arange(0,tsamp+ippSeconds,tsamp)


        CohThresh           =   kwargs.get('CohThresh',0.7)
        DerPhaThresh        =   kwargs.get('DerPhaThresh',0.5)
        SNRThresh           =   kwargs.get('SNRThresh',8)
        mode_load           =   kwargs.get('mode_load',False)
        SNRVar              =   kwargs.get('SNRVar',False)
        Hmin                =   kwargs.get('Hmin',75)
        Hmax                =   kwargs.get('Hmax',120)
        Hdelay              =   kwargs.get('Hdelay',0)
        FlagPlot            =   kwargs.get('create_plot',False)

        Hmin+=Hdelay
        Hmax+=Hdelay



        self.tsamp  =   tsamp
        self.FlagPlot   =   FlagPlot

        #---------------- Data saved in path -------------------------
        if mode_load is True:

            path        =   kwargs.get('load_path')
            nFile       =   kwargs.get('nFile',1)
            self.nFile  =   nFile
            pairs       =   kwargs.get('pairs',[[0,1],[1,2]])

            
            
            heightsList   =   self.__load_data(path,nFile)
            self.select_heights(Hmin,Hmax)
 
            self.operation_thresh([CohThresh,DerPhaThresh,SNRThresh],pairs,SNRVar)
            contours        =   self.detect_events(pairs)

            buff            =   self.select_events(contours,pairs=pairs,timestamp=timeList
                                                    ,heights=heightsList)
            buff            =   numpy.array(buff)
            

            vObj            =   WindProfiler2()
            
            self.output=vObj.run(pairsList=pairs,
                    tsamp=tsamp,
                    buff=buff,
                    timeList=timeList,
                    heightList=self.ry,
                    FlagPlot=FlagPlot)

            

            return None


        else:
            
            dataOut.tsamp       =   tsamp
            pairs               =   dataOut.pairs
            
            dataOut.paramInterval = dataOut.nProfiles*dataOut.nCohInt*dataOut.ippSeconds
            dataOut.abscissaList=   numpy.arange(0,paramInterval+ippSeconds,ippSeconds)
            paramInterval       =   dataOut.paramInterval

            




        #heightsList         =   dataOut.heightsList



            if dataOut.FlagDetect==True:

                heightsList         =   dataOut.heightsList
                timestamp           =   dataOut.tms

                self.get_data(dataOut)
                self.operation_thresh([CohThresh,DerPhaThresh,SNRThresh],pairs)

                contours        =   self.detect_events(pairs)

                buff            =   self.select_events(contours=contours,pairs=pairs,timestamp=timestamp,heightsList=heightsList)
                buff            =   numpy.array(buff)

                dataOut.FlagDetect  = False
                dataOut.buffDetect  = buff
                dataOut.tsamp       = tsamp
                dataOut.FlagBuff    = True
 

        
            return dataOut









#########################################################################
######                                                              #####
######                                                              #####
######                                                              #####
#########################################################################
######                                                              #####
######                                                              #####
#########################################################################

#
#
#
#------------------------- COMPUTE VELOCITY -----------------------------
#
#
#
#
#
#
#
#########################################################################
######                                                              #####
######                                                              #####
######                                                              #####
#########################################################################
######                                                              #####
######                                                              #####
#########################################################################


class WindProfiler2(object):

    flagCreateNPY=True
    flagConfigPath=True
    FlagConfig=False
    __dataReady=False
    FlagDetect=False


    loc  = '/data/test/scripts/winds'

    def __calculateAzimuth1(self, rx_location, pairslist, azimuth0):

        azimuth1 = numpy.zeros(len(pairslist))
        dist = numpy.zeros(len(pairslist))

        for i in range(len(pairslist)):
            ch0 = pairslist[i][0]
            ch1 = pairslist[i][1]

            diffX = rx_location[ch0][0] - rx_location[ch1][0]
            diffY = rx_location[ch0][1] - rx_location[ch1][1]
            azimuth1[i] = numpy.arctan2(diffY,diffX)*180/numpy.pi
            dist[i] = numpy.sqrt(diffX**2 + diffY**2)

        azimuth1 -= azimuth0 
        return azimuth1, dist

    
    def _rotation_matrix(self,tetha):
        
        return numpy.array([[numpy.cos(tetha),numpy.sin(tetha)],[-1*numpy.sin(tetha),numpy.cos(tetha)]])
    



   


    def run(self, buff, **kwargs):
 
        # if dataOut.abscissaList != None:
        #     absc = dataOut.abscissaList[:-1]
       
        technique   =   kwargs.get('technique','MeteorsInterferiometry')

        

        if technique=='MeteorsInterferiometry':

            #---------------- Constants -----------------------------------
       
            

            rx_location     =   kwargs.get('rx_location',[[0,1],[1,1],[1,0]])
            azimuth         =   kwargs.get('azimuth',51.06)
            #Kilometers
            dfactor         =   kwargs.get('dfactor',150)
            heightList      =   kwargs.get('heightList')


            pairsList       =   kwargs.get('pairsList',None)
            hmin            =   kwargs.get('hmin',70)
            hmax            =   kwargs.get('hmax',120)
            tsamp           =   kwargs.get('tsamp',TSAMP)
            timeList        =   kwargs.get('timeList')
            FlagPlot        =   kwargs.get('FlagPlot',False)

            metArray        =   buff
 


            C=3E8
            freq=50E6
            lamb=C/freq
            k = 2*numpy.pi/lamb

            self.FlagPlot   =   FlagPlot

            

         
 
            output=  self.techniqueNSM_SA(  rx_location=rx_location, 
                                            pairsList=pairsList, 
                                            azimuth=azimuth, 
                                            dfactor=dfactor, 
                                            k=k,
                                            metArray=metArray,
                                            heightList=heightList,
                                            timeList=timeList)
            self.output=output
            
    
        
        return output

    def get_winds(self):

        return self.output

    def get_uR(self,value):

        if value>=3:
            return 0.50

        elif value>2.5 and value<3:
            return 0.55

        elif value>2 and value<=2.5:
            return 0.6
        
    
        else:

            return None


    def __getPhaseSlope(self,data,pairsList,**kwargs):

        MIN_POINTS_PER_TRAIL    =   10

        slopeList           =   []

        phaseDerThresh      =   kwargs.get('phaseDerTrhesh',numpy.pi/6)
        TrailThresh         =   kwargs.get('TrailThresh',2)
        tsamp               =   kwargs.get('tsamp',TSAMP)
        ippSeconds          =   kwargs.get('ippSeconds',1)

        MOD_FILTER          =   kwargs.get('MOD_FILTER','kalman')

        meteorList      =   []

        #-------------------- parameters for kalman filter ------------------

        estimated_error=50
        uncertanity=0.17
        measurement_error=1
        xi=45
  
        #----------------------------------------------------------------------------

        if self.FlagPlot:
            fign=plt.figure(figsize=(16,9),dpi=150)
            axn=fign.add_subplot(1,1,1)
        


        #------ Create model---------
        obj=KalmanLinearFiltering(estimated_error=estimated_error,
                                  uncertanity=uncertanity,
                                measurement_error=measurement_error,
                                xi=xi)

 
        
        for n,pair in enumerate(pairsList):

            
            
            
            dataPair        =   data[n]

            chain=None


            for nxr,meteor in enumerate(dataPair):

                tmet    =   meteor[0]
                hmet    =   meteor[1]
                Pha     =   meteor[2]
                p       =   meteor[3]


                if nxr>0:
                    with open(chain+'.txt','w') as f:
                        f.write(str(count_trail))
                chain   =   meteor[4]



                #En 180 grados 
                Wrap2D  =   meteor[5]

                count_trail=0
                

                Wrap2D=Wrap2D*numpy.pi/180
  
                (x,y)=Pha.shape 
                

                Pha=Pha*numpy.pi/180 

                MaskPha =   numpy.abs(Pha[:,1:]-Pha[:,:-1]) 
                phDerAux = numpy.concatenate((numpy.full((x,1), False, dtype=bool), MaskPha>30/180*numpy.pi),axis=1)

                 
                Pha[phDerAux]=numpy.nan

                if self.FlagPlot: PhasePlot((Pha.T*180/numpy.pi).reshape(1,y,x),name=chain,h=hmet-187.5)    

 
                
 
                for nr in range(x):
                    mode_unwrapp='1D'
                    flag_trail=False
                    kalman_filter=False
                    FLAG_PATIENCE=False
                    count_size_out=0
                    
                    rsq0=0
                    count_patience=0

                    if mode_unwrapp=='2D':

                        indNoNan=~numpy.isnan(Pha)
                        Pha[indNoNan]=Wrap2D[indNoNan]

                    PhaLine     =   Pha[nr,:]
                    height      =   hmet[nr]
 
 
                    if mode_unwrapp=='1D':
                        PhaLine[~numpy.isnan(PhaLine)] = numpy.unwrap(PhaLine[~numpy.isnan(PhaLine)])   #Unwrap


                    indValid= numpy.where(~numpy.isnan(PhaLine))
                    
                    indValid=indValid[0]


                    xe=numpy.arange(PhaLine.shape[0])*tsamp
                        

                    if len(indValid)>0:

                        initPos=indValid[0]

                        nMax=len(indValid)-1

                        for n,ind in enumerate(indValid):

                            
                            if n==nMax: 
                                    break


                            i1=indValid[n]
                            i2=indValid[n+1]

                            idiff=(i2-i1)*tsamp
                          
                            phDiff=numpy.abs(PhaLine[i2]-PhaLine[i1])
                            
                         
                            #Nueva logica 
                            #-----------------------------------

                            sizeTrail = round((i1-initPos+1)*tsamp,4)

                            if sizeTrail>2.999999999999999999999999999999999999999999991:
                                
                                count_patience=count_patience+1

                                #Empieza patience o ruptura 

                                #Siempre mochamos un segundo de datos
                                tr=0.5
                                if flag_trail == False or round(count_size_out,2)<tr: 
                                    
                                    nP=int(tr/TSAMP) 
                                    nQ=int(count_size_out/TSAMP)
              
                                    nP=nP-nQ
                                    nP=0
                                                           
                                else: 
                                    nP=0

                                line=PhaLine[initPos+nP:i1+1]
                                x=xe[initPos+nP:i1+1]
 
                                
                                x_f=x[~numpy.isnan(line)]
                                y_f=line[~numpy.isnan(line)] 

                                if PATIENCE<=count_patience:

                                    fitting =   stats.linregress(x_f,y_f)  
                                    rsq     =   (fitting.rvalue)**2  
                                    vel     =   fitting.slope 

                                    if rsq>rsq0:
                                        rsq0=rsq
                                    elif rsq<rsq0:
                                        if (rsq-rsq0)<EPSILON_RSQ:

                                            FLAG_PATIENCE=True
                                    

                                    count_patience=0




                            if ((idiff>=0.9) or (i2==indValid[-1]))  or phDiff >=40/180*numpy.pi or FLAG_PATIENCE==True:
                                          
                                tr=0.5
                                if sizeTrail>2.999999999999999999999999999999999999999999991:

                                    if flag_trail == False or round(count_size_out,2)<tr: 
                                        
                                        nP=int(tr/TSAMP) 
                                        nQ=int(count_size_out/TSAMP)
                
                                        nP=nP-nQ
                                        nP=0
                                                            
                                    else: 
                                        nP=0


                                                   

                                    line=PhaLine[initPos+nP :i1+1]
                                    x=xe[initPos +nP:i1+1]
 
                                
                                
                                    x_f=x[~numpy.isnan(line)]
                   
                                    y_f=line[~numpy.isnan(line)] 
                              

                                    fitting =   stats.linregress(x_f,y_f)  
                                    rsq     =   (fitting.rvalue)**2  
                                    vel     =   fitting.slope 
                                    b0      =   fitting.intercept 
 
                                    if rsq>0.50  :
                             
  
                                            if flag_trail==False:   
                                                count_trail=count_trail+1
                                                flag_trail=True
                                  
                                            
                                            y_est=b0 + vel*x_f
                      
                                            rmse=1/numpy.var(y_est-y_f)
 

                                            estAux  =   numpy.array([tmet,p,round(height -187.5,1),vel,rmse,sizeTrail])  
                                            meteorList.append(estAux)                       
                                            
                                            if self.FlagPlot:

                                                    #Fit curve 
                                                                        
                                                    axn.plot(x,b0 + vel*x, color="r", lw=4,label='fitting R:{}  '.format(round(rsq,3) ))
                                                    

                                                    axn.grid()
                                                    

                                                    axn.scatter(x_f,y_f,s=20,marker='x',label='estimated wrapped')
                                                    axn.scatter(xe,PhaLine,s=15,marker='o',label='line entire',c='g') 
                                                    axn.legend(loc='best')
                                                    fign.savefig(chain+"--{}--linefit{}.png".format(round(height-187.5,1),i2))
                                                  #  numpy.save(chain+"--{}--x_f.npy".format(round(height-187.5,1)),x_f)
                                                   # numpy.save(chain+"--{}--y_f.npy".format(round(height-187.5,1)),y_f)
                                                    axn.clear()

                                count_patience=0;FLAG_PATIENCE=False;rsq0=0;       
                                count_size_out+=sizeTrail; 
                                initPos=i2

            if chain == None:
                pass
            else:
            
                with open(chain+'.txt','w') as f:
                    f.write(str(count_trail))


        
        meteorList=numpy.array(meteorList)

        return meteorList


                

    def _velparts(self,tetha):

        return numpy.array([[numpy.cos(tetha)],[numpy.sin(tetha)]])

    def _variance(self,dim_1):
     
        sh=dim_1.shape
        emp=numpy.empty((sh))
        emp[:]=numpy.nan 
        
        
        #dim_1=numpy.sort(dim_1)
        data=list(zip(dim_1,dim_1))
 
        obj=KMeans(n_clusters=2)
        obj.fit(data)

        all_=obj.labels_.shape[0]

        ones=numpy.count_nonzero(obj.labels_)
        zeros=all_-ones
        
        if zeros>ones:
            ind=numpy.where(obj.labels_==0)
            dim1=numpy.mean(dim_1[ind])
            emp[ind]=True

        elif ones>zeros:
            ind=numpy.where(obj.labels_==1)
            dim1=numpy.mean(dim_1[ind])
            emp[ind]=True

        else:
            #Calcular desviacion estandar 
            ind0=numpy.where(obj.labels_==0)
            ind1=numpy.where(obj.labels_==1)

            ar0=dim_1[ind0]
            ar1=dim_1[ind1]

            if numpy.std(ar0)>numpy.std(ar1):

                dim1=numpy.mean(ar1)
                emp[ind0]=True

            else:
                dim1=numpy.mean(ar0)
                emp[ind1]=True

        del obj
        return dim1 ,emp
                
    def techniqueNSM_SA(self, **kwargs):
        
        metArray = kwargs['metArray']
        heightList = kwargs['heightList']
        timeList = kwargs['timeList']

        rx_location = kwargs['rx_location']
        pairsList = kwargs['pairsList']
        azimuth = kwargs['azimuth']
        dfactor = kwargs['dfactor']
        k = kwargs['k']

        azimuth1, dist = self.__calculateAzimuth1(rx_location, pairsList, azimuth)
        d = dist*dfactor
        

        #--------------------- Compute Phase Slope ------------------------------

        
        metArray1 = self.__getPhaseSlope(metArray,pairsList)
        print("sxhape",metArray1.shape)
        
        if metArray1.ndim==2:

            metArray1[:,-3] = metArray1[:,-3]*metArray1[:,2]*1000/(k*d[metArray1[:,1].astype(int)]) #angles into velocities

        else:

            #Sometime,any spaces is nothing 

            metArray1=numpy.array([numpy.nan,numpy.nan,heightList[-1],5000,numpy.nan])
            metArray1=metArray1.reshape((1,5))


        velEst = numpy.zeros((heightList.size,2))*numpy.nan

        azimuth1 = azimuth1*numpy.pi/180 

        #--------------------- Velocity separation -------------------------------
        azimuth=51.06*numpy.pi/180
        numpy.save('velocitys.npy',metArray1)
        numpy.save('heightsList.npy',heightList)
        print('ocurrencias')
        print(collections.Counter(metArray1[:,2].tolist()))
        for i in range(heightList.size):
          
            h = deepcopy(heightList[i])-187.5
            h = round(h,1)
            
 
            
            
            #print("alturas",metArray1[:,2])
            indH = numpy.where((metArray1[:,2] == h)&(numpy.abs(metArray1[:,-3]) <= 120))[0]

            metHeight = metArray1[indH,:]
            #Velocidades a una cierta altura
            velocitys = metArray1[indH,:]
            # rmses=metArray1[indH,-1]
            
            # metArray1[indH,-1]=rmses/numpy.nanmean(rmses)
            sh=velocitys.shape[0]

 
           # if  sh>= 2 or (sh==1 and velocitys[0][-1]>0.75):
            if  sh>= 2 :
                print('---------------- height:{}---------------------------'.format(h))
  
      


                # 
                buff=numpy.empty((2,sh))
                brsq=numpy.empty((sh))
       

                # arr0=[]
                # arr1=[]

                # rms0=[]
                # rms1=[]

                # for n, array in enumerate(velocitys):

                #     p=array[1]
                #     vel=array[-3]
                #     rmse=array[-2]
                  
 

                #     if p == 0 :
                #         arr0.append(vel*rmse)

                #         rms0.append(rmse)
                    
                #     else:
                #         arr1.append(vel*rmse)
                #         rms1.append(rmse)
            
                # arr0=numpy.array(arr0)
                # arr1=numpy.array(arr1)

                # rms0=numpy.array(rms0)
                # rms1=numpy.array(rms1)

                # v0=numpy.sum(arr0)/numpy.sum(rms0)
                # v1=numpy.sum(arr1)/numpy.sum(rms1)

                # buff=numpy.array([v0,v1]) 

                        
                    

                    
                 


                # emp=numpy.empty(2,2)
                # rot0=self._rotation_matrix(azimuth)
                # des0=self._velparts()

                # for nr in [0,1]:

                #     rot0=self._rotation_matrix(azimuth)
                #     des0=self._velparts(nr*numpy.pi*0.5)






                         
                    


                for n,array in enumerate(velocitys):

                    p=array[1]
                    vel=array[-3]
                    rmse=array[-2]

        
                    if p==0:
                        tetha=0
                    else:
                        tetha=numpy.pi*-0.5
                        tetha=numpy.pi*0.5


                    rot=self._rotation_matrix(azimuth)
                    des=self._velparts(tetha)
                   
                    xyvel=vel*numpy.dot(rot,des)*rmse
                  
                    
 
                    buff[:,n]=xyvel.ravel()

                    brsq[n]=rmse
                
 
                    

                # # else:


                buff=numpy.sum(buff,axis=1)/numpy.sum(brsq)
                
                
                #Media armonic 
                #buff=sh/numpy.sum(1/buff,axis=1)
                
                
                velEst[i,:]=buff 

                #Agregar condicion de discontinuidad
                
                #print("promedio",p,buff)

                



                 
         
                # pair 0 is between 0-1
                # pair 1 is between 1-2
                # (0) (1) 
                # (3) (2)

                 
                # velAux = numpy.asmatrix(metHeight[:,-3]).T    #Radial Velocities 
                # iazim = metHeight[:,1].astype(int)
                
                # azimAux = numpy.asmatrix(azimuth1[iazim]).T    #Azimuths
             
                # A = numpy.hstack((numpy.cos(azimAux),numpy.sin(azimAux)))
                
                # A = numpy.asmatrix(A)
         
                # A1 = numpy.linalg.pinv(A.transpose()*A)*A.transpose()
 
                # velHor = numpy.dot(A1,velAux)  
                # velEst[i,:] = numpy.squeeze(velHor)
                # print('vel:{} h:{}'.format(numpy.squeeze(velHor),h))

                
        #numpy.save('velMythod.npy',velEst)
 
        return velEst








#########################################################################
######                                                              #####
######                                                              #####
######                                                              #####
#########################################################################
######                                                              #####
######                                                              #####
#########################################################################










class concatenate_data(object):

    nInit=0
    nSave=0

    count_1=0
    count_2=0

    def __init__(self,**kwargs):

        self.path   =   kwargs.get('path',os.getcwd())
        #Numero de bloques a unir ---> 30*0.5min per block = 15 minutes 

        SEC_PER_BLOCK=30
        MINUTOS=15
        n=int(MINUTOS*60/SEC_PER_BLOCK)
        self.nJoin  =   kwargs.get('nJoin',n)
        self.pSave  =   kwargs.get('pSave',os.getcwd())


    def name_generator(self,nInit):

        nFinal   =   nInit+self.nJoin
        numr      =   0

        coh     =   []
        pha     =   []
        dop     =   []
        snr     =   []
        xti     =   []

        block=[]

        #----For this ---------


        for number in range(nInit,nFinal):
            rr=self.path+'/{}'.format(self.count_1)

            num =   str(number).zfill(4)

            _coh =   rr+"/coh-{}.npy".format(num)  
            _pha =   rr+"/phase-{}.npy".format(num)
            _dop =   rr+"/dop-{}.npy".format(num)
            _snr =   rr+"/snr-{}.npy".format(num)
            _xti =   rr+"/x_time-{}.npy".format(num)      
            _block=rr+"/sblock-{}.h5".format(num)      

            coh.append(_coh)
            pha.append(_pha)
            dop.append(_dop)
            snr.append(_snr)
            xti.append(_xti)   
            block.append(_block) 

            numr+=1
            self.count_2+=1

            if self.count_2>50:
                
                self.count_1+=1
                self.count_2=0

        return [coh,pha,dop,snr,xti,numr,block]

    
    
    def get_data(self,path):

        coh_,pha_,dop_,snr_,xti_,num,block_=path[0],path[1],path[2],path[3],path[4],path[5],path[6]

        flagData=False

        for nr in range(num):
 
            try:
          
                _coh=numpy.load(coh_[nr])
                _pha=numpy.load(pha_[nr])
                _dop=numpy.load(dop_[nr])
                _snr=numpy.load(snr_[nr])
                # with h5py.File(block_[nr],'r') as hf:
                #         _coh=hf['coh'][:]
                #         _pha=hf['pha'][:]
                #         _snr=hf['snr'][:]
                #         _dop=hf['dop'][:]
                # _xti=numpy.load(xti_[nr])
                # _xti=numpy.array(_xti)
                
                
            except FileNotFoundError as e:
                print(nr,"error")
                break


            else:

                if flagData is False:

                    coh=deepcopy(_coh)
                    pha=deepcopy(_pha)
                    dop=deepcopy(_dop)
                    snr=deepcopy(_snr)
                   #xti=deepcopy(_xti)

                    flagData=True

                else:

                    coh=numpy.concatenate((coh,_coh),axis=1)
                    pha=numpy.concatenate((pha,_pha),axis=1)
                    dop=numpy.concatenate((dop,_dop),axis=1)
                    snr=numpy.concatenate((snr,_snr),axis=1)
                    # xti=numpy.concatenate((xti,_xti))

        pBlock=self.pSave+'/sblock-{}.h5'.format(self.nSave)
        # pSnr=self.pSave+'/ssnr-{}.h5'.format(self.nSave)
        # pDop=self.pSave+'/sdop-{}.h5'.format(self.nSave)
        # pPha=self.pSave+'/spha-{}.h5'.format(self.nSave)
        # pXti=self.pSave+'/sxti-{}.npy'.format(self.nSave)

        # print(pCoh)
        #Rewrite to h5f file


        with h5py.File(pBlock,'w') as f:
            f.create_dataset('coh',data=coh)
            f.create_dataset('dop',data=dop)
            f.create_dataset('pha',data=pha)
            f.create_dataset('snr',data=snr)




        # numpy.save(pCoh,coh)
        # numpy.save(pSnr,snr)
        # numpy.save(pPha,pha)
        # numpy.save(pDop,dop)
        # numpy.save(pXti,xti)

        self.nSave+=1


 

        
    
    def run(self,):

        paths=self.name_generator(self.nInit)

        self.get_data(paths)

        self.nInit+=self.nJoin



# path_save='/media/armando/TOSHIBA EXT/data2022'
# path_load='/media/armando/TOSHIBA EXT/proc2022'


# obj=concatenate_data(path=path_load,
#                      pSave=path_save)

# for nnrx in range(200):

#     obj.run()






        
    









































def step(dim):
    dim,=dim
    if dim <20:
        return 2
    
    elif dim <40:

        return 4
    elif dim <60:

        return 5

    elif dim <80:

        return 6

    elif dim <100:

        return 10

    elif dim <130:
        return 12
    else:
        return 15


def generator(data):
    data=data[-1]

    x,y=data.shape

    x_range=numpy.array([xn+1 for xn in range(x)])

    y_range=numpy.array([xn+1 for xn in range(y)])

    return x_range,y_range 

fig_=plt.figure(figsize=(16,9),dpi=300)
ax_=fig_.add_subplot(1,1,1)


flag=False
 

 

