import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import keyact
import random
import time
class KeyAct(nn.Module):
    def __init__(self,corss,uncorss):
        super(KeyAct, self).__init__()
        self.corss = int(corss)
        self.uncorss = int(uncorss)
        
    def part1(self,keys,Masks,t1,t2):
        
        ########################## PART1 #################################
        #Ma = MaskI[0]; Mb = MaskI[1]
        S1=0;SI=0;S12=0;S2=0;
        vsrc=[];vdes=[];
        SameInstances=[];Instance2BG=[];Instance2False=[]
        
        
        if self.corss==1:
            
            #print('self.corss',self.corss)
        
            for ki in range(len(keys[0])):

              src = keys[0][ki]
              des = keys[1][ki]

              x1 = int(np.ceil(src[0][0])); y1 = int(np.ceil(src[0][1]));
              x2 = int(np.ceil(des[0][0])); y2 = int(np.ceil(des[0][1]));


              try:
                C1=np.sum(Masks[:,t1,y1,x1])
                C2=np.sum(Masks[:,t2,y2,x2])
              except:
                continue

              if C1>=1 and C2>=1:
              #if Ma[y1,x1]!=0 and Mb[y2,x2]!=0:
                  S12+=1;

                  C3 = np.where(Masks[:,t1,y1,x1]>0)[0];
                  C4 = np.where(Masks[:,t2,y2,x2]>0)[0];

                  if len(C3)==0 or len(C4)==0:
                    continue

                  stu =[False]
                  try:
                    stu = list(C3==C4)
                  except:
                    pass

                  #print('sut 1', stu)

                  # Same instance
                  if False not in stu:
                  #if Ma[y1,x1]==Mb[y2,x2]:
                    #print(list(C3==C4))
                    SI+=1;
                    SameInstances.append([keys[0][ki],keys[1][ki]])
                    #vsrc.append(keys[0][ki]);
                    #vdes.append(keys[1][ki]);

                  # Instance to False Instance
                  else:
                    Instance2False.append([keys[0][ki],keys[1][ki]])
                    #vsrc.append(keys[0][ki]);
                    #vdes.append(keys[1][ki]);
                    S2+=1;


              # Instance to BG
              if C1!=0 and C2==0:

                  #print(Masks[:,0,y1,x1])
                  Instance2BG.append([keys[0][ki],keys[1][ki]])
                  #vsrc.append(keys[0][ki]);
                  #vdes.append(keys[1][ki]);
    
        return SameInstances,Instance2False,Instance2BG
    
    def part2(self,keys,keypoints,transformed_keypoints,unmatched_keypoints,MaskO,H,W,Masks,t1,t2):
        

        Mao = MaskO[t1]; Mbo = MaskO[t2]
        vsrc=[];vdes=[];Self_Occlusion=[];Occlusion=[]
        Instance2False_N=[];Instance2BG_N=[];plus=0;

        if self.uncorss==1:
            
            #print('self.uncorss',self.uncorss)
            
            for uk in range(len(transformed_keypoints)):

                pointA = unmatched_keypoints[uk].pt
                pointB = transformed_keypoints[uk]

                if pointB[0]<0 or pointB[1]<0 or pointB[1]>(H-2) or pointB[0]>(W-2):
                    continue

                x1 = int(np.ceil(pointA[0])); y1 = int(np.ceil(pointA[1]));
                x2 = int(np.ceil(pointB[0])); y2 = int(np.ceil(pointB[1]));

                V1 = np.asarray(pointA)
                V2 = np.asarray(pointB)

                try:
                    C1=np.sum(Masks[:,t1,y1,x1])
                    C2=np.sum(Masks[:,t2,y2,x2])
                except:
                    continue

                if C1>=1 and C2>=1:
                #if Ma[y1,x1]!=0 and Mb[y2,x2]!=0:

                  C3 = np.where(Masks[:,t1,y1,x1]>0)[0];
                  C4 = np.where(Masks[:,t2,y2,x2]>0)[0];

                  if len(C3)==0 or len(C4)==0:
                    continue

                  stu =[False]
                  try:
                    stu = list(C3==C4)
                  except:
                    pass

                  #print('sut 2', stu)

                  if False not in stu:
                    Self_Occlusion.append([V1,V2])
                    #vsrc.append(np.asarray([[x1,y1]]));
                    #vdes.append(np.asarray([[x2,y2]]));

                  ## Occlusion
                  elif Mbo[y2,x2]>1:
                      #print(C3,C4)
                      #print(Masks[:,0,y1,x1])
                      #print(Masks[:,1,y2,x2])

                      Occlusion.append([V1,V2])
                      plus=1
                      #vsrc.append(np.asarray([[x1,y1]]));
                      #vdes.append(np.asarray([[x2,y2]]));

                  else:
                    Instance2False_N.append([V1,V2])
                    #vsrc.append(np.asarray([[x1,y1]]));
                    #vdes.append(np.asarray([[x2,y2]]));

                # Instance to BG
                if C1!=0 and C2==0:
                  Instance2BG_N.append([V1,V2])
                  #vsrc.append(np.asarray([[x1,y1]]));
                  #vdes.append(np.asarray([[x2,y2]]));



        return Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N,plus

    def normalize_tensor(self,tensor):
        min_val = tensor.min()  # Find the minimum value
        max_val = tensor.max()  # Find the maximum value
        
        if max_val - min_val != 0:
            tensor = (tensor - min_val) / (max_val - min_val)  # Normalize
        return tensor

    def losscal_2(self,Pz1,Pz2,power):
        
        return 1 / (1 + torch.mean(torch.abs(self.normalize_tensor(Pz1) - self.normalize_tensor(Pz2)) ** power))
    
    def losscal_1(self,Pz1,Pz2,power):
        
        return torch.mean(torch.abs(self.normalize_tensor(Pz1) - self.normalize_tensor(Pz2)) ** power)
    
    def get_dist_base(self,pairbase,response,H,W,Hf,Wf,pz,plus,powers):
        
        rH = Hf/H; rW = Wf/W; 
        lossbase = 0.0
        act=0;
        for pair in pairbase:
        
            if pair[2]=='BG':
                for k1,k2 in zip(pair[0],pair[1]):
                    #print(k1,k2)
                    x=int(k1['value'][0]*rW); y=int(k1['value'][1]*rH)
                    xb=int(k2['value'][0]*rW); yb=int(k2['value'][1]*rH)
                    
                    for JJ in [1,2,4]:
                        
                        Pz1 = response[:,k1['time'],y-(JJ*pz):y+(JJ*pz),x-(JJ*pz):x+(JJ*pz)]
                        Pz2 = response[:,k2['time'],yb-(JJ*pz):yb+(JJ*pz),xb-(JJ*pz):xb+(JJ*pz)]

                        if Pz1.shape!=Pz2.shape or 0 in Pz1.shape or 0 in Pz2.shape:
                            continue

                        #lossbase = lossbase + (1/torch.mean(torch.abs(self.normalize_tensor(Pz1) - self.normalize_tensor(Pz2)) ** 3))
                        lossbase = lossbase + self.losscal_2(Pz1,Pz2,powers[0])
                        #print('inverse',self.losscal_2(Pz1,Pz2,3))
                        act+=1;
                    
            if pair[2]=='I':
                 for k1,k2 in zip(pair[0],pair[1]):
                    x=int(k1['value'][0]*rW); y=int(k1['value'][1]*rH)
                    xb=int(k2['value'][0]*rW); yb=int(k2['value'][1]*rH)

                    for JJ in [1,2,4]:
                        
                        Pz1 = response[:,k1['time'],y-(JJ*pz):y+(JJ*pz),x-(JJ*pz):x+(JJ*pz)]
                        Pz2 = response[:,k2['time'],yb-(JJ*pz):yb+(JJ*pz),xb-(JJ*pz):xb+(JJ*pz)]

                        if Pz1.shape!=Pz2.shape or 0 in Pz1.shape or 0 in Pz2.shape:
                            continue

                        #if k1['time']==k2['time']:
                        lossbase = lossbase +  self.losscal_2(Pz1,Pz2,powers[1]) + (plus/2)
                        #print('direct cross',self.losscal_2(Pz1,Pz2,powers[2]))
                        #else:
                        #    lossbase = lossbase +  self.losscal_2(Pz1,Pz2,powers[2])

                        act+=1;
                    
            if pair[2]=='IS':
                 for k1,k2 in zip(pair[0],pair[1]):
                    x=int(k1['value'][0]*rW); y=int(k1['value'][1]*rH)
                    xb=int(k2['value'][0]*rW); yb=int(k2['value'][1]*rH)

                    for JJ in [1,2,4]:
                        
                        Pz1 = response[:,k1['time'],y-(JJ*pz):y+(JJ*pz),x-(JJ*pz):x+(JJ*pz)]
                        Pz2 = response[:,k2['time'],yb-(JJ*pz):yb+(JJ*pz),xb-(JJ*pz):xb+(JJ*pz)]

                        if Pz1.shape!=Pz2.shape or 0 in Pz1.shape or 0 in Pz2.shape:
                            continue


                        lossbase = lossbase +  self.losscal_1(Pz1,Pz2,powers[2]) + (plus/2)
                        #print('direct inside',self.losscal_1(Pz1,Pz2,powers[2]))

                        act+=1;
                    
            #print('lossbase',lossbase)
        if act!=0:
            return lossbase/act
        else:
            return 0.0
    
    def create_bag(self,Masks,I,keypoints,t,bag_bg,bag_instance):
      for r in keypoints:
        x1 = int(r.pt[0])
        y1 = int(r.pt[1])

        status=0;

        for i in range(I):
            
          #jet=-1
          try:
            jet = Masks[i,t,y1,x1]
          except:
            continue
            
          if jet>0:
            tmp = i;
            status+=1;
          else:
            tvalue = {'value':np.array(r.pt),'time':t}
            bag_bg.append(tvalue)

        if status==1:
          value = bag_instance[tmp]
          tvalue = {'value':np.array(r.pt),'time':t}
          bag_instance.update({tmp:value+[tvalue]})

      return bag_bg,bag_instance

    def base(self,Masks,I,keypoints1,keypoints2,keypoints3,response,H,W,Hf,Wf,plus,pz,powers):
        
        bag_bg=[];bag_instance={}
        for i in range(I):
          bag_instance.update({i:[]})
        
        
        bag_bg,bag_instance = self.create_bag(Masks,I,keypoints1,0,bag_bg,bag_instance)
        bag_bg,bag_instance = self.create_bag(Masks,I,keypoints2,1,bag_bg,bag_instance)
        bag_bg,bag_instance = self.create_bag(Masks,I,keypoints3,2,bag_bg,bag_instance)

        counts=[];tmpG=[]
        if len(bag_bg)>5:
          counts = [len(bag_bg)]

        for key in bag_instance.keys():
          if len(bag_instance[key])>5:
            counts.append(len(bag_instance[key]))
        try:
            select_random = min(np.min(counts),20)
        except:
            return 0.0
        
        pairbase=[]

        if len(bag_bg)>5:
          tmpG = random.sample(bag_bg, select_random)

          for i1 in range(0,I):
            if len(bag_instance[i1])<=5:
              continue

            tmpA = random.sample(bag_instance[i1], select_random)

            if len(tmpG)!=0:
              pairbase.append([tmpA,tmpG,'BG'])

        for i1 in range(0,I):
          if len(bag_instance[i1])<=5:
            continue

          tmpA = random.sample(bag_instance[i1], select_random)
          
          ds = len(tmpA)//2
          pairbase.append([tmpA[:ds],tmpA[ds:2*ds],'IS'])
            
          #print('tmpA',len(tmpA[:ds]),len(tmpA[ds:2*ds]),'IS')
          
          
  
          for i2 in range(i1,I):
            if i1!=i2 and len(bag_instance[i2])>5:

              tmpB = random.sample(bag_instance[i2], select_random)
              pairbase.append([tmpA,tmpB,'I'])
        
                
              
              #print(asd)
                
        return self.get_dist_base(pairbase,response,H,W,Hf,Wf,pz,plus,powers)
    
    def forward_base(self, Masks,I,keypoints1,keypoints2,keypoints3,response,H,W,Hf,Wf,plus,pz,powers):

        
        loss_base = self.base(Masks,I,keypoints1,keypoints2,keypoints3,response,H,W,Hf,Wf,plus,pz,powers)
                
        return loss_base
    
    
    def compare(self,list_pair,response,H,W,Hf,Wf,pz,t1,t2,pow_det):
        YN = [1,0,0,1,0,0,0]
        
        rH = Hf/H; rW = Wf/W; 
        lossdet = 0.0
        q=-1;
        act=0;
        for lists in list_pair:
            
            q+=1;
            
            if len(lists)==0:
                continue
                
            random.shuffle(lists)
                
            #print('len lists',q,len(lists))
            
            #print(lists)
            
            for index in range(len(lists)):
            
                k1 = lists[index][0]
                k2 = lists[index][1]
                
                #print('k1,k2',q,k1,k2,len(k1))
                
                if len(k1)==1:
                    x=int(k1[0][0]*rW); y=int(k1[0][1]*rH)
                    xb=int(k2[0][0]*rW); yb=int(k2[0][1]*rH)
                else:
                    x=int(k1[0]*rW); y=int(k1[1]*rH)
                    xb=int(k2[0]*rW); yb=int(k2[1]*rH)
                    
                for JJ in [1,2,4]:
                    
                    Pz1 = response[:,t1,y-(JJ*pz):y+(JJ*pz),x-(JJ*pz):x+(JJ*pz)]
                    Pz2 = response[:,t2,yb-(JJ*pz):yb+(JJ*pz),xb-(JJ*pz):xb+(JJ*pz)]



                    if Pz1.shape!=Pz2.shape or 0 in Pz1.shape or 0 in Pz2.shape:
                        continue

                    if YN[q]==1:
                        lossdet = lossdet + self.losscal_1(Pz1,Pz2,pow_det[q])
                    else:
                        lossdet = lossdet + self.losscal_2(Pz1,Pz2,pow_det[q])


                    act+=1;
                
                if index>=20:
                    break
                    
                #print('lossdet ',YN[q],lossdet) 
            
            #print('lossdet ',YN[q],lossdet)
            #print(Asd)
            
        if act!=0:
            return lossdet/act
        else:
            return 0.0
        
    def forward(self,Cx,Cy,Cz,response__x,response__y,response__z,Masks,MaskO,H,W,Hf,Wf,I,pz,powers):
        
        pow_det = [powers[3],powers[4],powers[5],powers[6],powers[7],powers[8],powers[9]]
                
        [keys__x,keypoints1__x,transformed_keypoints__x,keypoints2__x,unmatched_keypoints__x] = Cx
        [keys__y,keypoints2__y,transformed_keypoints__y,keypoints3__y,unmatched_keypoints__y] = Cy
        [keys__z,keypoints3__z,transformed_keypoints__z,keypoints1__z,unmatched_keypoints__z] = Cz

        #start_time = time.time()
            
        ############# Cx #################################################################################################################################
        SameInstances,Instance2False,Instance2BG = self.part1(keys__x,Masks,0,1)
        #print("--- %s key det s1 0 ---" % (time.time() - start_time))   

        #start_time = time.time()
        Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N,plus__x = self.part2(keys__x,keypoints1__x,transformed_keypoints__x,unmatched_keypoints__x,MaskO,H,W,Masks,0,1)
        #print("--- %s key det s1 1 ---" % (time.time() - start_time))   

        #start_time = time.time()
        losskey_x = self.compare([SameInstances,Instance2False,Instance2BG,Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N],response__x,H,W,Hf,Wf,pz,0,1,pow_det)  
        
        #print("--- %s key det s2 ---" % (time.time() - start_time))   
        
        ############# Cy #################################################################################################################################
        SameInstances,Instance2False,Instance2BG = self.part1(keys__y,Masks,1,2)
        Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N,plus__y = self.part2(keys__y,keypoints2__y,transformed_keypoints__y,unmatched_keypoints__y,MaskO,H,W,Masks,1,2)
        losskey_y = self.compare([SameInstances,Instance2False,Instance2BG,Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N],response__y,H,W,Hf,Wf,pz,1,2,pow_det)  
        
        
        ############# Cz #################################################################################################################################
        SameInstances,Instance2False,Instance2BG = self.part1(keys__z,Masks,2,0)
        Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N,plus__z = self.part2(keys__z,keypoints3__z,transformed_keypoints__z,unmatched_keypoints__z,MaskO,H,W,Masks,2,0)
        losskey_z = self.compare([SameInstances,Instance2False,Instance2BG,Self_Occlusion,Occlusion,Instance2False_N,Instance2BG_N],response__z,H,W,Hf,Wf,pz,2,0,pow_det)  
        
        return losskey_x,losskey_y,losskey_z,(plus__x+plus__y+plus__z)/3