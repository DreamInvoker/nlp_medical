import torch as tc
import torch.nn as nn
from torch.optim import Adam
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F
import math,json,os,random,tqdm
from preprocess1 import text_process,process_sym_attr,getnooverlap_entity
device = tc.device("cuda:3" if tc.cuda.is_available() else "cpu")
#tc.set_printoptions(threshold=10000)
#device="cpu"
hidden_size=256
batch_size=1
lam=0.8

# enttype:
# 0:None
# 1:bod
# 2:dis
# 3:sym
# 4:ite

class dismodel(nn.Module):

    def __init__(self,prob=0.2):
        super(dismodel,self).__init__()
        config = BertConfig('./bert')
        self.enc = BertModel.from_pretrained("./bert")
        self.space = nn.Linear(768+2*32,256)
        self.posl = nn.Embedding(300+5,32)
        self.posr = nn.Embedding(300+5,32)
        self.entemb = nn.Embedding(5,hidden_size)

        #self.qw,self.kw,self.vw=nn.Linear(hidden_size,hidden_size),nn.Linear(hidden_size,hidden_size),nn.Linear(hidden_size,hidden_size)

        #self.qew,self.kew,self.vew=nn.Linear(hidden_size,hidden_size),nn.Linear(hidden_size,hidden_size),nn.Linear(hidden_size,hidden_size)
        self.scorew1 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.scorew2 = nn.Parameter(tc.Tensor(2*hidden_size))
        self.scorec = nn.Parameter(tc.Tensor(batch_size))


        self.scorew2.data.uniform_(-1/math.sqrt(hidden_size),1/math.sqrt(hidden_size))
        self.scorec.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        self.out=nn.Linear(hidden_size,hidden_size)
        self.piececnn1 = nn.Sequential(
            nn.Conv1d(hidden_size,hidden_size,kernel_size=3,padding=1),
            nn.ReLU()
        )

        self.piececnn2 = nn.Sequential(
            nn.Conv1d(hidden_size,hidden_size,kernel_size=5,padding=2),
            nn.ReLU()

        )
        '''
        self.piececnn3 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=4),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU()
        )
        '''
        self.dh=8
        self.dk=hidden_size//self.dh
        self.xw=nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(3)])
        self.exw=nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(3)])
        self.gru = nn.GRU(hidden_size, hidden_size, 2, dropout=prob, bidirectional=True)
        self.gate=nn.Linear(2*hidden_size,hidden_size)
        self.criterion=nn.BCELoss()


    def forward(self,**params):
        xinp=params["input_id"].unsqueeze(0)
        att_mask = xinp.gt(0)
        #print(xinp,att_mask)
        x=self.enc(input_ids=xinp,attention_mask=att_mask)
        

        seginfo=params["ent_seg"]
        txtlen=params["length"]
        
        se=[]
        if seginfo[0][0]!=0:
            se.append([0,seginfo[0][0]-1,0])
        for k in range(len(seginfo)-1):
            se.append(seginfo[k])
            if seginfo[k+1][0]-1 >= seginfo[k][1]+1:
                se.append([seginfo[k][1]+1,seginfo[k+1][0]-1,0])
        se.append(seginfo[len(seginfo)-1])
        if seginfo[-1][1]!=txtlen-1:
            se.append([seginfo[-1][1]+1,txtlen-1,0])

        left_entpos,right_entpos,ent_type=[],[],[]
        
        for lp,rp,enttype in se:
            ent_type.append(enttype)
            if enttype!=0:
                left_entpos += [0]*(rp-lp+1)
                right_entpos += [0]*(rp-lp+1)

            else:
                px = [i for i in range(rp-lp+1)]
                py = px[::-1]
                left_entpos += px
                right_entpos += py
        ent_type = tc.LongTensor(ent_type).unsqueeze(0).to(device)
        
        left_entpos = tc.LongTensor(left_entpos).unsqueeze(0)
        right_entpos = tc.LongTensor(right_entpos).unsqueeze(0)
        self.posl,self.posr=self.posl.cpu(),self.posr.cpu()        
        #print(left_entpos.size())
        left_emb=self.posl(left_entpos)
        right_emb=self.posr(right_entpos)
        #print(left_emb.size(),right_emb.size())
        left_emb,right_emb=left_emb.to(device),right_emb.to(device)
        #bertout=self.enc(xinp)
        #berthid=self.space(x[0])
        berthid=x[0][:,:txtlen,:]
        #print(left_emb.size(),right_emb.size())
        try:
            cnnxin=tc.cat((berthid,left_emb,right_emb),-1)
        except RuntimeError:
            return 0.0,0.0,0.0,0.0,0.0
        cnnxin=self.space(cnnxin).cpu()
        self.piececnn1,self.piececnn2=self.piececnn1.cpu(),self.piececnn2.cpu()
        
        pieces=[]
    
        for lp,rp,_ in se:
            cnnpiecex=cnnxin[:,lp:rp+1,:]
            cnnpiecex=cnnpiecex.transpose(-1,-2)
            cnnpiecex=self.piececnn1(cnnpiecex)
            cnnpiecex=self.piececnn2(cnnpiecex)

            cnnpiecex=F.max_pool1d(cnnpiecex,cnnpiecex.size(2))

            pieces.append(cnnpiecex)

        cnnout=tc.cat(pieces,-1)               #b*d*L

        cnnout=cnnout.transpose(-1,-2)       # b*L*d
        #cnnout=tc.randn(1,cnnout.size(1),256).to(device)
        '''     
        gruin=cnnout.transpose(0,1)           #L*b*d
        input_len=tc.LongTensor([gruin.size(0)])
        
        packed = nn.utils.rnn.pack_padded_sequence(gruin, input_len, enforce_sorted=False)
        hidden=None


        gruout,_=self.gru(packed,hidden)
        gruout,_ = nn.utils.rnn.pad_packed_sequence(gruout)
        gruout=gruout[:,:,:hidden_size]+gruout[:,:,hidden_size:]   #L*b*d
        gruout=gruout.transpose(0,1).cpu()                 # b*L*d
        '''
        
        
        self.xw,self.exw=self.xw.cpu(),self.exw.cpu() 
        
        ex=self.entemb(ent_type).cpu()

        qkx=[cnnout,cnnout,cnnout]
        ekx=[ex,ex,ex]
       
        #qx,kx,vx=self.qw(qx),self.kw(kx),self.vw(vx)
        #qex,kex,vex=self.qew(qex),self.kew(kex),self.vew(vex)
        qx,kx,vx=[l(x).view(1,-1,self.dh,self.dk).transpose(1,2)
                                        for (l,x) in zip(self.xw,qkx)]
        
        qex,kex,vex=[l(x).view(1,-1,self.dh,self.dk).transpose(1,2)
                                                        for (l,x) in zip(self.exw,ekx)]
        #print(qx.matmul(kx.transpose(-1,-2))) 
        attnx=F.softmax(qx.matmul(kx.transpose(-1,-2)),-1)
        attne=F.softmax(qex.matmul(kex.transpose(-1,-2)),-1)
        
        attn=lam*attnx+(1-lam)*attne
    
            
        gruin=attn.matmul(vx)
        gruin=gruin.transpose(1,2).contiguous().view(1,-1,hidden_size)
        gruin=gruin.to(device)
        gruin=self.out(gruin)
        
        gp=self.gate(tc.cat((gruin,cnnout.to(device)),-1))
        #print(gp)
        one=tc.ones_like(gp).to(device)
        gruin=gp*gruin.to(device)+(one-gp)*cnnout.to(device)

        gruin=gruin.transpose(0,1)           #L*b*d
        input_len=tc.LongTensor([gruin.size(0)])
        packed = nn.utils.rnn.pack_padded_sequence(gruin, input_len, enforce_sorted=False)
        hidden=None

        gruout,_=self.gru(packed,hidden)
        gruout,_ = nn.utils.rnn.pad_packed_sequence(gruout)
        gruout=gruout[:,:,:hidden_size]+gruout[:,:,hidden_size:]   #L*b*d
        modely=gruout.transpose(0,1)


        '''
        modely=gruout
        '''
        #print(modely.size())
        
        attr=params["attr"]
        start,end=attr["self"][0],attr["self"][1]
        ansx,ansy=None,None

        try:
            ansx,ansy=attr["disease"][0],attr["disease"][1]
        except IndexError:
            pass


        for k in range(len(se)):
            if se[k][0]==start and se[k][1]==end:
                break
        symx=k
        print("symx:",symx)
        symy=modely[:,symx,:]        #b*d
        loss=0.0
        alldis,corr=0.0,0.0
        pos,rightpos=0.0,0.0
        
        for k in range(len(se)):
            if se[k][2]==2:
                
                cand=modely[:,k,:]

                quad=tc.sum((self.scorew1(cand))*symy,-1)
                lind=tc.sum(tc.cat((cand,symy),-1)*self.scorew2,-1)
                scorep=tc.sigmoid(quad+lind+self.scorec)
                
                print("scorep:",scorep)
                if len(attr["disease"])==0:
                    y=tc.FloatTensor([0])

                else:
                    print(se[k][0],se[k][1])
                    print(attr["disease"][0],attr["disease"][1])
                    ok=False
                    for j in range(len(attr["disease"])//2):
                        if attr["disease"][j]==se[k][0] or attr["disease"][j+1]==se[k][1]:
                            ok=True
                            pos += 1
                            break
                    if ok:
                        print("ok")
                        y=tc.FloatTensor([1])
                    else:
                        y=tc.FloatTensor([0])
                y=y.to(device)
                if y.item()==0:
                    rx=random.random()
                    if rx<0.7:
                        continue
                infer=tc.ge(scorep,tc.FloatTensor([0.5]).to(device)).float()
                alldis+=1
                corr += infer.eq(y).long().item()
                if y.item()==1 and infer.eq(y).long().item()==1:
                    rightpos += 1

                lossp=self.criterion(scorep,y)
                loss += lossp
       
        return loss,alldis,corr,pos,rightpos
        
        

istrain=False
txt="train.txt" if istrain else "test.txt"
tokenizer=BertTokenizer.from_pretrained("bert/vocab.txt")
DisModel=dismodel()
opt=Adam(DisModel.parameters(),lr=0.0005)
if os.path.exists("model1.pkl"):
    model=tc.load("model1.pkl")
    opt.load_state_dict(model["opt"])
    DisModel.load_state_dict(model["dm"])
DisModel=DisModel.to(device)


if __name__=="__main__":

    traindata=[]
    f=open(txt,encoding="utf-8",mode="r")
    fjs=json.load(f)
    dis,textdis,txtlen=[],[],[]
    texts=[dic for dic in fjs]
    for text in texts:
        raw_text=text_process(text["text"])
        ent_seg,valid=getnooverlap_entity(text["text"])
        if valid==False:
            continue
        txtlen.append(len(raw_text))
        symset=text["symptom"]
        
        
        num=0
        for s,e,t in ent_seg:
            if t==2:
                num+=1
        textdis.append(num)
        for syn in symset:
            attv=symset[syn]
            syninfo=process_sym_attr(raw_text,attv)
            dis.append(len(syninfo["disease"])//2)
            traindata.append((raw_text,ent_seg,syninfo))
    print("data_size:",len(traindata))
    print("average disease number per sympton",sum(dis)/len(dis))
    print("average disease number per text",sum(textdis)/len(textdis))
    print("average length of raw text",sum(txtlen)/len(txtlen))
    
    print("10% length, 90% length",txtlen[len(txtlen)//10],txtlen[len(txtlen)*9//10])
    p=0
    random.shuffle(traindata)
    
    opt.zero_grad()
    loss=0.0
    S,T=0.0,0.0
    ST,TP=0.0,0.0
    for epoch in range(len(traindata)//32):
        loss , item = 0.0, 0.0
        appdis,corrdis=0.0, 0.0
        apos,arightpos=0.0,0.0
        opt.zero_grad()
        for k in tqdm.tqdm(range(p,min(p+32,len(traindata)))):

            raw_text,ent_seg,syninfo=traindata[k]
            if len(syninfo["self"]) == 0 :
                 continue
            
            inp_token=list(raw_text)
            
            #print(leni(inp_token))

            inputid=tokenizer.convert_tokens_to_ids(inp_token)
            
            if len(inputid)>300:            
                continue
            else:
                t = inputid
                t = t + [0]*(300-len(t))
            
            
            
            inputid0=tc.LongTensor(t).to(device)            
            loss_sample,alldis,corr,pos,rightpos=DisModel(input_id=inputid0,ent_seg=ent_seg,attr=syninfo,length=len(inputid))

            
           
            loss , item = loss+loss_sample, item+1
            appdis , corrdis = appdis+alldis , corrdis+corr
            apos , arightpos = apos+pos, arightpos+rightpos
            S,T= S+appdis,T+corrdis
            ST,TP=ST+apos, TP+arightpos
        print("apearance disease",appdis)
        print("correct disease",corrdis)
        print("all pos",apos)
        print("correct pos",arightpos)
        
        
        if istrain:
            if appdis > 0:
                print("the %d th iter loss: %.4f"%(epoch,loss.item()/appdis))
                loss.backward()
                item = 0.0
                
                _ = nn.utils.clip_grad_norm_(DisModel.parameters(), 5)
                
                opt.step(device=device)
                
                
                if p%320 == 0:
                    para={
                    "opt":opt.state_dict(),
                    "dm":DisModel.state_dict()
                    }
                    tc.save(para,"model1.pkl")
                   
                
        else:
           print(corrdis,appdis)
           print("recall: %.4f"%(corrdis/appdis))
        
        p += 32
        if p>=len(traindata):
            p = 0

def testmetrices():
    print("precious %.4f"%(T/S))
    print("recall %.4f"%(TP/ST))
    
if not istrain:
    testmetrices()
       
    
    





































