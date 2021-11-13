import numpy as np

worddic,tagdic,pretagdic,nowtagdic={},{},{},{}
endevlist=[]

def getkey(dict,value):
    return list(dict.keys())[int(value)]

def readinput(file):
    wordnum,tagnum=0,0
    pretagcount,nowtagcount=0,0
    pretag = "###"
    with open(file, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split('/')
            nowword=temp[0]
            nowtag=temp[1]
            if nowword not in worddic.keys():
                worddic[nowword]=wordnum
                wordnum+=1
            if nowtag not in tagdic.keys():
                tagdic[nowtag]=tagnum
                tagnum+=1
            if pretag not in pretagdic.keys():
                pretagdic[pretag]=pretagcount
                pretagcount+=1
            if nowtag not in nowtagdic.keys():
                nowtagdic[nowtag]=nowtagcount
                nowtagcount+=1
            pretag=nowtag


def generate_matrix(file):
    array_wordtag=np.zeros((len(worddic),len(tagdic)))
    array_tag=np.zeros((len(nowtagdic),len(pretagdic)))
    with open(file, 'r') as f:
        pretag="###"
        for line in f.readlines():
            temp = line.strip().split('/')
            nowword=temp[0]
            nowtag=temp[1]
            wordx=worddic[nowword]
            wordy=tagdic[nowtag]
            tagx=nowtagdic[nowtag]
            tagy=pretagdic[pretag]
            array_wordtag[wordx][wordy]+=1
            array_tag[tagx][tagy]+=1
            pretag=nowtag
    # calculate wordscore,tagscore

    #print(array_wordtag[2])

    array_wordscore =array_wordtag
    value1=np.sum(array_wordtag, axis=0)
    for i in range(len(worddic)):
        for j in range(len(tagdic)):
            array_wordscore[i][j]=(array_wordtag[i][j]+0.5)/(len(worddic)+value1[j])

    #print(array_wordscore)
    #value3 = np.sum(array_wordscore, axis=0)
    #print(value3)

    array_tagscore=array_tag
    value2=np.sum(array_tag,axis=0)

    for i in range(len(nowtagdic)):
        for j in range(len(pretagdic)):
            array_tagscore[i][j]=(array_tag[i][j]+0.5)/(len(nowtagdic)+value2[j])
    #print(array_tagscore)
    #value4 = np.sum(array_tagscore, axis=0)
    #print(value4)
    return array_wordscore,array_tagscore

def inference(testfile,scores):


    with open(testfile, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split('/')
            nowword=temp[0]
            endevlist.append(nowword)

    ans=[]
    wordscores, tagscores = scores

    wordnum, tagnum = wordscores.shape #wordnum,tagnum
    final_seq = None
    for batch in range(0, 1):
        pi = np.zeros((len(endevlist),tagnum))
        #pi=np.full((len(endevlist),tagnum),float("-inf"))

        tag = np.zeros((len(endevlist),tagnum))
        for j in range(0,len(endevlist)):#测试及长度
            for t in range(0,tagnum):#nowtag indice
                scores = None

                if endevlist[j] not in worddic.keys():
                    temp=1/len(tagdic)
                    for i in range(0, tagnum):  # pretag indice
                        prev = 0
                        if (j == 0):
                            prev =np.log(wordscores[0, i])

                        score = np.log(tagscores[i,t]) + np.log(temp) + pi[j - 1, i] + prev
                        scores = score if scores is None else np.vstack((scores, score))
                    pi[j, t] = np.max(scores.T)
                    tag[j - 1, t] = np.argmax(scores.T)
                else:
                    m = int(worddic[endevlist[j]])
                    temp =(wordscores[m,t])
                    for i in range(0,tagnum):#pretag indice
                        prev = 0
                        if(j==0):
                            prev = np.log(wordscores[0,i])

                        score = np.log(tagscores[i,t]) + np.log(temp) + pi[j-1,i] + prev
                        scores = score if scores is None else np.vstack((scores, score))
                    pi[j,t] = np.max(scores.T)
                    tag[j-1,t] = np.argmax(scores.T)

        tag[-1,-1] = np.argmax(pi[-1].T)
        chseq = np.zeros((wordnum))
        chosen = tag[-1,-1]
        chseq[wordnum-1] = chosen
        for k in range(1,wordnum):
            chseq[wordnum-1-k] = tag[wordnum-1-k,int(chosen)]
            chosen = int(tag[wordnum-1-k,int(chosen)])

        final_seq = chseq
        '''
        chseq = np.expand_dims(chseq,0)

        final_seq = chseq if final_seq is None else np.vstack((final_seq, chseq))
        '''
    return final_seq





#print(worddic["such"],tagdic["N"])


def train_and_test():
    readinput("entrain")
    #readinput(endev)
    a,b=generate_matrix("entrain")

    #np.savetxt('a.txt', a)
    #np.savetxt('b.txt', b)

    towrite=inference("endev",(a,b))
    #print(towrite)
    '''    
    nowwrite=[]

    for i in towrite:
        nowwrite.append(getkey(tagdic,int(i)))
    print(nowwrite)    
'''
    with open("output.txt", "w", encoding='utf-8') as f:
        for i in range(len(towrite)):
            temp=getkey(tagdic,towrite[i])

            line = endevlist[i] + "/" + temp + "\n"

            f.writelines(line)

train_and_test()

