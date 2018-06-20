from scipy.sparse import dok_matrix
import numpy as np
import pickle
import math
import time, threading
from multiprocessing import Pool
import shutil, os

def GetRealData(inputFileName, outputFileName):
    '''
    将序号正确映射
    '''
    print("GetRealData...")
    s = dict()
    i = 0
    with open(inputFileName, "r", encoding="utf8") as f:
        with open(outputFileName, "w", encoding="utf8") as g:
            for line in f:
                _in = line.split()[0]
                _out = line.split()[1]
                if _in not in s:
                    s[_in] = i
                    i += 1
                g.write(str(s[_in]) + " ")
                if _out not in s:
                    s[_out] = i
                    i += 1
                g.write(str(s[_out]) + "\n")
    ff = open("trans.txt", "w", encoding="utf8")
    for i in s.keys():
        print(i, s[i], file=ff)
    ff.close()
    print("OK")


def GetOutDegree(mat, k):
    '''
    得到矩阵的出度，导出至output.txt
    '''
    print("GetOutDegree...")
    h = open("out_degree.txt", "w", encoding="utf8")
    for i in range(k):
        _sum = np.sum(mat[i])
        if _sum == 0:
            print(i, 0, file=h)
        else:
            print(i, int(_sum), file=h)
    print("OK")


def GetNodeNum(filename):
    '''
    得到节点数量, 输入WikiData.txt
    '''
    print("GetNodeNum")
    k = set()
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            _in = line.split()[0]
            _out = line.split()[1]
            k.add(_in)
            k.add(_out)
    print("OK")
    return len(k)


def InitMatrix(k, filename="RealData.txt"):
    '''
    将RealData.txt转化为稀疏矩阵
    '''
    Mat = dok_matrix((k, k), dtype=np.float32)
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            _in = line.split()[0]
            _out = line.split()[1]
            Mat[int(_in), int(_out)] = 1
    return Mat

def InitRank(filename, k):
    '''
    初始化评分为1/k, k为节点数量 
    '''
    print("InitRank...")
    with open(filename, "w", encoding="utf8") as f:
        for i in range(k):
            f.writelines(str(1 / k) + "\n")
    print("OK")

def NewPageRank(matCol, oldRank, degreeList, k, beta=0.85):
    '''
    计算一个阶段的新的pagerank
    '''
    rank = 0
    for i in range(k):
        degree = degreeList[i]
        rankk = oldRank[i]
        if degree !=0:
            rank += (matCol[i][0,0]/degree*beta+(1-beta)/k)*rankk
        else:
            rank += (1 / k * beta + (1 - beta) * (1 / k))*rankk
    return rank

def Threadd(i, rank, degreeList, k):
    '''
    更新一块内容的pagerank
    '''
    GetTime("before Threadd"+str(i))
    with open("./block/block"+str(i)+".pkl", "rb") as f:
        mat = pickle.load(f)
        for j in range(500):
            newRank = NewPageRank(mat[:,j], rank, degreeList, k)
            with open("./oldrank/newRank"+str(i)+".txt", "a", encoding="utf8") as g:
                g.write(str(i)+"*"+str(j)+" "+str(newRank)+"\n")
            print(str(i)+"*"+str(j), newRank)
    GetTime("After Threadd"+str(i))


def Rewrite(filename, l):
    '''
    l为 list, 将 l 中的内容以覆盖的形式写到文件中
    '''
    f = open(filename, "w", encoding="utf8")
    for i in range(len(l)):
        f.write(str(l[i]) + "\n")
    f.close()


def Cosine(vector1, vector2, k):
    '''
    计算余弦相似度
    '''
    vector1 = [float(i) for i in vector1]
    vector2 = [float(i) for i in vector2]
    dotProduct = 0
    for i in range(k):
        dotProduct += (vector1[(i)]) * (vector2[(i)])
    vector1 = [(i) * (i) for i in vector1]
    vector2 = [(i) * (i) for i in vector2]
    lenVec1 = math.sqrt(sum(vector1))
    lenVec2 = math.sqrt(sum(vector2))
    if lenVec1 * lenVec2 == 0:
        return 0
    result = dotProduct / (lenVec1 * lenVec2)
    return result

def Divide2Block(degreeList, mat, k, beta=0.85):
    '''
    将稀疏矩阵分块
    '''
    os.mkdir("block")
    for i in range(14):
        with open("./block/block"+str(i)+".pkl", "wb") as f:
            pickle.dump(mat[:,500*i:500*i+500], f)
    with open("./block/block14.pkl", "wb") as f:
        pickle.dump(mat[:, 7000:k], f)

def GetTime(i):
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime, i)

def GetNewRankList():
    '''
    拼接得到新的pagerank
    '''
    newRankList = []
    for i in range(15):
        newRankList += [float(i.split()[1]) for i in open("./oldrank/newRank"+str(i)+".txt")]
    return newRankList

def UpdateFolder():
    '''
    删除oldrank文件夹和里面的内容, 新建一个
    '''
    try:
        shutil.rmtree("./oldrank")
        os.mkdir("oldrank")
    except:
        print("try again")
        os.mkdir("oldrank")


def Initialize():
    GetRealData("WikiData.txt", "RealData.txt")
    k = GetNodeNum("WikiData.txt")
    InitRank("pagerank.txt", k)
    Mat = InitMatrix(k)
    with open("mat.pkl", "wb") as f:
        pickle.dump(Mat ,f)
    os.mkdir("oldrank")
    return k

def SecondStep(mat, k):
    GetOutDegree(mat, k)
    degreeList = [int(i.split()[1]) for i in open("out_degree.txt")]
    Divide2Block(degreeList, mat, k)

def GetFinalResult(trans, pagerank):
    '''
    将原来的 index 和 pagerank 拼接起来
    '''
    originalData = [int(i.split()[0]) for i in open(trans)]
    pagerank = [float(i) for i in open(pagerank)]
    ff = open("finalResult.txt", "w", encoding="utf8")
    for i in range(len(pagerank)):
        print(originalData[i], pagerank[i], file=ff)
    ff.close()



if __name__ == "__main__":
    # k = GetNodeNum("WikiData.txt")
    k = 7115


    with open("mat.pkl", "rb") as f:
        mat = pickle.load(f)


    degreeList = [int(i.split()[1]) for i in open("out_degree.txt")]

    for j in range(500):
        oldRankList = open("pagerank.txt", "r", encoding="utf8").readlines()

        GetTime("before update")

        pool_rank = np.array(oldRankList, dtype=np.float32)

        pool = Pool(14)
        for i in range(14):
            pool.apply_async(Threadd, (i,pool_rank, degreeList, k))
        pool.close()
        pool.join()

        with open("./block/block14.pkl", "rb") as f:
            pool_mat = pickle.load(f)
            for i in range(115):
                newRank = NewPageRank(pool_mat[:, i], pool_rank, degreeList, k)
                with open("./oldrank/newRank14.txt", "a", encoding="utf8") as g:
                    g.write(str(14)+"*"+str(i)+" "+str(newRank)+"\n")
                print(str(14) + "*" + str(i), newRank)

        GetTime("after update")

        newRankList = GetNewRankList()
        sim = Cosine(oldRankList, newRankList, k)
        f = open("sim.txt", "a", encoding="utf8")
        print(sim, file=f)
        f.close()
        Rewrite("pagerank.txt", newRankList)
        UpdateFolder()
        if sim >= 0.9999:
            break
    GetFinalResult("trans.txt", "pagerank.txt")