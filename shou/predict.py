import numpy as np
import os,operator

def classify0(inX,dataSet,labels,k):
    # 获取样本个数
    dataSetSize = dataSet.shape[0]
    # 样本数据和测试数据一一对应求差值
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    # 计算差方
    sqDiffMat = diffMat**2
    # 计算没条数据对应的差方和-方差（横向求和）
    sqDistances = sqDiffMat.sum(axis=1)
    # 计算标准差（方差开方）
    distances = sqDistances**0.5
    # 获取标准差的排序下标
    sortedDistIndicies = distances.argsort()
    # 创建字典，用来存放最终结果 （k个样本中每个分类的个数）
    classCount = {}
    # 循环k值
    for i in range(k):
        # 通过排序下标产看对应的标签值
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中添加类别对应的数量信息
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # 对字典根据值来进行排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def img2vector(filename):
    # 创建容纳一个样本数据的np数组
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    # 循环所有行
    for i in range(32):
        lineStr = fr.readline()
        # 循环每一行
        for j in range(32):
            # 将文件数据替换no数组
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 获取训练数据的目录结构
    trainingFileList = os.listdir("./digits/trainingDigits")
    # 获取文件数量
    m = len(trainingFileList)
    # 创建全为0的对应数组
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 获取第i个文件名
        fileNameStr = trainingFileList[i]
        # 过滤后缀名
        fileStr = fileNameStr.split(".")[0]
        # 获取文件对应数字（标签）
        classNumStr = int(fileStr.split("_")[0])
        # 将对应标签放入标签数组中
        hwLabels.append(classNumStr)
        # 将每个转换后的样本数据更换全零数组的对应位置
        trainingMat[i, :] = img2vector("./digits/trainingDigits/%s" % (fileNameStr))


    # 将每个转换后的样本数据更换全零数组的对应位置
    vectorUnderTest = img2vector("aa.txt")
    # 获得预测分类
    classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    # 对比测试值和预测值
    print("预测值为：%d;"%(classifierResult))


handwritingClassTest()
