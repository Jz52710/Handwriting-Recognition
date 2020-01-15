from flask import Flask,render_template,request
import os,operator
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
basedir = os.path.abspath(os.path.dirname(__file__))
# C:\Users\asus\Desktop\shou
app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/getImg/',methods=['GET','POST'])
def getImg():
    # imgData = request.files.get("image")
    imgData = request.files["image"]
    path = basedir + "/static/upload/img/"

    # 图片名称
    imgName = imgData.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + imgName

    # 保存图片
    imgData.save(file_path)

    # url是图片的路径
    url = '/static/upload/img/' + imgName


    # 将图片转化为数组
    # 打开图片
    img = Image.open("static/upload/img/"+imgName).convert('RGBA')

    # 得到图片的像素值
    raw_data = img.load()

    # 将其降噪并转化为黑白两色
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][0] < 90:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][1] < 136:
                raw_data[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if raw_data[x, y][2] > 0:
                raw_data[x, y] = (255, 255, 255, 255)

    # 设置为32*32的大小
    img = img.resize((32, 32), Image.LANCZOS)

    # 进行保存，方便查看
    img.save("static/upload/txt/test.png")

    # 得到像素数组，为(32,32,4)
    array = plt.array(img)

    # 按照公式将其转为01, 公式： 0.299 * R + 0.587 * G + 0.114 * B

    gray_array = np.zeros((32, 32))

    # 行数
    for x in range(array.shape[0]):
        # 列数
        for y in range(array.shape[1]):
            # 计算灰度，若为255则白色，数值越小越接近黑色
            gary = 0.299 * array[x][y][0] + 0.587 * array[x][y][1] + 0.114 * array[x][y][2]

            # 设置一个阙值，记为0
            if gary == 255:
                gray_array[x][y] = 0
            else:
                # 否则认为是黑色，记为1
                gray_array[x][y] = 1

    # 得到对应名称的txt文件
    name01 = imgName.split('.')[0]
    name01 = name01 + '.txt'

    # 保存到文件中
    np.savetxt("static/upload/txt/"+name01, gray_array, fmt='%d', delimiter='')




    # 预测手写体

    def classify0(inX, dataSet, labels, k):
        # 获取样本个数
        dataSetSize = dataSet.shape[0]
        # 样本数据和测试数据一一对应求差值
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
        # 计算差方
        sqDiffMat = diffMat ** 2
        # 计算没条数据对应的差方和-方差（横向求和）
        sqDistances = sqDiffMat.sum(axis=1)
        # 计算标准差（方差开方）
        distances = sqDistances ** 0.5
        # 获取标准差的排序下标
        sortedDistIndicies = distances.argsort()
        # 创建字典，用来存放最终结果 （k个样本中每个分类的个数）
        classCount = {}
        # 循环k值
        for i in range(k):
            # 通过排序下标产看对应的标签值
            voteIlabel = labels[sortedDistIndicies[i]]
            # 在字典中添加类别对应的数量信息
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 对字典根据值来进行排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def img2vector(filename):
        # 创建容纳一个样本数据的np数组
        returnVect = np.zeros((1, 1024))
        fr = open(filename)
        # 循环所有行
        for i in range(32):
            lineStr = fr.readline()
            # 循环每一行
            for j in range(32):
                # 将文件数据替换no数组
                returnVect[0, 32 * i + j] = int(lineStr[j])
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
        vectorUnderTest = img2vector("static/upload/txt/"+name01)
        # 获得预测分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # 对比测试值和预测值
        print("预测值为：%d;" % (classifierResult))
        return classifierResult
    classifierResult = str(handwritingClassTest())
    return '预测结果为：'+classifierResult
app.run()