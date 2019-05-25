# -*- coding: UTF-8 -*-
import numpy


def generateWordCloud(content):
    """
    函数说明:生成词云
    """
    from matplotlib import pyplot
    from wordcloud import WordCloud
    # 词云参数
    wc = WordCloud(collocations=False, font_path='simfang.ttf', width=1400, height=1400, margin=2).generate(content)

    pyplot.imshow(wc)
    pyplot.axis("off")
    pyplot.show()


def createVocabList(dataSet):
    """
    函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    Parameters:
        dataSet - 整理的样本数据集
    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    # 创建一个空的不重复列表
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    Parameters:
        vocabList - createVocabList返回的列表
        inputSet - 切分的词条列表
    Returns:
        returnVec - 文档向量,词集模型
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        # 遍历每个词条
        if word in vocabList:
            # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    # 返回文档向量
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    函数说明:朴素贝叶斯分类器训练函数
    Parameters:
        trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
    Returns:
        p0Vect - 正常邮件类的条件概率数组
        p1Vect - 垃圾邮件类的条件概率数组
        pAbusive - 文档属于垃圾邮件类的概率
    """
    # 计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每篇文档的词条数
    numWords = len(trainMatrix[0])
    # 文档属于垃圾邮件类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Num = numpy.ones(numWords)
    p1Num = numpy.ones(numWords)
    # 分母初始化为2 ,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 取对数，防止下溢出
    p1Vect = numpy.log(p1Num / p1Denom)
    p0Vect = numpy.log(p0Num / p0Denom)
    # 返回属于正常邮件类的条件概率数组，属于侮辱垃圾邮件类的条件概率数组，文档属于垃圾邮件类的概率
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    函数说明:朴素贝叶斯分类器分类函数
    Parameters:
    	vec2Classify - 待分类的词条数组
    	p0Vec - 正常邮件类的条件概率数组
    	p1Vec - 垃圾邮件类的条件概率数组
    	pClass1 - 文档属于垃圾邮件的概率
    Returns:
    	0 - 属于正常邮件类
    	1 - 属于垃圾邮件类
    """
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if p1 >= p0:
        return 1, p1 / (p1 + p0)
    else:
        return 0, p0 / (p1 + p0)


def textParse(bigString):
    """
    函数说明:字符串解析函数，可处理中文和英文（分词）
    """
    import re
    import jieba
    # 切分文本
    listOfTokens = jieba.lcut(bigString)
    # 去掉标点符号
    newList = [re.sub(r'\W*', '', s) for s in listOfTokens]
    # 删除长度为0的空值
    return [tok.lower() for tok in newList if len(tok) > 0]


def readLearnFile():
    """
    函数说明:读取样本文件
    """
    docList = []
    classList = []
    fileNameList = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        fileWordList = textParse(open('email/spam/' + str(i), 'r').read())
        docList.append(fileWordList)
        fileNameList.append('spam/' + str(i) + '.txt')
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        fileWordList = textParse(open('email/ham/' + str(i), 'r').read())
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(fileWordList)
        # 标记正常邮件，0表示正常文件
        classList.append(0)
        fileNameList.append('ham/' + str(i) + '.txt')
    return docList, classList, fileNameList


def selectTestFile():
    """
    函数说明:随机选取10个文件测试
        Returns:
    	trainingSet - 训练文件索引
    	testSet - 测试文件索引
    """
    import random
    # 训练文件索引
    trainingSet = list(range(50))
    # 测试文件索引
    testSet = []
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    for i in range(10):
        # 随机选取索索引值
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del (trainingSet[randIndex])
    return trainingSet, testSet


def randFileTest():
    """
    函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
    """
    docList, classList, fileNameList = readLearnFile()
    # 创建词汇表，不重复
    vocabList = createVocabList(docList)
    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
    trainingSet, testSet = selectTestFile()
    # 创建训练集矩阵和训练集类别标签系向量
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签系向量中
        trainClasses.append(classList[docIndex])
        # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 验证分类是否错误
        isSpan, pValue = classifyNB(numpy.array(wordVector), p0V, p1V, pSpam)
        if isSpan != classList[docIndex]:
            # 错误计数加1
            errorCount += 1
            print("分类错误：", fileNameList[docIndex])
            generateWordCloud(open('email/' + fileNameList[docIndex]).read())
        else:
            print("分类正确：", fileNameList[docIndex])
    print('分类错误率：%.2f' % (float(errorCount) / len(testSet)))


def customContentTest(file):
    """
    函数说明:测试朴素贝叶斯分类器，使用给定邮件
    """
    docList, classList, fileNameList = readLearnFile()
    vocabList = createVocabList(docList)
    testWordList = textParse(open(file, 'r').read())
    trainMat = []
    for i in range(len(docList)):
        trainMat.append(setOfWords2Vec(vocabList, testWordList))
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(classList))
    isSpan, pValue = classifyNB(setOfWords2Vec(vocabList, testWordList), p0V, p1V, pSpam)
    if isSpan == 1:
        generateWordCloud(open(file, 'r').read())
    print("垃圾邮件的概率：", pValue)


if __name__ == '__main__':
    """
    分词演示
    """
    text = "是垃圾邮件的概率为,随机文件交叉验证Loading model cost 1.151 seconds.Prefix dict has been built succesfully."
    generateWordCloud(text)  # 词云展示
    wordList = textParse(text)  # 文字分割
    print(wordList)  # 控制台输出

    """
    自定义文件，验证是否为垃圾邮件
        Parameter: 文件名
    """
    # customContentTest('/email/testFile.txt')

    """
    随机文件交叉验证，查看分类错误率
    """
    # randFileTest()
