from numpy import array, zeros, log, ones
import feedparser
import random


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1  # new
    else:
        return 0  # SF


def trainNB0(trainMatrix, trainCategory):
    # 获取文档矩阵中文档的数目
    numTrainDocs = len(trainMatrix)
    # 获取词条向量的长度
    numWords = len(trainMatrix[0])
    # 所有文档中属于类1所占的比例p(c=1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建一个长度为词条向量等长的列表
    p0Num = ones(numWords);
    p1Num = ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0
    # 遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        # 如果该词条向量对应的标签为1
        if trainCategory[i] == 1:
            # 统计所有类别为1的词条向量中各个词条出现的次数
            p1Num += trainMatrix[i]
            # 统计类别为1的词条向量中出现的所有词条的总数
            # 即统计类1所有文档中出现单词的数目
            p1Denom += sum(trainMatrix[i])  # [1010]=2
        else:
            # 统计所有类别为0的词条向量中各个词条出现的次数
            p0Num += trainMatrix[i]
            # 统计类别为0的词条向量中出现的所有词条的总数
            # 即统计类0所有文档中出现单词的数目
            p0Denom += sum(trainMatrix[i])
    # 利用NumPy数组计算p(wi|c1)
    p1Vect = log(p1Num / p1Denom)  # 为避免下溢出问题，后面会改为log()
    # 利用NumPy数组计算p(wi|c0)
    p0Vect = log(p0Num / p0Denom)  # 为避免下溢出问题，后面会改为log()
    return p0Vect, p1Vect, pAbusive  # [0.5,.025],[0.3,0.41],0.5


def bagOfWords2VecMN(vocabList, inputSet):
    # 词袋向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 某词每出现一次，次数加1
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    # 新建一个存放词条的集合
    vocabSet = set([])
    # 遍历文档集合中的每一篇文档
    for document in dataSet:
        # 将文档列表转为集合的形式，保证每个词条的唯一性
        # 然后与vocabSet取并集，向vocabSet中添加没有出现
        # 的新的词条
        vocabSet = vocabSet | set(document)
    # 再将集合转化为列表，便于接下来的处理
    return list(vocabSet)


def calMostFreq(vocabList, fullTest):
    # 导入操作符
    import operator
    # 创建新的字典
    freqDict = {}
    # 遍历词条列表中的每一个词
    for token in vocabList:
        # 将单词/单词出现的次数作为键值对存入字典
        freqDict[token] = fullTest.count(token)
    # 按照键值value(词条出现的次数)对字典进行排序，由大到小
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的前30个单词
    return sortedFreq[:30]


def localWords(feed1, feed0):  # feed1 newyork feed0 sF
    import feedparser
    # 新建三个列表
    docList = [];  # 原始数据
    classList = [];  # 分类
    fullTest = []
    # 获取条目较少的RSS源的条目数
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 遍历每一个条目
    for i in range(minLen):
        # 解析和处理获取的相应数据
        wordList = textParse(feed1['entries'][i]['summary'])  # ['my','name','is','**']
        # 添加词条列表到docList
        docList.append(wordList)  # [['hellow','world'],['my','name']]行号
        # 添加词条元素到fullTest
        fullTest.extend(wordList)  # [hello ,world ,my ,name]列号
        # 类标签列表添加类1
        classList.append(1)  # newyork
        # 同上
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        # 此时添加类标签0
        classList.append(0)  # SF
    # 构建出现的所有词条列表
    vocabList = createVocabList(docList)
    # 找到出现的单词中频率最高的30个单词
    top30Words = calMostFreq(vocabList, fullTest)
    # 遍历每一个高频词，并将其在词条列表中移除
    # 这里移除高频词后错误率下降，如果继续移除结构上的辅助词
    # 错误率很可能会继续下降
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 下面内容与函数spaTest完全相同
    trainingSet = list(range(2 * minLen));
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # [[00110],[11000],[11001]]
        trainClasses.append(classList[docIndex])  # [1,0,0]
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 列相加为单次频率，行相加为总词数
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def gettopwords(ny, sf):
    import operator
    vocablist, p0v, p1v = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0v)):
        if p0v[i] > -5.0: topSF.append((vocablist[i], p0v[i]))
        if p1v[i] > -5.0: topNY.append((vocablist[i], p1v[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("the most related words of SF is :")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("\033[*****************************************************************]")
    print("the most related words of NY is :")
    for item in sortedNY:
        print(item[0])


ny = feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
sf = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss')
vocablist, psf, pny = localWords(ny, sf)
gettopwords(ny, sf)
