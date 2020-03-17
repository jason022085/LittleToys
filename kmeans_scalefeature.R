# 1.import data
library(readxl)
df_raw<-read_excel("D:\\Google 雲端硬碟實驗室\\NCTU_2018_JasonHF\\6.Thesis\\預警系統\\可分析的資料\\混成式課程特徵_一次性.xlsx")#混成式課程特徵_一次性.xlsx
df_raw = data.frame(df_raw)
df <- df_raw[-c(1,2)]#扣掉代號和姓名
df <-df[-c(5,9,11,13,17)]#扣掉FBage,FBfri,FBclose,selfaca,Major
describe(df)
#標準化
str(df)
df = scale(df)
library(psych)
df <- df[-c(4,16),]#刪掉林彥廷(沒填問卷的),再刪掉張王彥

#只選取問卷相關特徵
df <- df[,c(13:36,54,60)]
describe(df)
df = data.frame(df)
# Elbow method for kmeans(選凹折點)
require(factoextra)
fviz_nbclust(df[c(1:24)], 
             FUNcluster = kmeans,# K-Means
             method = "wss",     # total within sum of square
             k.max = 6 ,
             print.summary = T)+        # max number of clusters to consider 
  labs(title="Elbow Method for K-Means")+
  geom_vline(xintercept = 2,linetype = 2)# 畫一條垂直虛線

# 2.牛逼分群
library(NbClust)
KM_all = NbClust(data = df[c(1:24)], diss = NULL, distance = "euclidean" , min.nc = 2, max.nc = 5,
                 method = "kmeans", index = c("gap","kl","ch","cindex","db","silhouette",
                                              "duda", "pseudot2", "beale","gamma"), alphaBeale = 0.05)
KM_all$Best.nc
#                    Gap     KL     CH Cindex     DB Silhouette   Duda PseudoT2   Beale  Gamma
#Number_clusters  2.0000 2.0000 2.0000 5.0000 5.0000     2.0000 2.0000   2.0000  2.0000 2.0000
#Value_Index     -1.0546 4.0263 7.2111 0.3135 1.5498     0.2256 1.6405  -4.6853 -5.4718 0.7152

###分成2群
# 3.kmeans
set.seed(123456)
KM_OP2 <- kmeans(df[c(1:24)], centers = 2, nstart = 10) #(資料、k、迭代次數、初始配置之次數)
KM_OP2
KM_OP2$size     #Number of Cases in each Cluster
df <- cbind(df, groups2=KM_OP2$cluster)
df$groups2 <- as.factor(df$groups2)

## scatter plot

library(ggplot2)
ggplot(df, aes(Score_total, FB_total)) +
  geom_point(aes(color = groups2, shape = groups2) )
#red:low FB, low score; blue:middle-low FB, middle-low score; green:middle-high FB, high score

## scatter plot
library(factoextra)
fviz_cluster(KM_OP2,           #result of kmean 
             data = df[25:26],              # data
             ellipse.type = "norm") # shape of cluster

# 4.MANOVA
df_raw<-read_excel("D:\\Google 雲端硬碟實驗室\\NCTU_2018_JasonHF\\6.Thesis\\預警系統\\可分析的資料\\混成式課程特徵_一次性.xlsx")#混成式課程特徵_一次性.xlsx
df_raw = data.frame(df_raw)
df <- df_raw[-c(1,2)]#扣掉代號和姓名
df <-df[-c(5,9,11,13,17)]#扣掉FBage,FBfri,FBclose,selfaca,Major
df <- df[-c(4,16),]#刪掉林彥廷(沒填問卷的),再刪掉張王彥
#只選取問卷相關特徵
df <- df[,c(13:36,54,60)]
describe(df)
df = data.frame(df)
df = cbind(df,groups2=KM_OP2$cluster)
df$groups2 = as.factor(df$groups2)

## 4-1.Homogeneity of covariance matrix across groups
# Box's M Test (alpha = .001)
library(heplots)
boxM( cbind(Score_total,FB_total) ~ groups2, data=df)
#Chi-Sq (approx.) = 4.5995, df = 3, p-value = 0.2036

## 4-2.Multivariate Normality
library(MVN)
DVs = df[c(25,26)]
mvn_op <- mvn(DVs, mvnTest = c("mardia"))
mvn_op$multivariateNormality #是常態
#            Test          Statistic           p value Result
#1 Mardia Skewness   6.10850638669036 0.191190169951433    YES
#2 Mardia Kurtosis -0.861092545494189 0.389187074520886    YES

## 4-3. 1-way MANOVA
manova_data_op <- manova(cbind(Score_total,FB_total)~ groups2,data=df)
summary(manova_data_op,test = "Wilks")

library(heplots)
etasq(manova_data_op)

## 5.Post-hoc (ANOVA is not the best one, but it's not bad)

#homogeniety of variance
library(car)
leveneTest(Score_total~ groups2,data=df)
post_score	<- aov(Score_total ~ groups2 , data=df)
summary(post_score)
etasq(post_score)

leveneTest(FB_total~ groups2,data=df)
# ANOVA : reject the homogeneiality assumption, inequal-variance across groups(welch's)
oneway.test(FB_total ~ groups2, data=df, var.equal=FALSE)

#
describeBy(df,group =  df$groups2)
