
# Definindo a pasta de trabalho
getwd()
setwd("C:/Dados/Projeto")

#################### Pacotes do R ####################

# Instalando os pacotes para o projeto (os pacotes precisam ser instalados apenas uma vez)
install.packages("Amelia", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("ggplot2", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("reshape", dependencies = TRUE)
install.packages("randomForest", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("klaR", dependencies = TRUE)

# Carregando os pacotes 
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)
library(rpart)
library(rpart.plot)
library(klaR)

# Carregando os datasets
dataset <- read.csv("credit-card.csv")

# Visualizando os dados e sua estrutura
View(dataset)
str(dataset) 
head(dataset) 

#################### Transformando e Limpando os Dados ####################

# Convertendo os atributos idade, sexo, escolaridade e estado civil para fatores (categorias)

# Idade
head(dataset$AGE) 
dataset$AGE <- cut(dataset$AGE, c(0,30,50,100), labels = c("Jovem","Adulto","Idoso"))
head(dataset$AGE) 

# Sexo
dataset$SEX <- cut(dataset$SEX, c(0,1,2), labels = c("Masculino","Feminino"))
head(dataset$SEX) 

# Escolaridade
dataset$EDUCATION <- cut(dataset$EDUCATION, c(0,1,2,3,4), 
                         labels = c("Pos Graduado","Graduado","Ensino Medio","Outros"))
head(dataset$EDUCATION) 

# Estado Civil
dataset$MARRIAGE <- cut(dataset$MARRIAGE, c(-1,0,1,2,3),
                        labels = c("Desconhecido","Casado","Solteiro","Outros"))
head(dataset$MARRIAGE) 

# Convertendo a variavel que indica pagamentos para o tipo fator
dataset$PAY_0 <- as.factor(dataset$PAY_0)
dataset$PAY_2 <- as.factor(dataset$PAY_2)
dataset$PAY_3 <- as.factor(dataset$PAY_3)
dataset$PAY_4 <- as.factor(dataset$PAY_4)
dataset$PAY_5 <- as.factor(dataset$PAY_5)
dataset$PAY_6 <- as.factor(dataset$PAY_6)

#dataset$BILL_AMT1 <- as.factor(dataset$BILL_AMT1)
#dataset$BILL_AMT2 <- as.factor(dataset$BILL_AMT2)
#dataset$LIMIT_BAL <- as.factor(dataset$LIMIT_BAL)

# Alterando a variavel dependente para o tipo fator
dataset$default.payment.next.month <- as.factor(dataset$default.payment.next.month)
head(dataset)
str(dataset)

# Renomeando a coluna de classe
colnames(dataset)
colnames(dataset)[25] <- "inadimplente"
colnames(dataset)

# Verificando valores missing e removendo do dataset
sapply(dataset, function(x) sum(is.na(x)))
missmap(dataset, main = "Valores Missing Observados")
dataset <- na.omit(dataset)

# Removendo a primeira coluna ID
dataset$ID <- NULL

# Arquivo com os dados processados
write.csv(dataset, "credit-card-process.csv")

# Total de inadimplentes versus nao-inadimplentes
t <- table(dataset$inadimplente)
str(t)

ggplot(dataset, aes(x=dataset$SEX)) + geom_bar() + labs(title="Frequency bar chart") 

head(dataset$PAY_0)
max(dataset$PAY_0)

# Plot da distribuicao usando ggplot
qplot(inadimplente, data = dataset, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set the seed
set.seed(12345)

# Amostragem estratificada. Selecione as linhas de acordo  
# com a variable inadimplente como strata
TrainingDataIndex <- createDataPartition(dataset$inadimplente, 
                                         p = 0.45, list = FALSE)

# Criar Dados de Treinamento como subconjunto do conjunto de dados  
# com numeros de indice de linha conforme identificado acima e todas as colunas
trainData <- dataset[TrainingDataIndex,]

# Tudo o que nao esta no dataset de treinamento esta no dataset de 
# teste. Observe o sinal - (menos)
testData <- dataset[-TrainingDataIndex,]

# Veja porcentagens entre as classes
prop.table(table(trainData$inadimplente))

# Numero de linhas no dataset de treinamento
nrow(trainData)

# Compara as porcentagens entre as classes de treinamento e dados originais
DistributionCompare <- cbind(prop.table(table(trainData$inadimplente)), prop.table(table(dataset$inadimplente)))
colnames(DistributionCompare) <- c("Treinamento", "Original")
DistributionCompare

# Melt Data - Converte colunas em linhas
meltedDComp <- melt(DistributionCompare)
meltedDComp

# Plot para ver a distribuicao do treinamento vs original - eh representativo ou existe sobre / sob amostragem?
ggplot(meltedDComp, aes(x = X1, y = value)) + geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

write.csv(testData, "testData.csv")

# Usaremos uma validacao cruzada de 10 folds 
# para treinar e avaliar modelo
TrainingParameters <- trainControl(method = "cv", number = 10)

################### Random Forest ###################

# Construindo o Modelo
rf_model <- randomForest(inadimplente ~ ., data = trainData)

# Salvando o modelo
saveRDS(rf_model, file = "rf_model.rds")

# Importancia das variaveis preditoras para as previsoes
varImpPlot(rf_model)

# Conferindo o erro do modelo
plot(rf_model, ylim = c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)

# Obtendo as variaveis mais importantes
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Criando o rank de variaveis baseado na importancia
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importancia relativa das variaveis
ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Previsoes
predictionrf <- predict(rf_model, testData)

# Confusion Matrix
cmrf <- confusionMatrix(predictionrf, testData$inadimplente, positive = "1")
cmrf

# Plotando a Matriz de Confusao
Matriz_Confusao_RF <- cmrf$table
plot(Matriz_Confusao_RF)

# Carregando o modelo
modelo <- readRDS("rf_model.rds")

# Calculando Precision, Recall e F1-Score, que sao metricas de avaliacao do modelo preditivo
y <- testData$inadimplente
predictions <- predictionrf

precision <- posPredValue(predictions, y)
precision

recall <- sensitivity(predictions, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#arvore = rpart(inadimplente ~  ., data = trainData,  cp = .02)

################### Arvore ###################

# Construindo o Modelo
arvore = rpart(inadimplente ~  PAY_0 + BILL_AMT1 + BILL_AMT2 + LIMIT_BAL, 
               data = trainData,  cp = .02)
arvore
summary(arvore)

# Salvando o modelo
saveRDS(arvore, file = "arvore_model.rds")

# Previsao
predictionarvore = predict(arvore, newdata = testData)

# Modelo
rpart.plot(arvore)

# Carregando o modelo
modeloArvore <- readRDS("arvore_model.rds")

# Verificando o resultado da predição
test = cbind(testData, predictionarvore)

# Renomeando a coluna de classe
colnames(test)[25] <- "nao"
colnames(test)[26] <- "sim"

# Verificando o resultado da previsao
test['result'] = ifelse(test$sim >= 0.5, '1', '0')

# Convertendo a class e o resultado da predicao para fator
test$result <- as.factor(test$result)
test$inadimplente <- as.factor(test$inadimplente)

# Confusion Matrix
cmrfArvore <- confusionMatrix(test$inadimplente, test$result)

# Plotando a matriz de confusao
Matriz_Confusao_Arvore <- cmrfArvore$table 
plot(Matriz_Confusao_Arvore)

################### Arvore ###################

# Construindo o Modelo
arvore1 = rpart(inadimplente ~  ., data = trainData,  cp = .02)
arvore1
summary(arvore1)

# Previsao
predictionarvore1 = predict(arvore1, newdata = testData)

# Modelo
rpart.plot(arvore1)

# Verificando o resultado da predição
test1 = cbind(testData, predictionarvore1)

# Renomeando a coluna de classe
colnames(test1)[25] <- "nao"
colnames(test1)[26] <- "sim"

# Verificando o resultado da previsao
test1['result'] = ifelse(test1$sim >= 0.5, '1', '0')

# Convertendo a class e o resultado da predicao para fator
test1$result <- as.factor(test1$result)
test1$inadimplente <- as.factor(test1$inadimplente)

# Confusion Matrix
cmrfArvore1 <- confusionMatrix(test1$inadimplente, test1$result)

# Plotando a matriz de confusao
Matriz_Confusao1 <- cmrfArvore1$table 
plot(Matriz_Confusao1)

#modeloNaiveBayes = NaiveBayes(inadimplente ~ ., trainData)

################### Naive Bayes ###################

# Criando o Modelo
modeloNaiveBayes = NaiveBayes(inadimplente ~ PAY_0 + BILL_AMT1 + 
                                BILL_AMT2 + LIMIT_BAL, trainData)
# Salvando o modelo
saveRDS(modeloNaiveBayes, file = "naivebayes_model.rds")

# Fazendo as predicoes
predicaoNaivesBayes = predict(modeloNaiveBayes, testData)

# Confusion Matrix
cmrfNaive <- confusionMatrix(testData$inadimplente, predicaoNaivesBayes$class)

# Plotando o modelo
plot(modeloNaiveBayes$tables$PAY_0)
plot(modeloNaiveBayes$tables$BILL_AMT2, modeloNaiveBayes$tables$BILL_AMT1)
plot(modeloNaiveBayes$tables$BILL_AMT2)
plot(modeloNaiveBayes$tables$LIMIT_BAL)

Matriz_Confusao_NaiveBayes <- cmrfNaive$table
plot(Matriz_Confusao_NaiveBayes)

# Carregando o modelo
modeloNaiveBayes <- readRDS("naivebayes_model.rds")


