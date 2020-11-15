# simulazione thesis:
library(mvtnorm) # multivariate normal distribution
library(glmnet) # package for lasso with alpha =1

set.seed(18)

##############
n = 50 #number of observations n
p = 100 # number of predictors 2n
k = 10 # number of non-zero coefficients 1/5 n
iter = 12 # number of iterations
e = rnorm(n)
lambda.grid = 10^seq(2,-5,length = 40)

X = matrix(rnorm(n*p,2,1),nrow=n,ncol=p)
dim(X) #data matrix, generated as iid normal observations

coef = matrix(c(seq(3,33,length = k),rep(0,p-k)),nrow=p) # vector of coefficients. 
dim(coef)

y= X %*% coef + e

lasso.mod = glmnet(X,y,alpha=1, lambda=lambda.grid)
q=lasso.mod$beta
dim(q)
#cbind(lasso.mod$beta,coef)
std.coeff = lasso.mod$beta * matrix(apply(X,2,sd),nrow=p,ncol=length(lambda.grid),byrow=FALSE)
dim(std.coeff)
matplot(lasso.mod$lambda,t(std.coeff),type='l',lty=1,lwd=2,col=rainbow(11),log='x',xlab='Lambda',ylab='Standradized coefficients',main='Lasso regression')


par(mfrow=c(1,1))
x.axis <- c(seq(1,100,by=1))
plot(x.axis,coef,type='p',pch=1,col=1,xlab='p predictors ',ylab='real coef vs Lasso estimates',main='Lasso vs coef')
points(x.axis, lasso.mod$beta[,20],type='p',pch=2,col=2)
legend(30,30,legend=c("real coef", "Lasso"), col=c("black","red"),cex = 1,pch=1:2)
#per 10^2 setta tutto a zero lambda grande prova colonne dopo...uno o due prove 

lasso.mod = cv.glmnet(X,y,alpha=1, lambda=lambda.grid)
plot(lasso.mod) # Plots the cross-validation curve,and upper and lower 
# standard deviation curves, as a function of the lambda values used
# Along the x axis we visualize log(lambda)
# two vertical lines correspond to lambda.1se and lambda.min

bestlam = lasso.mod$lambda.min
bestlam
log(bestlam)
abline(v=log(bestlam), lty=1,col='orange')

# compare of coefficients
coef.lasso <- predict(lasso.mod,type="coefficients",s=bestlam)
cbind(coef.lasso,c(0,coef))

#####################################################
n = 50 #number of observations n
p = 100 # number of predictors 2n
k = 10 # number of non-zero coefficients 1/5 n
iter = 12 # number of iterations

###LASSO
# to perform lasso glmnet package must set the parameter alpha=1:
time.lasso = numeric(length(iter))
mse.lasso=numeric(length(iter))
colbest =numeric(length(iter))

time.ADMM = numeric(length(iter))
mse.ADMM = numeric(length(iter))

set.seed(18)
lambda.grid = 10^seq(2,-5,length = 40)

for (i in 1:iter){
  
  X = matrix(rnorm(n*p,2,1),nrow=n,ncol=p)
  coef= matrix(c(runif(k),rep(0,p-k)), nrow=p)
  e = rnorm(n)
  
  y = X %*% coef + e
  
  start.lasso <- proc.time()
  lasso.mod = cv.glmnet(X,y,alpha=1, lambda=lambda.grid)
  std.coeff = lasso.mod$glmnet.fit$beta * matrix(apply(X,2,sd),nrow=p,ncol=length(lambda.grid),byrow=FALSE)
  
    stop.lasso <- proc.time()
  time.lasso[i] <- (stop.lasso - start.lasso)[3]#elapsed time
  names(lasso.mod$glmnet.fit)
  bestlam = lasso.mod$lambda.min
  colbest[i] = which(lasso.mod$lambda==bestlam)#lasso.mod$glmnet.fit$lambda oppure lambda.grid??
  
  opar <- par(no.readonly=TRUE)
  par(mfrow=c(1,4))
  
  plot(lasso.mod)
  abline(v=log(bestlam), lty=1,col='turquoise')

    plot(lasso.mod$lambda,lasso.mod$cvm,log='x',type='l',lwd=3,xlab=expression(paste(lambda)), ylab='CV error',main='Optimal Lambda')
  abline(v=bestlam,col='turquoise',lwd=4)
  
  matplot(lasso.mod$lambda,t(std.coeff),type='l',lty=1,lwd=2,col=rainbow(11),log='x',xlab=expression(paste(lambda)),ylab='Standradized coefficients',main='Lasso')
  abline(v=bestlam,col='turquoise',lwd=3)
  
  
  mse.lasso[i]=(sum(lasso.mod$glmnet.fit$beta[,colbest[i]] - coef))^2 #squared error

  start.ADMM<- proc.time()
  
  res = admm_lasso(X, y)$penalty(lambda.grid)$opts(maxit = 100, eps_rel = 0.001)$parallel(nthread = 4)$fit()
  
  std.coeff.ADMM = res$beta * rbind(0,matrix(apply(X,2,sd),nrow=p,ncol=length(lambda.grid),byrow=FALSE))
  
  stop.ADMM <- proc.time()
  
  time.ADMM[i] <- (stop.ADMM - start.ADMM)[3]#elapsed time
  
  matplot(res$lambda,t(std.coeff.ADMM),type='l',log='x',lty=1,lwd=2,col=rainbow(11),xlab=expression(paste(lambda)),ylab='Standradized coefficients',main='Lasso_ADMM')
  abline(v=bestlam,col='purple',lwd=3)
  (opar)
  coef = rbind(0,coef)
  
  mse.ADMM[i]=(sum(res$beta[,colbest[i]]-coef))^2 
  }
print(colbest)
print(mse.lasso)
mse.av.lasso = print(mean(mse.lasso))#vero mse fai anche sd--> riassumi in tabella o grafico al variare npk
mse.sd.lasso = print(sd(mse.lasso))

print(time.lasso)
mean.time.lasso=print(mean(time.lasso))


print(mse.ADMM)
mse.av.ADMM= print( mean(mse.ADMM))
mse.sd.ADMM = print(sd(mse.ADMM))

print(time.ADMM)
mean.time.ADMM=print(mean(time.ADMM))


#########################################################

#ADMM:Github
#questo pacchetto crea conflitto con quello che si trova su Rcran.
#prima di installarlo è necessario disinstallare il precedente.

##ho avuto problemi a installare questo pacchetto: ma il vantaggio
#è che ha memeber functions callable $penalty (dove posso fare cv su lambda) $fit e i beta $opts

#https://rdrr.io/github/yixuan/ADMM/man/admm_lasso.html
#https://rdrr.io/github/yixuan/ADMM/f/README.md

#install.packages("remotes")
#remotes::install_github("yixuan/ADMM")
#library(devtools)
#install_github("yixuan/ADMM")

#ADMM metodo iterativo differenza di glmnet
time.ADMM = numeric(length(iter))
mse.ADMM = numeric(length(iter))

for (i in 1:iter){
  
  X = matrix(rnorm(n*p,2,1),nrow=n,ncol=p)
  coef= matrix(c(runif(k),rep(0,p-k)), nrow=p)
  #dim(coef)# matrix 50 x 1
  
  y = X %*% coef + e
  
  start.ADMM<- proc.time()
  
  res = admm_lasso(X, y)$penalty(lambda.grid)$opts(maxit = 100, eps_rel = 0.001)$fit()
  #(nlambda = 20, lambda_min_ratio = 0.01)#(bestlam)(lambda.grid)
  
  # model$parallel(nthread = 2)  ## Use parallel computing just for huge datasets
  
  #res$niter#non capisco perchè 40? poi ho capito perchè lambda grid ha 40 values
  
  stop.ADMM <- proc.time()

  time.ADMM[i] <- (stop.ADMM - start.ADMM)[3]#elapsed time
  
  #plot coeff vs log(lambda) ggplot in python
  res$plot()#non risultano dal ciclo for : ho provato dopo con matplot
  dim(coef)
  dim(res$beta)
  coef = rbind(0,coef)#perchè dim(res$beta) 51 x 40
  mse.ADMM[i]=(sum(res$beta[,colbest]-coef))^2#al posto di 20 metti lambda iesimo da cv glmnet 
  
}
print(mse.ADMM)
mse.av.ADMM= print( mean(mse.ADMM))

print(time.ADMM)
mean.time.ADMM=print(mean(time.ADMM))

#come printo i grafici in admm? per lasso vanno bene accostati con par(mfrow=c(1,3))? ma poi se ho tante iterazioni, prendo primo e ultimo? 

###provo a fare grafici in admm simili a lasso per confrontarli
time.ADMM = numeric(length(iter))
mse.ADMM = numeric(length(iter))

for (i in 1:iter){
  
  X = matrix(rnorm(n*p,2,1),nrow=n,ncol=p)
  coef= matrix(c(runif(k),rep(0,p-k)), nrow=p)
  #dim(coef)# matrix 50 x 1
  
  y = X %*% coef + e
  
  start.ADMM<- proc.time()
  
  res = admm_lasso(X, y)$penalty(lambda.grid)$opts(maxit = 100, eps_rel = 0.001)$parallel(nthread = 2)$fit()
  #(nlambda = 20, lambda_min_ratio = 0.01)#(bestlam)(lambda.grid)
  std.coeff.ADMM = res$beta * rbind(0,matrix(apply(X,2,sd),nrow=p,ncol=length(lambda.grid),byrow=FALSE))
  
  # model$parallel(nthread = 2)  ## Use parallel computing just for huge datasets
  
  #res$niter#non capisco perchè 40? poi ho capito perchè lambda grid ha 40 values
  
  stop.ADMM <- proc.time()
  
  time.ADMM[i] <- (stop.ADMM - start.ADMM)[3]#elapsed time
  
  #res$plot()#plot coeff vs log(lambda) ggplot in python
  opar <- par(no.readonly=TRUE)
  par(mfrow=c(1,1))
  # plot(res)
  # abline(v=log(bestlam), lty=1,col='orange',lwd=5)
  #plot(lambda.grid,lasso.mod$cvm,log='x',type='l',lwd=3,xlab='Lambda',ylab='CV error',main='Optimal Lambda')
  #abline(v=bestlam,col='red',lwd=3)
  
  matplot(res$lambda,t(std.coeff.ADMM),type='l',log='x',lty=1,lwd=2,col=rainbow(11),xlab=expression(paste(lambda)),ylab='Standradized coefficients',main='Lasso_ADMM')
  abline(v=bestlam,col='purple',lwd=3)
  (opar)
  
  coef = rbind(0,coef)
  
  mse.ADMM[i]=(sum(res$beta[,colbest]-coef))^2 
  
}
print(mse.ADMM)
mse.av.ADMM= print( mean(mse.ADMM))

print(time.ADMM)
mean.time.ADMM=print(mean(time.ADMM))


dim(lasso.mod$glmnet.fit$beta)
dim(coef)
names(res)
print(res)
dim(res$beta)
cbind(res$lambda,lambda.grid)
##prova mfrow 
colbest=18

###provo a fare grafici in admm simili a lasso per confrontarli
time.ADMM = numeric(length(iter))
mse.ADMM = numeric(length(iter))
par(mfrow=c(3,4))
for (i in 1:iter){
  
  X = matrix(rnorm(n*p,2,1),nrow=n,ncol=p)
  coef= matrix(c(runif(k),rep(0,p-k)), nrow=p)

  y = X %*% coef + e
  
  start.ADMM<- proc.time()
  
  res = admm_lasso(X, y)$penalty(lambda.grid)$opts(maxit = 100, eps_rel = 0.001)$parallel(nthread = 2)$fit()

    std.coeff.ADMM = res$beta * rbind(0,matrix(apply(X,2,sd),nrow=p,ncol=length(lambda.grid),byrow=FALSE))
  
  #res$niter#non capisco perchè 40? poi ho capito perchè lambda grid ha 40 values
  
  stop.ADMM <- proc.time()
  
  time.ADMM[i] <- (stop.ADMM - start.ADMM)[3]#elapsed time
  
  #res$plot()#plot coeff vs log(lambda) ggplot in python
  #opar <- par(no.readonly=TRUE)
  #par(mfrow=c(1,1))
  # plot(res)
  # abline(v=log(bestlam), lty=1,col='orange',lwd=5)
  #plot(lambda.grid,lasso.mod$cvm,log='x',type='l',lwd=3,xlab='Lambda',ylab='CV error',main='Optimal Lambda')
  #abline(v=bestlam,col='red',lwd=3)
  
  matplot(res$lambda,t(std.coeff.ADMM),type='l',log='x',lty=1,lwd=2,col=rainbow(11),xlab=expression(paste(lambda)),ylab='Standradized coefficients',main='Lasso_ADMM')
  abline(v=bestlam,col='purple',lwd=3)
  #(opar)
  
  coef = rbind(0,coef)
  
  mse.ADMM[i]=(sum(res$beta[,colbest]-coef))^2 
  
}
print(mse.ADMM)
mse.av.ADMM= print( mean(mse.ADMM))

print(time.ADMM)
mean.time.ADMM=print(mean(time.ADMM))


names(res)
print(res)
dim(res$beta)
cbind(res$lambda,lambda.grid)
#quando ho molti dati e predictors come faccio a prendere grafici significativi?
#faccio grafico ad ogni iter 1:12 su asse x metto mse.medio di lasso e admm?ha senso?

#come printo i grafici in admm? prova a aumentare npk e vedi risultati
#grafici metti quelli 1 per admm(matplot- non res$plot)
#e 1 per lasso(par(mfrow=c(1,3))) significativo per ogni prova che fai variando npk
#lambda.grid lascia quella baste che il lambdaa ottimo non sia troppo a destra altrimenti è come un ols?
#metti tutto in un unico ciclo for e fai colbest[i vettore]