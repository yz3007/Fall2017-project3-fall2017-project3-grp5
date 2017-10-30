train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=1000,
                     distribution="multinomial",
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  
  return(list(fit=fit_gbm, iter=best_iter))
}