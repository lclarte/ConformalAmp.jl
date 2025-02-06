ConfLasso <- function(X,Y,beta0,lambda,x0,alpha,verbose=F,shift.range=F) {
  y.range <- range(Y)
  n0 <- dim(x0)[1]
  res <- list()
  for (ii in 1:n0) {
    if (verbose) {cat(sprintf("\r%sProcessing prediction point %i (of %i) ...", 
                    "", ii, n0))
                  flush.console()
    }
    # direction of move
    x <- as.vector(x0[ii,])
    y.center <- sum(x*beta0)
    if (shift.range) {
      y.range <- y.range - mean(y.range) + y.center
    }
    if (y.center>=y.range[2] | y.center<=y.range[1]) {
      res[[ii]] <- list(pieces=y.center,y.change=c(y.center,y.center),
                        type.change=c(0,1),
                        J.list=matrix((1:p)%in%which(beta0!=0),ncol=1))
    } else {
      res.1 <- DirectionalSearch(X,Y,beta0,lambda,x,alpha,1,y.range,shift.range)
      res.2 <- DirectionalSearch(X,Y,beta0,lambda,x,alpha,-1,y.range,shift.range)
      y.change <- c(res.2$y.change,res.1$y.change)
      type.change <- c(1-res.2$type.change, res.1$type.change)
      if (type.change[1] == 1) {
        y.change <- c(y.range[1], y.change)
        type.change <- c(0, type.change)
      }
      if (tail(type.change, 1) == 0) {
        y.change <- c(y.change, y.range[2])
        type.change <- c(type.change, 1)
      }
      res[[ii]] <- list(y.change = y.change, type.change=type.change,
                        pieces = c(rev(res.2$y.list[-1]), res.1$y.list[-1]),
                        J.list = c(t(apply(res.2$J.list,1,rev)),
                                   res.1$J.list[,-1]))
    }
  }
  return(res)
}

ConfLassoSimple <- function(X,Y,beta0,lambda,x0,alpha,verbose=F,check=T,intercept=F,rg.factor=0.25,first.change.only=T) {
  n0 <- dim(x0)[1]
  n <- nrow(X)
  if (is.null(beta0)) {
    out <- glmnet(X,Y,standardize=F,intercept=intercept,lambda=lambda/n)
    beta0 <- out$beta[,1]
    if (intercept) beta0 <- c(out$a0, beta0)
  }
  if (intercept) {
    X <- cbind(rep(1,n), X)
    x0 <- cbind(rep(1,n0), x0)
  }
  res <- matrix(0,ncol=6,nrow=n0)
  y.min <- min(Y)
  y.max <- max(Y)
  y.rg <- y.max - y.min
  y.max <- y.max + y.rg * rg.factor
  y.min <- y.min - y.rg * rg.factor
  y.range <- c(y.min, y.max)
  for (ii in 1:n0) {
    if (verbose) {cat(sprintf("\r%sProcessing prediction point %i (of %i) ...", 
                    "", ii, n0))
                  flush.console()
    }
    # direction of move
    x <- as.vector(x0[ii,])
    res.1 <- DirectionalSearchSimple(X,Y,beta0,lambda,x,alpha,1,check=check,
               y.range=y.range,intercept=intercept,first.change.only=first.change.only)
    res.2 <- DirectionalSearchSimple(X,Y,beta0,lambda,x,alpha,-1,check=check,
               y.range=y.range,intercept=intercept,first.change.only=first.change.only)
    lo <- ifelse(length(res.2$y.change), max(res.2$y.change, y.min), y.min)
    up <- ifelse(length(res.1$y.change), min(res.1$y.change, y.max), y.max)
    pce <- max(0, length(res.1$y.list) + length(res.2$y.list) - 1)
    act <- mean(apply(cbind(res.1$J.list, res.2$J.list)[,-1,drop=F], 2, sum))
    n.piece <- max(length(res.1$y.change) + length(res.2$y.change) - 1, 0)
    res[ii,] <- c(lo, up, (res.1$check.res & res.2$check.res), pce, act, n.piece)
  }
  return(res)
}

Reduce <- function(conf.lasso.res) {
  n0 <- length(conf.lasso.res)
  n.change <- sapply(conf.lasso.res,function(x){length(x$y.change)})
  if (any(n.change != 2 * floor(n.change/2))) {
    return(list(res=NULL, status="error", changes=NULL))
  }
  n.cols <- max(n.change) / 2
  lo.mat <- matrix(0,n0,n.cols)
  up.mat <- matrix(0,n0,n.cols)
  n.changes <- rep(0,n0)
  J.size <- rep(0,n0)
  for (jj in 1:n0) {
    lo.mat[jj,1:n.change[jj]/2] <- conf.lasso.res[[jj]]$y.change[2*(1:(n.change[jj]/2))-1]
    up.mat[jj,1:n.change[jj]/2] <- conf.lasso.res[[jj]]$y.change[2*(1:(n.change[jj]/2))]
    n.changes[jj] <- length(conf.lasso.res[[jj]]$pieces)
  }
  return(list(interval=list(lo=lo.mat,up=up.mat), status=all(n.change == 2),
              n.changes=n.changes))
}

DirectionalSearchSimple <- function(X, Y, beta, lambda, x, alpha, direc, check = T, 
                                    y.range=NULL, intercept=T, rg.factor=0.25,
                                    first.change.only = T) {
  n <- nrow(X)
  p <- ncol(X)
  check.res <- ifelse(check, TRUE, NULL)
  S <- t(X) %*% X
  beta.list <- matrix(0, ncol=0, nrow=p)
  y.list <- vector()

  eta.list <- matrix(0, ncol=0, nrow=p)
  J.list <- matrix(FALSE, ncol=0, nrow=p)
  y.change <- vector()
  type.change <- vector()
  
  beta.t <- beta
  y <- sum(x*beta.t)
  J <- which(beta.t!=0)
  if (intercept) J <- sort(union(1,J))
  Jc <- setdiff(1:p,J)
  n.act <- length(J)
  n.inact <- p - n.act
  j <- 0
  status <- "in"

  if (is.null(y.range)) {
    y.rg <- range(Y)
    y.min <- y.rg[1] - diff(y.rg) * rg.factor
    y.max <- y.rg[2] + diff(y.rg) * rg.factor
    y.range <- c(y.min,y.max)
  }

  direc.ind <- (direc + 3) / 2 
  y.bd <- y.range[direc.ind]
  while ((y.bd - y)*direc>=0) {
    # start from t, beta, J, y
    y.hat <- sum(x * beta.t)
    y.res <- y - y.hat
    res.vec <- Y - X %*% beta.t
    # v1 <- t(X[,J]) %*% res.vec + x[J] * y.res
    v2 <- t(X[,Jc]) %*% res.vec + x[Jc] * y.res
    if (j %in% Jc) {
      v.temp <- v2[which(Jc==j)]
      v2[which(Jc==j)] <- sign(v.temp) * (lambda - 1e-6)
    }
  
    # find key quantities
    if (length(J)) {
      A <- solve(S[J,J])
      a <- t(x[J]) %*% A %*% x[J]
      s <- 1 / (1+a[1])
      eta <- direc * A %*% x[J] * s
      gamma <- direc * (x[Jc] - S[Jc,J] %*% A %*% x[J]) * s
      r.slope <- as.vector(X[,J] %*% eta)
      if (max(abs(r.slope)) > s) {
        check.res <- FALSE
      }
    } else {
      eta <- vector(mode="numeric",length=0)
      s <- 1
      gamma <- direc * x
      r.slope <- rep(0, n)
    }

    # record current state
    beta.list <- cbind(beta.list, beta.t)
    y.list <- c(y.list, y)
    eta.long <- rep(0,p)
    eta.long[J] <- eta
    eta.list <- cbind(eta.list, eta.long)
    J.list <- cbind(J.list, (1:p) %in% J)
    
    if (first.change.only & status=="out") break

    # search for change point
    t.primal <- - beta.t[J] / eta
    t.dual <- (sign(gamma) * lambda - v2) / gamma
    t.primal[t.primal<=0] <- Inf
    t.dual[t.dual<=0] <- Inf
    t.total <- rbind(t.primal, t.dual)
    j.temp <- ifelse(intercept, 1+which.min(as.vector(t.total[-1])),
                      which.min(as.vector(t.total)))
    t <- t.total[j.temp]

    # search for and record in-out change points of residual quantile
    if (status=="in") {
      search.res <- SimpleSearch(r=as.vector(res.vec),
                                  r.slope=r.slope,
                                  y.res, s, t.max=t, alpha, n, status)
      status <- search.res$status
      if (status=="out") {
        y.change <- y + direc * search.res$t.change
      }      
    }

    # update parameters
    beta.t[J] <- beta.t[J] + t * eta
    y <- y + direc * t

    # find changed coordinate
    if (j.temp <= n.act) {
      j <- J[j.temp]
      J <- setdiff(J,j)
    } else {
      j <- Jc[j.temp - n.act]
      J <- sort(c(J,j))
    }
    Jc <- setdiff(1:p, J)
    n.act <- length(J)
    n.inact <- p - n.act
  }
  return(list(y.list=y.list, y.change=y.change,
              J.list=J.list, check.res=check.res))
}



DirectionalSearch <- function(X, Y, beta, lambda, x, alpha,
                              direc, y.range=NULL, shift.range=F, simple=F) {
  S <- t(X) %*% X
  beta.list <- matrix(0, ncol=0, nrow=p)
  y.list <- vector()

  eta.list <- matrix(0, ncol=0, nrow=p)
  J.list <- matrix(FALSE, ncol=0, nrow=p)
  y.change <- vector()
  type.change <- vector()
  
  beta.t <- beta
  y <- sum(x*beta.t)
  J <- which(beta.t!=0)
  Jc <- setdiff(1:p,J)
  n.act <- length(J)
  n.inact <- p - n.act
  j <- 0
  status <- "in"
  
  if (is.null(y.range)) {
    y.range <- range(Y)
  }
  if (y >= y.range[2] | y <= y.range[1]) {
    return(NULL)
  }
  if (shift.range) {
    y.range <- y.range - sum(y.range)/2 + y
  }
  
  while (y <= y.range[2] & y >= y.range[1]) {
    # start from t, beta, J, y
    y.hat <- sum(x * beta.t)
    y.res <- y - y.hat
    res.vec <- Y - X %*% beta.t
    v1 <- t(X[,J]) %*% res.vec + x[J] * y.res
    v2 <- t(X[,Jc]) %*% res.vec + x[Jc] * y.res
    if (j %in% Jc) {
      v.temp <- v2[which(Jc==j)]
      v2[which(Jc==j)] <- sign(v.temp) * (lambda - 1e-6)
    }
  
    # find key quantities
    A <- solve(S[J,J])
    a <- t(x[J]) %*% A %*% x[J]
    s <- 1 / (1+a[1])
    eta <- A %*% x[J] * s
    gamma <- (x[Jc] - S[Jc,J] %*% A %*% x[J]) * s

    # record current state
    beta.list <- cbind(beta.list, beta.t)
    y.list <- c(y.list, y)
    eta.long <- rep(0,p)
    eta.long[J] <- eta
    eta.list <- cbind(eta.list, eta.long)
    J.list <- cbind(J.list, (1:p) %in% J)

    # search for change point
    t.temp <- direc * (lambda-v2) / gamma
    t.temp1 <- direc * (-lambda-v2) / gamma
    t.temp2 <- -direc * beta.t[J] / eta
    t.temp[t.temp<=0] <- Inf
    t.temp1[t.temp1<=0] <- Inf
    t.temp2[t.temp2<=0] <- Inf
    t.total <- rbind(t.temp, t.temp1, t.temp2)
    j.temp <- which.min(as.vector(t.total))
    t <- t.total[j.temp]

    # search for and record in-out change points of residual quantile
    search.res <- ConfIntSearch(r=as.vector(res.vec),
                                r.slope=as.vector(X[,J] %*% eta),
                                y.res, s, t.max=t, alpha, n, status, direc)
    status <- search.res$status
    if (direc == 1) {
      y.change <- c(y.change, y + direc * cumsum(search.res$t.change))
      type.change <- c(type.change, search.res$type.change)      
    } else {
      y.change <- c(rev(y + direc * cumsum(search.res$t.change)), y.change)
      type.change <- c(rev(search.res$type.change), type.change)            
    }

    # update parameters
    beta.t[J] <- beta.t[J] + direc * t * eta
    y <- y + direc * t

    # find changed coordinate
    if (j.temp <= n.inact) {
      j <- Jc[j.temp]
      new.sign <- 1
    } else if (j.temp <= 2*n.inact) {
      j <- Jc[j.temp - n.inact]
      new.sign <- -1
    } else {
      j <- J[j.temp - 2*n.inact]
      new.sign <- 0
    }

    # update active set
    if (j %in% J) {
      J <- setdiff(J, j)
      beta.t[j] <- 0
    } else {
      J <- sort(c(J,j))
    }
    Jc <- setdiff(1:p,J)
    n.act <- length(J)
    n.inact <- p - n.act
  }
  return(list(y.list=y.list, y.change=y.change, type.change=type.change, J.list=J.list))
}

ConfIntSearch <- function(r, r.slope, y.res, s, 
                          t.max, alpha, n, status, direc) {
  threshhold <- ceiling((1-alpha)*(n+1)) - 1
  if (status == "in") {
    r.abs.end <- abs(r - r.slope * direc * t.max)
    y.abs.end <- abs(y.res + direc * t.max * s)
    if (max(abs(r.slope)) <= s & sum(r.abs.end <= y.abs.end) <= threshhold) {
      return(list(t.change=vector(),type.change=vector(),status=status))
    } else {
      return(SubSearch(r, r.slope, y.res, s, t.max, threshhold, status, direc))
    }
  } else {
    if (max(abs(r.slope)) <= s) {
      return(list(t.change=vector(),type.change=vector(),status=status))
    } else {
      return(SubSearch(r, r.slope, y.res, s, t.max, threshhold, status, direc))
    }
  }
}

SimpleSearch <- function(r, r.slope, y.res, s, 
                          t.max, alpha, n, status) {
  threshhold <- ceiling((1-alpha)*(n+1)) - 1
  position.ini <- (abs(r) <= abs(y.res))
  r.abs.end <- abs(r - r.slope * t.max)
  y.abs.end <- abs(y.res) + t.max * s
  position.end <- (r.abs.end <= y.abs.end) 
  if (sum(position.end) <= threshhold) {
    return(list(t.change = NULL, status="in"))
  } else {
    ind1 <- which(position.end & (!position.ini))
    r1 <- r[ind1] * sign(r[ind1])
    rs1 <- r.slope[ind1] * sign(r[ind1])
    t.vec <- sort((r1 - abs(y.res)) / (s+rs1))
    return(list(t.change = t.vec[threshhold+1-sum(position.ini)], status="out"))
  }
}

SubSearch <- function(r, r.slope, y.res, s, t.max, threshhold, status, direc) {
  new.ind <- which(abs(r)>abs(y.res) | abs(r.slope) > s)
  r.new <- r[new.ind]
  r.slope.new <- r.slope[new.ind]
  t1 <- (r.new - abs(y.res)) / (s + direc * r.slope.new)
  s1 <- sign(as.numeric(s > -direc * r.slope.new) - 0.5)
  t2 <- (-r.new - abs(y.res)) / (s - direc * r.slope.new)
  s2 <- sign(as.numeric(s > direc * r.slope.new) - 0.5)
  t.vec <- c(t1,t2)
  s.vec <- c(s1,s2)
  tmp.ind <- which(t.vec > 0 & t.vec <= t.max)
  if (length(tmp.ind) == 0) {
    return(list(t.change=vector(), type.change=vector()))
  } else {
    t.vec <- t.vec[tmp.ind]
    s.vec <- s.vec[tmp.ind]
    sort.res <- sort(t.vec, index.return=T)
    incr <- cumsum(s.vec[sort.res$ix])
    t.vec <- t.vec[sort.res$ix]
    gap <- threshhold - sum(abs(r) <= abs(y.res))
    k <- length(incr)
    t.change <- vector()
    type.change <- vector()
    i <- 1
    while (i <= k) {
      if (status == "in") {
        if (incr[i] > gap) {
          t.change <- c(t.change, t.vec[i])
          type.change <- c(type.change, 1)
          status <- "out"
        }
      } else {
        if (incr[i] <= gap) {
          t.change <- c(t.change, t.vec[i])
          type.change <- c(type.change, 0)
          status <- "in"
        }
      }
      i <- i+1
    }
    return(list(t.change=t.change, type.change=type.change, status=status))
  }
}

ConfRank <- function(X,Y,x,y,lambda) {
  xx <- rbind(X,x)
  yy <- c(Y,y)
  out <- glmnet(xx,yy,standardize=F,intercept=F,lambda=lambda)
  r <- abs(yy-predict(out,xx))
  return(sum(r<=r[length(Y)+1]))
}