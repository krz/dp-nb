#install.packages("rstan", repo="http://cran.uni-muenster.de/")
library(tidyverse)
library(rstan)

model_string <-"
data {
int<lower=1> K; // number of mixture components
int<lower=1> N; // number of data points
int<lower=1> M; //the number of columns in the model matrix
matrix[N,M] X; //the model matrix
int y[N]; // observations
real<lower=0> alpha0 ; // dirichlet prior
}
transformed data {
vector<lower=0>[K] alpha0_vec = rep_vector(alpha0, K); //symmetric dirichlet prior
}
parameters {
  positive_ordered[K] gamma; // primitives of mixing proportions
  vector[M] betas[K];
  real<lower=0> sigma[K]; // scales of mixture components
}
transformed parameters {
  vector[K] theta = gamma / sum(gamma);
}
model {
gamma ~ gamma(alpha0_vec, 1); // implies: theta ~ dirichlet(alpha_vec)
sigma ~ lognormal(0, 2);
for (k in 1:K)
betas[k,] ~ normal(0,10);
for (n in 1:N) {
real lps[K];
for (k in 1:K)
lps[k] = log(theta[k]) + neg_binomial_2_lpmf(y[n] | exp(X[n] * betas[k]), sigma[k]);
target += log_sum_exp(lps);
}
}"

set.seed(41)


# heavy overlap
betas.hvy <- list(c(0, .5, -.5),
                  c(1, 0, 0),
                  c(1.7, -.5, .5),
                  c(2, 0, 0),
                  c(2.3, .5, -.5))
sizes.hvy <- seq(1,50,l=5)

# medium overlap
betas.med <- list(c(0, .5, -.5),
                  c(1.8, 0, 0),
                  c(2.4, -.5, .5),
                  c(2.7, 0, 0),
                  c(3, .5, -.5))
sizes.med <- seq(1,100,l=5)

# low overlap
betas.low <- list(c(0, .5, -.5),
                  c(3.2, 0, 0),
                  c(3.9, -.5, .5),
                  c(4.3, 0, 0),
                  c(4.6, .5, -.5))
sizes.low <- seq(1,200,l=5)

N <- c(3,5,3,4,2)*1000

X <- list(data.frame(V=rep(1,N[1]), 
                     V1=rnorm(N[1], 0, .5),
                     V2=rnorm(N[1], 0, .5)),
          data.frame(V=rep(1,N[2]), 
                     V1=rnorm(N[2], 0, .5),
                     V2=rnorm(N[2], 0, .5)),
          data.frame(V=rep(1,N[3]), 
                     V1=rnorm(N[3], 0, .5),
                     V2=rnorm(N[3], 0, .5)),
          data.frame(V=rep(1,N[4]), 
                     V1=rnorm(N[4], 0, .5),
                     V2=rnorm(N[4], 0, .5)),
          data.frame(V=rep(1,N[5]), 
                     V1=rnorm(N[5], 0, .5),
                     V2=rnorm(N[5], 0, .5)))
mus.hvy <- mapply(function(x,y) exp(x%*% y),
                  x=lapply(X,as.matrix),y=betas.hvy)
mus.med <- mapply(function(x,y) exp(x%*% y),
                  x=lapply(X,as.matrix),y=betas.med)
mus.low <- mapply(function(x,y) exp(x%*% y),
                  x=lapply(X,as.matrix),y=betas.low)
y.hvy <- mapply(rnbinom,n=N,size=sizes.hvy,mu=mus.hvy)
y.med <- mapply(rnbinom,n=N,size=sizes.med,mu=mus.med)
y.low <- mapply(rnbinom,n=N,size=sizes.low,mu=mus.low)

df.hvy <- tbl_df(data.frame(y = unlist(y.hvy), 
                            comp = rep(1:5,N),
                            do.call("rbind",X))) %>%
  mutate(comp=factor(comp))

df.med <- tbl_df(data.frame(y = unlist(y.med), 
                            comp = rep(1:5,N),
                            do.call("rbind",X))) %>%
  mutate(comp=factor(comp))
df.low <- tbl_df(data.frame(y = unlist(y.low), 
                            comp = rep(1:5,N),
                            do.call("rbind",X))) %>%
  mutate(comp=factor(comp))



# 5 comp medium
X <- df.med %>% filter(comp%in%c(1,2,3,4,5)) %>% select(V1,V2) %>%
cbind(rep(1, nrow(filter(df.med,comp%in%c(1,2,3,4,5)))), .)
y <- df.med %>% filter(comp%in%c(1,2,3,4,5)) %>% select(y) 

m5m <- stan(model_code = model_string, data = list(X=as.matrix(X), M=ncol(X),
K=6, y = pull(y), N = nrow(X), iter=2000, warmup=1000,
                                                   alpha0=0.1), chains=3,
						   cores=3,
						   control=list(adapt_delta=0.9))
m5m


save.image(file="/home/igm/christoph.kurz/R/dpmix/simstudyc5m.Rdata")













