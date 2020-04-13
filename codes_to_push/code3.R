mydata = read.csv("G:/ocrug/bank-full.csv")


s = stratified(mydata, group = "y", size = 0.8, bothSets = TRUE)

val = as.data.frame(s$SAMP2)
mod = as.data.frame(s$SAMP1)

up_training <- upSample(x = mod[, -ncol(mod)],
                            y = mod$y)

write.csv(up_training, "G:/ocrug/10nov_mod.csv")
write.csv(val, "G:/ocrug/10nov_test.csv")
