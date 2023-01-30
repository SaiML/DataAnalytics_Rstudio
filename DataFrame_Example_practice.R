# Creating a dataframe
# Example: RPI Weather dataframe.

days <- c('Mon', 'Tue','Wed','Thur','Fri','Sat','Sun') #  days
temp <- c(28,30.5,32,31.2,29.3,27.9,26.4) # Temperature in F' during the winter :)
snowed <- c('T','T','F','F','T','T','T') # Snowed on that day: T = TRUE, F= FALSE
snowed
class(snowed)
help("data.frame")

emptyDataframe <- data.frame()
emptyDataframe

RPI_Weather_Week <- data.frame(days,temp,snowed) # creating the dataframe using the data.frame() function

RPI_Weather_Week 
head(RPI_Weather_Week) # head of the data frame, NOTE: it will show only 6 rows, usually head() function shows the 
# first 6 rows of the dataframe, here we have only 6 rows in our dataframe. 

str(RPI_Weather_Week) # we can take a look at the structure of the dataframe using the str() function.

summary(RPI_Weather_Week) # summary of the dataframe using the summary() function
summary((RPI_Weather_Week$temp))

RPI_Weather_Week[1,] # showing the 1st row and all the columns
RPI_Weather_Week[,1] # showing the 1st coulmn and all the rows

RPI_Weather_Week[,'snowed']
RPI_Weather_Week[,'days']
RPI_Weather_Week[,'temp']
RPI_Weather_Week[1:5,c("days","temp")]
RPI_Weather_Week$temp

# finding a subsets in a datafare
RPI_Weather_Week
subset(RPI_Weather_Week,subset=snowed=="T")

# Ordering a column in DataFrame

print("sort the data in decreasing order based on subjects ")
print(RPI_Weather_Week[order(RPI_Weather_Week$temp, decreasing = TRUE), ]   )

# Creating Dataframes
# creating an empty dataframe
empty.DataFrame <- data.frame()


v1 <- 1:10
v1
letters
v2 <- letters[1:10]
v2
df <- data.frame(x= v1,y = v2)
df
# importing data and exporting data
# writing to a CSV file:
write.csv(df,file = 'saved_df1_Section2.csv',row.names = FALSE)
df_section2 <- read.csv('saved_df1_Section2.csv')
df_section2

nrow(df)
ncol(df)
colnames(df)
rownames(df)
str(df)
summary(df)

df
# Referencing Cells
df[[5,2]]
df[[5,'y']]
df[[3,'x']]
df[[4,'y']]

df
df[[3,'x']] <- 3000
df 

df[[3,'y']] <- 'ce'
df

df[2,] # referencing rows
head(mtcars)
mtcars$mpg
mtcars[,'mpg']
mtcars[['mpg']]
mtcars['mpg']
mtcars[1]
mtcars[c('mpg','cyl')]
head(mtcars[c('mpg','cyl')])
# Adding rows to the Data Frames
df2 <-data.frame(x= 2500, y  = 'new')
df2
dfNew <- rbind(df,df2)
dfNew

persons.df <- data.frame(FirstName = 'Thilanka', LastName ='Munasinghe', ID = 123456, Instituete = 'RPI')
persons.df

# Adding new Columns to the Data Frame.
df$newcol <- 3*df$x
df
# making a copy of a column and adding that to the data frame
df$copyCol2 <- df$y
df
# Also you can do this way as well:
df[,'copy2_Col3'] <- df$y
df
# Find out the Column names
colnames(df)
# Rename the coulumns
# renaming the 4th column
colnames(df)[4] <- 'LETTERS'
df
# Not selecting a row in a data frame
# Not selecting the 3rd row in this data frame
df[-3,]
# Not selecting the 4th column in this data frame
df[,-4]
# Conditional Collection:
head(mtcars)
# selecting the mpg > 20
mtcars$mpg >20
mtcars[mtcars$mpg >20,]
# selecting the mpg > 20 and cyl == 4
mtcars[mtcars$mpg >20 & mtcars$cyl == 4 ,]
# it is better to use the () in your logical statement, therefore you can do the above with () 
mtcars[(mtcars$mpg >20) & (mtcars$cyl ==4) , ]
# if you want to select specific columns, 
mtcars[(mtcars$mpg >20) & (mtcars$cyl == 4) , c('mpg','cyl','hp')]
mtcars[(mtcars$mpg >20) & (mtcars$cyl == 4) , c('cyl','hp')]
mtcars[(mtcars$mpg >20) & (mtcars$cyl == 4) , c('mpg')]
mtcars[(mtcars$mpg >20) & (mtcars$cyl == 4) , c('hp')]
mtcars[, c(1,2,3)]

# using the subset() function 
subset(mtcars, mpg > 20 & cyl == 4)

# Checking for missing data
is.na(mtcars)
any(is.na(mtcars))
any(is.na(mtcars$mpg))

df[is.na(df)] <- 0
df

mtcars$mpg[is.na(mtcars$mpg)] <- mean(mtcars$mpg)

Age <- c(27,25,26,28)
Weight <- c(155,175,130,155)
Sex <- c('M','M','F','O')
Names <- c('Mike','Jason','Silvia','Dell')
people <- data.frame(Names, Age = Age, Weight = Weight, Sex = Sex)
people
people2 <- data.frame(Names, Age, Weight, Sex)
people2

df <- data.frame(row.names = Names, Age, Weight,Sex)
df
is.na(df)
is.data.frame(df)
# names <- data.frame(Age,Weight,Sex)
# names

mat <- matrix(1:25,nrow = 5)
mat
mat2 <- as.data.frame(mat)
mat2
is.data.frame(mat2)

mat3 = data.frame(mat)
mat3
is.data.frame(mat3)

df <- mtcars
head(df)
Avg.mpg <- mean(df$mpg)
Avg.mpg
df[df$cyl==6,]
df[df$cyl==6, c('am','gear','carb')]
df[, c('am','gear','carb')]
df$performance <- df$hp/df$wt
head(df)
help("round") # Read the help function for round() 
# round the performance column values to 2 decimal points.
df$performance <- round(df$hp/df$wt,2)
head(df)

m1 <- df[df$hp >100 & df$wt >2.5,]
m1
mean(m1$mpg)

mean(df[df$hp >100 & df$wt > 2.5,c('mpg')])

mean((df[df$hp >100 & df$wt > 2.5 ,])$mpg)

df['Hornet Sportabout','mpg']

# Lists 
vec <- c(6,7,8) # vector
vec
mat <- matrix(1:30, nrow = 5) # matrix
mat
class(vec)
class(mat)
# if you want to include different data structures in to one single variable, you can use the list()
# list() function allow us to combine different datastructure into a single variable.
my.list <- list(vec,mat,df)
my.list
# instead of having automatically numbered, we can name the item in the list as follows:
my.named.list <- list(sampleVec = vec, SampleMatrix = mat, SampleDataFrame = df)
my.named.list

# List is more like an organizational tool, you can organize various dataframes 
# One advantage is, you can call the items in the list using the $ sign to call them as you call them like coulums.
my.named.list$sampleVec
my.named.list$SampleMatrix
my.named.list$SampleDataFrame

# Data Input/Output in R
# Read and Write CSV file
# Write CSV 
write.csv(mtcars, file = 'vehicles.csv')
# Read CSV
# if the CSV is in your working-directory in RStudio, then you can directly call it as follows ( otherwise you need to give the path to that file)
Vehicles <- read.csv('vehicles.csv') 
head(Vehicles)
tail(Vehicles)
class(Vehicles) 
# Excell Files input in RStudio
# In order to read the Excel file in R, you need to install the "readxl" package.
install.packages('readxl')
library(readxl)

excel_sheets('C:/Users/91983/OneDrive/Desktop/RPI-work/DA/2010EPI_data.xls')
help("excel_sheets")
#my_data <- read_excel(file.choose())
#my_data
#my_data$`2010 Environmental Performance Index (EPI)`

# Writing xlsx files
# first install the package 'xlsx'
#install.packages('xlsx')
#library(xlsx)
write.xlsx(mtcars,'Vehicles.xlsx')





# Train test split
set.seed(1)
train <- sample(1:nrow(X_scaled_matrix ),nrow(X_scaled_matrix)/1.5)
test <- (-train)



x.test <- X_scaled_matrix [test,]
x.train <- X_scaled_matrix [train,]
y.test <- Y[test]
y.train <- Y[train]
# ```

my.mse <- function(pred,act){
  mse <- mean((pred-act)^2)
  return(mse)
}


# Ridge Regression
# Fit some models and save MSE:
MSE <- c(NA)
grid <- 10^seq(10,-2,length=100)
ridge.mse <- glmnet(x.train,y.train,alpha=0,lambda=grid,thresh=1e-12)

# Compute the MSE of each model
for(i in 1:length(grid)){
  ridge.pred.tmp <- predict(ridge.mse,s=grid[i],newx <- x.test)
  MSE[i] <- my.mse(ridge.pred.tmp,y.test)
}
plot(MSE)

lambda.star <- grid[which.min(MSE)]
sprintf("Optimal value of lambda is %.1f",lambda.star)

# Lasso Regression

MSE.lasso <- c(NA)
grid <- 10^seq(10,-2,length=100)
lasso.mse <- glmnet(x.train,y.train,alpha=1,lambda=grid,thresh=1e-12)

# Compute the MSE of each model
for(i in 1:length(grid)){
  lasso.pred.tmp <- predict(lasso.mse,s=grid[i],newx <- x.test)
  MSE.lasso[i] <- my.mse(lasso.pred.tmp,y.test)
}
plot(MSE.lasso)

lambda.star.lasso <- grid[which.min(MSE.lasso)]
sprintf("Optimal value of lambda is %.1f",lambda.star.lasso)


# evaluation metric

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  mse <- mean((predicted-true)^2)
  
  
  # Model performance metrics
  data.frame(
    MSE = mse,
    Rsquare = R_square
  )
  
}

#Ridge Ression
# Prediction and evaluation on train data
predictions_train <- predict(ridge.mse, s = lambda.star, newx = x.train)
eval_results(y.train, predictions_train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge.mse, s = lambda.star, newx =  x.test)
eval_results(y.test, predictions_test)


#Lasso Ression

# Prediction and evaluation on train data
predictions_train <- predict(lasso.mse, s = lambda.star.lasso, newx = x.train)
eval_results(y.train, predictions_train)

# Prediction and evaluation on test data
predictions_test <- predict(lasso.mse , s = lambda.star.lasso, newx =  x.test)
eval_results(y.test, predictions_test)




