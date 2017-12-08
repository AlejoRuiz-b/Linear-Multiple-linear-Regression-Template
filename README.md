# Linear-Multiple-linear-Regression-Template
This repository is created with the intention of providing a template with the steps to follow when faced with a Simple Linear Regression or a Multiple Linear Regression problem. The template is in R &amp; Python
 1.Collect data and prepare data
	R : dataset = read.csv('iqsize.csv', sep=";")
	Py: dataset = pd.read_csv('50_Startups.csv')
	    X = dataset.iloc[:, :-1].values
	    y = dataset.iloc[:, 4].values

2.Observe if there are categorical data, If so, encode categorical data
	R
		dataset$State = factor(dataset$State,
                         	levels = c('New York', 'California', 'Florida'),
                         	labels = c(1, 2, 3))
	Python
		from sklearn.preprocessing import LabelEncoder, OneHotEncoder
		labelencoder_X = LabelEncoder()
		X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
		onehotencoder = OneHotEncoder(categorical_features = [3])
		X = onehotencoder.fit_transform(X).toarray()

3.If the data is huge -> Splitting the dataset into the training set and test set
	R
		set.seed(123)
		split = sample.split(dataset$Profit, SplitRatio = 0.8)
		training_set = subset(dataset, split == TRUE)
		test_set = subset(dataset, split == FALSE)
	
	Python
		from sklearn.cross_validation import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


4.Fitting Multiple Linear Regression to the Trainin set
   #the Dependent variable expressed as a linear combination of the independent varibale
	R:
		regressor =lm(formula = Profit ~ R.D.Spend + Administration + Marketing.send, state, 
              		data = training_set)
		#Metod # 2
		regressor =lm(formula = Profit ~ .,
              		      data = training_set)
		summary(regressor)
	Python:
		from sklearn.linear_model import LinearRegression
		regressor = LinearRegression()
		regressor.fit(X_train, y_train)
		Summary(regressor)
   # Regression Equeation = PIQ = 111.4 + 2.060 Brain - 2.73 Height + 0.001 Weight


4.b Analyse the P-values if is greater (i.e 0.5 )has a lower independent relation

5. Predicting the Test set results
	R:	
		y_pred = predict(regressor, newdata = test_set)
	Python:
		y_pred = regressor.predict(X_test) # profit predicted by our model

6. If the P values are greather than 0.05 is necessary to use backwardElimination
   using the Dataset ( no the training set)
    #Building the optimal model using backward Elimination
	R:
		regressor =lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
              		      data = dataset)
		summary(regressor)
	Python:
		import statsmodels.formula.api as sm
		X = np.append(arr =np.ones((50,1)).astype(int), values = X, axis = 1)
		X_opt = X[:, [0,1,2,3,4,5]]                           #Step1 Backward Elimination eliminar numeros dependiendo de donde se encuentren
		regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step2 BE
		regressor_OLS.summary()

7. #Predicting the Test set results
	R     : y_pred = predict(regressor, newdata = test_set)
	Python: y_pred = regressor.predict(X_test) # profit predicted by our model

8. # Plot the results
   # take care when you are visualizing the Data, If you didnt split the data into Trains
   # and Test set, in that case you need to use Dataset instead of Training or test set 
	Linear Regression 
	#Visualising the Linear Regression Results

	install.packages('ggplot2')
	library (ggplot2)
	ggplot() + 
  	geom_point(aes(x = dataset$Level, y = dataset$Salary),
             	colour ='red') +
  	geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            	colour = 'blue') +
  		ggtitle('Truth or Bluff (Linear Regression)')+
  		xlab('Level') +
  		ylab('Salary')

9. # Basic Scatterplot Matrix
pairs(~PIQ+Brain+Height+Weight,data=dataset, 
      main="Simple Scatterplot Matrix")
