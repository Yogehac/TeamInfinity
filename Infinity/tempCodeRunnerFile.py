X = dataset.iloc[:, [33,35,24, 0]].values 
# y = dataset.iloc[:, [2]].values # Total food expenditure

# X1 = dataset.iloc[:, [33,35,24, 0]].values 
# y1 = dataset.iloc[:, [2]].values # Total food expenditure

# X2 = dataset.iloc[:, [33,35,24, 0]].values 
# y2 = dataset.iloc[:, [2]].values # Total food expenditure



# print(X[-1])
# print(y[-1])
# print(type(X))

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle=False)

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# poly_reg = PolynomialFeatures(degree = 2)
# X_poly = poly_reg.fit_transform(X_train)
# regressor = LinearRegression()
# regressor.fit(X_poly, y_train)

# na = np.array([[3,0,5000, 108000]])
# y_pred = regressor.predict(poly_reg.transform(na))
# print(na, y_pred[-1])
