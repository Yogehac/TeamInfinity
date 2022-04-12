import numpy as np 
import pandas as pd
path = 'F:\\PROJECTS\\Infinity\\Family Income and Expenditure.csv'
dataset = pd.read_csv(path)

def predict(totalPpl, totalChild, totalIncome, otherIncome, totalBike, totalCar):
    X = dataset.iloc[:, [33,35,24, 0]].values 
    y = dataset.iloc[:, [2]].values # Total food expenditure

    X1 = dataset.iloc[:, [33,35,24, 0]].values 
    y1 = dataset.iloc[:, [20]].values # Total edu expenditure

    X2 = dataset.iloc[:, [53,59,24, 0]].values 
    y2 = dataset.iloc[:, [18]].values # Total transport expenditure


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, shuffle=False)

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.3, random_state = 0, shuffle=False)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.3, random_state = 0, shuffle=False)


    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)

    poly_reg1 = PolynomialFeatures(degree = 2)
    X1_poly = poly_reg1.fit_transform(X1_train)
    regressor1 = LinearRegression()
    regressor1.fit(X1_poly, y1_train)

    poly_reg2 = PolynomialFeatures(degree = 2)
    X2_poly = poly_reg2.fit_transform(X2_train)
    regressor2 = LinearRegression()
    regressor2.fit(X2_poly, y2_train)

    # # user Inputs!!!!!!!!
    # totalIncome = 120000
    # otherIncome = 10000
    # totalPpl = 4
    # totalChild = 2
    # totalBike = 1
    # totalCar = 0

    foodIpt = np.array([[totalPpl, totalChild, otherIncome, totalIncome]])
    y_pred = regressor.predict(poly_reg.transform(foodIpt))
    print(foodIpt, y_pred[-1])


    eduIpt = np.array([[totalPpl, totalChild, otherIncome, totalIncome]])
    y_pred1 = regressor1.predict(poly_reg.transform(eduIpt))
    print(eduIpt, y_pred1[-1])

    transIpt = np.array([[totalCar, totalBike, otherIncome, totalIncome]])
    y_pred2 = regressor2.predict(poly_reg.transform(transIpt))
    print(transIpt, y_pred2[-1])

    Estimation = {
        'Food' : int(y_pred.tolist()[0][0]),
        'Education' : int(y_pred1.tolist()[0][0]),
        'Transport' : int(y_pred2.tolist()[0][0])
        }
    Estimated = [totalIncome, int(y_pred.tolist()[0][0]), int(y_pred1.tolist()[0][0]), int(y_pred2.tolist()[0][0])]
    return Estimated
