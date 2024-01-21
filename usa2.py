import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy import stats

# Define the err_ranges function for confidence intervals
def err_ranges(x, y, model, params, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    nstd = abs(stats.norm.ppf(alpha / 2.))  # 2-tailed z score
    lower = np.zeros_like(x)
    upper = np.zeros_like(x)
    for i in range(len(x)):
        lower[i], upper[i] = stats.norm.interval(alpha, loc=model(x[i], *params), scale=perr.sum())
    return lower, upper

# Curve fitting function
def linear_func(x, a, b):
    return a * x + b

# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to perform k-means clustering and plot results
def kmeans_clustering(data):
    normalized_data = (data - data.mean()) / data.std()
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_features = ['usa_GDP_Per_Capita', 'usa_CO2']
    data['Cluster'] = kmeans.fit_predict(normalized_data[cluster_features])

    # Plotting CO2 Emissions
    plt.figure(figsize=(12, 6))
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data['Year'], cluster_data['usa_CO2'], label=f'Cluster {cluster + 1}')

    plt.xlabel('Year')
    plt.ylabel('USA CO2 Emissions')
    plt.title('Clustering of USA CO2 Emissions over Time')
    plt.legend()
    plt.show()

    # Plotting GDP Per Capita
    plt.figure(figsize=(12, 6))
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data['Year'], cluster_data['usa_GDP_Per_Capita'], label=f'Cluster {cluster + 1}')

    plt.xlabel('Year')
    plt.ylabel('USA GDP_Per_Capita')
    plt.title('Clustering of USA GDP_Per_Capita over Time')
    plt.legend()
    plt.show()

# Function to perform curve fitting and plot results
def curve_fitting(data):
    # Curve fitting for GDP
    popt_gdp, pcov_gdp = curve_fit(linear_func, data['Year'], data['usa_GDP_Per_Capita'])
    future_years = np.arange(2021, 2031)
    predicted_values_gdp = linear_func(future_years, *popt_gdp)
    lower_bound_gdp, upper_bound_gdp = err_ranges(future_years, data['usa_GDP_Per_Capita'], linear_func, popt_gdp, pcov_gdp)

    # Curve fitting for CO2 Emissions
    popt_co2, pcov_co2 = curve_fit(linear_func, data['Year'], data['usa_CO2'])
    predicted_values_co2 = linear_func(future_years, *popt_co2)
    lower_bound_co2, upper_bound_co2 = err_ranges(future_years, data['usa_CO2'], linear_func, popt_co2, pcov_co2)

    # Plotting the curve fit with forecasting for GDP
    plt.figure(figsize=(12, 6))
    plt.scatter(data['Year'], data['usa_GDP_Per_Capita'], label='Actual Data (GDP)')
    plt.plot(future_years, predicted_values_gdp, label='Best Fitting Function (GDP)', color='pink')
    plt.fill_between(future_years, lower_bound_gdp, upper_bound_gdp, color='orange', alpha=0.2, label='Confidence Range (GDP)')

    # Plotting the forecasted values for GDP
    plt.plot(future_years, linear_func(future_years, *popt_gdp), linestyle='dashed', color='green', label='Forecasted Values (GDP)')

    plt.xlabel('Year')
    plt.ylabel('USA GDP Per Capita')
    plt.title('Curve Fit and Forecast of USA GDP Per Capita over Time')
    plt.legend()
    plt.show()

    # Plotting the curve fit with forecasting for CO2 Emissions
    plt.figure(figsize=(12, 6))
    plt.scatter(data['Year'], data['usa_CO2'], label='Actual Data (CO2)')
    plt.plot(future_years, predicted_values_co2, label='Best Fitting Function (CO2)', color='blue')
    plt.fill_between(future_years, lower_bound_co2, upper_bound_co2, color='cyan', alpha=0.2, label='Confidence Range (CO2)')

    # Plotting the forecasted values for CO2 Emissions
    plt.plot(future_years, linear_func(future_years, *popt_co2), linestyle='dashed', color='red', label='Forecasted Values (CO2)')

    plt.xlabel('Year')
    plt.ylabel('USA CO2 Emissions')
    plt.title('Curve Fit and Forecast of USA CO2 Emissions over Time')
    plt.legend()
    plt.show()

    # Calculate residuals for GDP
    residuals_gdp = data['usa_GDP_Per_Capita'] - linear_func(data['Year'], *popt_gdp)
'''
    # Plot Q-Q plot for residuals for GDP
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals_gdp, dist="norm", plot=plt)
    plt.title('Normality Q-Q Plot of Residuals (GDP)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()

    # Calculate residuals for CO2 Emissions
    residuals_co2 = data['usa_CO2'] - linear_func(data['Year'], *popt_co2)

    # Plot Q-Q plot for residuals for CO2 Emissions
    plt.figure(figsize=(8, 8))
    stats.probplot(residuals_co2, dist="norm", plot=plt)
    plt.title('Normality Q-Q Plot of Residuals (CO2)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()
'''
# Load data from CSV file
file_path = r"C:\Users\heman\OneDrive\Desktop\usadataset 1.csv"
data = load_data(file_path)

# Perform k-means clustering and plot results
kmeans_clustering(data)

# Perform curve fitting and plot results with forecasting for both GDP and CO2 Emissions
curve_fitting(data)
