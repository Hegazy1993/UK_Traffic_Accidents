#!/usr/bin/env python
# coding: utf-8

# 
# 
# # _A 5-year Analysis of UK Traffic Accidents from 2017-2021_
# 
# 
# * Abdelrahman Ibrahim
# * Callum Scoby
# * Fraser McKnespiey

# # 1. Project Plan
# 
# ## 1.1 The Data
# 
# Data was sourced from the UK Government Department of Transport. All data are police recorded traffic accidents.
# 
# ### Primary datasets
# <b> a. Road accidents from 2017 to 2021:</b>
# 
# Each row in this dataset corresponds to one accident recorded during the time period. Each accident has a number of attributes including an accident index (a unique reference), day of the week, accident severity, number of vehicles and casualties, speed limit, local authority ONS district code, and driving conditions. Presented below are the data types of each column in this dataset.
# 
#  <html>
#  <head>
#  <style>
#  table, th, td {
#    border: 1px solid black;
#    border-collapse: collapse;
#  }
#  </style>
#  </head>
#  <body>
# 
#  <table>
#   <tr>
#     <th>Variable</th>
#     <th>Type</th>
#     <th>Notes</th>
# 
#   </tr>
#   <tr>
#     <td>accident_index</td>
#     <td>Nominal</td>
#     <td>Unique identifier</td>
# </tr>
#   <tr>
#     <td>accident_year</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   <tr>
#     <td>longitude</td>
#     <td>Continuous</td>
#     <td></td></tr>
#   <tr>
#     <td>latitude</td>
#     <td>Continuous</td>
#     <td></td></tr>
#   <tr>
#     <td>accident_severity</td>
#     <td>Ordinal</td>
#     <td>Ranking accident severity from 3 (slight) to 1 (fatal)</td></tr>
#   <tr>
#     <td>number_of_vehicles</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   <tr>
#     <td>number_of_casualties</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   <tr>
#     <td>date</td>
#     <td>Interval</td>
#     <td></td></tr>
#   <tr>
#     <td>day_of_week</td>
#     <td>Nominal</td>
#     <td>A nominal representation of the days of week in values from 1 to 7</td>
#       </tr>
#   <tr>
#     <td>time</td>
#     <td>Continuous</td>
#     <td></td></tr>
#   <tr>
#     <td>local_authority_ons_district</td>
#     <td>Nominal</td>
#     <td>Codes for UK districts</td></tr>
#   <tr>
#     <td>road_type</td>
#     <td>Nominal</td>
#     <td></td></tr>
#   <tr>
#     <td>speed_limit</td>
#     <td>Ordinal</td>
#     <td>Speed limits of the road where the accident occured</td></tr>
#   <tr>
#     <td>light_conditions</td>
#     <td>Nominal</td>
#     <td>Categories of different lighting conditions in the time of the accident</td></tr>
#   <tr>
#     <td>weather_conditions</td>
#     <td>Nominal</td>
#     <td>Categories of different weather in the time of the accident </td></tr>
#   <tr>
#     <td>road_surface_conditions</td>
#     <td>Nominal</td>
#     <td>Categories of road surface condition in the time of the accident</td></tr>
#   <tr>
#     <td>urban_or_rural_area</td>
#     <td>Nominal</td>
#     <td>Categories for the type of the area (urban, rural, unidentified)</td></tr>
#   
# </table> 
# 
# <b> b. Vehicles involved in road accidents from 2017 to 2021 </b>
# 
# Each row in this dataset corresponds to one vehicle involved in a recorded accident. Thus, a single accident may be represented by multiple rows. Each accident has attributes including accident index (suitable for joining with the accidents dataset), driver age bands, and driver sex. Presented below are the data types of each column in this dataset.     
# <html>
#  <head>
#  <style>
#  table, th, td {
#    border: 1px solid black;
#    border-collapse: collapse;
#  }
#  </style>
#  </head>
#  <body>
# 
#  <table>
#   <tr>
#     <th>Variable</th>
#     <th>Type</th>
#     <th>Notes</th>
# 
#   </tr>
#   <tr>
#     <td>accident_index</td>
#     <td>Nominal</td>
#     <td>Unique identifier and the common variable with the accident dataset</td>
# </tr>
#   <tr>
#     <td>vehicle_type</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   <tr>
#     <td>sex_of_driver</td>
#     <td>Nominal</td>
#     <td></td></tr>
#   <tr>
#     <td>age_of_driver</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   <tr>
#     <td>age_band_of_driver</td>
#     <td>Ordinal</td>
#     <td>Age bands of drivers from 1 (0 to 5 years) to 11 (75 plus years). Note that a driver can be younger than 5 because a bicycle is considered a vehicle</td></tr>
#   <tr>
#     <td>age_of_vehicle</td>
#     <td>Discrete</td>
#     <td></td></tr>
#   
# </table> 
# 
# 
# ### Supporting Datasets
# 
# The Road Safety Data Guide published by the Department of Transport acts as the data dictionary for this analysis. It is also used to convert each Local Authority ONS District code into their respective name.
#      
# ### Data Accuracy
# 
# Accidents recorded in these datasets indicate a road accident where the police were notified, and at least one person was injured, or one vehicle was damaged. Thus, the data reflects only those accidents that were reported to or attended by police.
# 
# A small proportion of the data are missing values- in the case of latitude and longitude this proportion was around 1%. On the whole, however, the dataset is comprehensive and detailed. Sourcing from official Government publications further enhances confidence in the data accuracy. 
# 
# 
# ## 1.2 Project Aim and Objectives (5 marks)
# 
# Our project aims to develop an understanding of how the occurrence and severity of traffic accidents in the UK are related to driving and demographic conditions through the analysis of UK Government data sourced over a 5-year period (2017-2021).
#      
# Our analysis seeks to examine how the occurrence of accidents varies spatially and temporally, and as a result of driving and demographic conditions. It also aims to understand how the odds of being in an accident of greater severity are associated with driving and demographic conditions.
#      
# To do so, we aim to initially visualise the spatial distribution of accident occurrences across the UK. We seek to do this in an interactive format that makes intuitive sense to a viewer. This will aid in the understanding of accident hotspots within the UK, while also providing locational context to the data. In tandem, we aim to understand the temporal patterns of accident occurrences in the dataset. This may involve identifying the rate of change of accidents across the 5-year period and will add temporal context to the data. 
# 
# Having gained an understanding of the distribution and pattern of accident occurrences, we seek to visualise trends in accident occurrences against contextual driving and demographic factors. Next, we aim to identify how factors affect the odds of being in a crash of greater severity. To do so, we aim to first statistically assess how accident severity odds vary as a function of contextual driving conditions such as light, weather, road conditions, speed limit, and road class. Finally, we seek to statistically assess how accident severity odds vary as a result of contextual demographic factors, including driver age and sex. 
# 
# 
# ### Specific Objective(s)
#      
# The four key objectives of this project are as follows: 
# 
# * __Objective 1:__ _Exploring the spatial and temporal distribution of accidents within the UK using the Folium library_
# * __Objective 2:__ _Using the Matplotlib package, visualise how accident occurrences vary with driving and demographic conditions_
# * __Objective 3:__ _Statistically assess, using the statsmodels package, how driving conditions affect the odds of being in an accident of greater severity_
# * __Objective 4:__ _Statistically assess, using the statsmodels package, how demographic conditions affect the odds of being in an accident of greater severity_

# ## 1.3 System Design (5 marks)

# ### Architecture
# 
# The following illustration highlights the pipeline taken during this Data Analysis Project.
# 
# 1. Gathering Data
#     - UK Government Transport Data: 
#         - After finding our data from the UK Government, the next stage was for cleaning and wrangling to prepare the data for analysis. 
# 2. Data Cleaning & Wrangling Using Pandas
#     - DataFrame Merges Using Like-for-Like Variables: 
#         - Extract demographic and vehicle data from the vehicles dataset and assign datapoints to their respective accident. 
#     - Encoding Categorical Variables of Interest to 'Dummy' Binary Variables: 
#         - We convert catagorical data (nominal and ordinal) to binary columns using numpy and list comprehension to prepare them for regression analysis
# 3. Data Visualisations
#     - Spatial and Temporal Visualisations using Folium:
#         - During the visualisation phase we aim to explore spatial and temporal patterns to find insightful information about the data. 
#     - Plotting visualisations using Matplotlib:
#         - Using novel methods of Matplotlib to generate appropriate visualisations to further explore data. 
#     - Statistically assessing how driving conditons/demographic conditions affect accident severity using StatsModels:
#         - Using Ordinal Linear Regression to assess how driving conditions and demographic factors affect the odds of being in an accident of greater severity
# 4. Interpretations, Explinations & Conclusions:
#     - Finally, we call the functions to display outputs, and derive real world conclusions on the spatial and statistical facts about road accidents in the targeted 5 years.

# ![image.png](attachment:image.png)

# ### Processing Modules and Algorithms

# - **Data cleansing**
#     - Once imported, each data frame (accidents, vehicles, district names) was subset to include only the variables of interest. 
#     - Missing data (represented by ‘-1’ in the original data frames) was not removed because it had no bearing on our results. This allowed us to keep a comprehensive number of data points.
#     - Accident severity levels (1 = fatal, 2 = serious, 3 = slight) were reversed (1 = slight, 2 = serious, 3 = fatal) for intuitive interpretation.
# - **Merging data frames**
#     - The cleansed data frames were merged to create a singular and uniform master data frame. 
#     - A column of ‘date_time’ was created using the datetime package to enable date and time manipulation in temporal analysis.
# - **Converting variables to special representations**
#     - The accident severity variable is an ordinal categorical variable (slight, serious, fatal) and so it was converted to a categorical and ordered data type, with the variable name ‘accident_severity_OLR’
#     - Dummy variables were created for all categorical variables of interest, producing binary columns for each value within a categorical variable. For example, the ‘light conditions’ variable contained values of ‘1’, ‘4’, ‘5’, ‘6’, ‘7’, and ‘-1’, with each representing a different light condition. Thus, each value was converted to its own binary variable. 
# - **Interactive spatial distribution maps were created using the Folium library**
#     - Spatial distribution observed through using latitude and longitude for each accident which are then visualised using clusters and heat mapping. 
# - **Ordinal Logistic Regression (OLR) using statsmodels package**
#     - OLR was undertaken to assess the statistical odds of being in an accident of greater severity as a result of the different binary variables. Odds ratios were created by calculating the inverse exponent of the regression coefficients. 
# 

# # 2. Program Code

# ## 2.1 Data aquisition, cleaning and wrangling

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt


# In[2]:


# Importing the two road safety datasets
accident_original = pd.read_csv('https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-accident-last-5-years.csv', dtype={'accident_index': 'str', 'accident_reference': 'str'})
vehicle_original = pd.read_csv('https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-last-5-years.csv', dtype={'accident_index': 'str', 'accident_reference': 'str', 'generic_make_model': 'str'})
district_name = pd.DataFrame(pd.read_excel('https://data.dft.gov.uk/road-accidents-safety-data/Road-Safety-Open-Dataset-Data-Guide.xlsx'))


# In[3]:


# Subsetting the accidents_original df to the columns we want
accident_clean = accident_original[['accident_index', 'accident_year',
                               'location_easting_osgr', 'location_northing_osgr',
                                'longitude', 'latitude', 'accident_severity',
                                      'number_of_vehicles', 'number_of_casualties',
                                      'date', 'day_of_week', 'time',
                                      'local_authority_ons_district', 'first_road_class',
                                      'road_type', 'speed_limit', 'light_conditions',
                                      'weather_conditions', 'road_surface_conditions',
                                      'urban_or_rural_area']]

# Subsetting the vehicle_original df to the columns we want
vehicle_clean = vehicle_original[['accident_index', 'vehicle_type', 
                                 'sex_of_driver', 'age_of_driver',
                                 'age_band_of_driver', 'engine_capacity_cc',
                                 'driver_imd_decile', 'age_of_vehicle']]

# Subsetting the district_name df to the rows and columns we want
district_name = district_name.loc[district_name['field name'] == 'local_authority_ons_district']
district_name = district_name.drop(['table', 'field name', 'note'], axis=1)


# In[4]:


district_name = district_name.drop_duplicates()


# In[5]:


#rename columns in district_name df to join with accident_clean df
district_name.rename(columns={'code/format': 'local_authority_ons_district', 'label': 'district_name'}, inplace=True)


# In[6]:


driver_age_bands = vehicle_clean.groupby('accident_index')['age_band_of_driver'].apply(list)  

#produce a series with a set of all driver age bands involved in each accident


# In[7]:


driver_age_bands.to_frame()
#convert the result to a new dataframe


# In[8]:


accident_master = pd.merge(accident_clean, driver_age_bands, on="accident_index", how="left")
#merge age bands of drivers with accindents dataframe  


# In[9]:


#we repeat the process for sex of drivers 
driver_sex = vehicle_clean.groupby('accident_index')['sex_of_driver'].apply(list)


# In[10]:


driver_sex.to_frame()


# In[11]:


accident_master = pd.merge(accident_master, driver_sex, on="accident_index", how="left")
#merge sex of drivers with accidents_drivers dataframe  


# In[12]:


age_0_5 = [1 if 1 in x else 0 for x in (accident_master['age_band_of_driver']) ]
accident_master["age_0_5"] = age_0_5


# In[13]:


age_6_10 = [1 if 2 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_6_10"] = age_6_10


# In[14]:


age_11_15 = [1 if 3 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_11_15"] = age_11_15


# In[15]:


age_16_20 = [1 if 4 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_16_20"] = age_16_20


# In[16]:


age_21_25 = [1 if 5 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_21_25"] = age_21_25


# In[17]:


age_26_35 = [1 if 6 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_26_35"] = age_26_35     


# In[18]:


age_36_45 = [1 if 7 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_36_45"] = age_36_45


# In[19]:


age_46_55 = [1 if 8 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_46_55"] = age_46_55


# In[20]:


age_56_65 = [1 if 9 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_56_65"] = age_56_65        


# In[21]:


age_66_75 = [1 if 10 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_66_75"] = age_66_75


# In[22]:


age_75plus = [1 if 11 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_75plus"] = age_75plus


# In[23]:


age_nodata = [1 if -1 in x else 0 for x in (accident_master['age_band_of_driver'])]
accident_master["age_nodata"] = age_nodata


# In[24]:


sex_male = [1 if 1 in x else 0 for x in (accident_master['sex_of_driver'])]
accident_master["sex_male"] = sex_male


# In[25]:


sex_female = [1 if 2 in x else 0 for x in (accident_master['sex_of_driver'])]
accident_master["sex_female"] = sex_female


# In[26]:


sex_NA = [1 if 3 in x else 0 for x in (accident_master['sex_of_driver'])]
accident_master["sex_NA"] = sex_NA


# In[27]:


sex_nodata = [1 if -1 in x else 0 for x in (accident_master['sex_of_driver'])]
accident_master["sex_nodata"] = sex_nodata


# In[28]:


accident_master = accident_master.drop(['age_band_of_driver', 'sex_of_driver'], axis=1)


# In[29]:


accident_master = accident_master.replace({'accident_severity': 1}, 5)
accident_master = accident_master.replace({'accident_severity': 3}, 1)
accident_master = accident_master.replace({'accident_severity': 5}, 3)


# In[30]:


accident_master['Monday'] = np.where((accident_master['day_of_week'] == 2), 1, 0)
accident_master['Tuesday'] = np.where((accident_master['day_of_week'] == 3), 1, 0 ) 
accident_master['Wednesday'] = np.where((accident_master['day_of_week'] == 4), 1, 0 ) 
accident_master['Thursday'] = np.where((accident_master['day_of_week'] == 5), 1, 0 ) 
accident_master['Friday'] = np.where((accident_master['day_of_week'] == 6), 1, 0 ) 
accident_master['Saturday'] = np.where((accident_master['day_of_week'] == 7), 1, 0 ) 
accident_master['Sunday'] = np.where((accident_master['day_of_week'] == 1), 1, 0 ) 


# In[31]:


accident_master['Daylight'] = np.where((accident_master['light_conditions'] == 1), 1, 0 ) 
accident_master['Darkness_lights_lit'] = np.where((accident_master['light_conditions'] == 4), 1, 0 ) 
accident_master['Darkness_lights_unlit'] = np.where((accident_master['light_conditions'] == 5), 1, 0 )
accident_master['Darkness_no_lighting'] = np.where((accident_master['light_conditions'] == 6), 1, 0 )
accident_master['Darkness_lighting_unknown'] = np.where((accident_master['light_conditions'] == 7), 1, 0 )


# In[32]:


accident_master['Fine_no_high_winds'] = np.where((accident_master['weather_conditions'] == 1), 1, 0 ) 
accident_master['Raining_no_high_winds'] = np.where((accident_master['weather_conditions'] == 2), 1, 0 ) 
accident_master['Snowing_no_high_winds'] = np.where((accident_master['weather_conditions'] == 3), 1, 0 )
accident_master['Fine_and_high_winds'] = np.where((accident_master['weather_conditions'] == 4), 1, 0 )
accident_master['Raining_and_high_winds'] = np.where((accident_master['weather_conditions'] == 5), 1, 0 )
accident_master['Snowing_and_high_winds'] = np.where((accident_master['weather_conditions'] == 6), 1, 0 )
accident_master['Fog_or_mist'] = np.where((accident_master['weather_conditions'] == 7), 1, 0 )
accident_master['Other'] = np.where((accident_master['weather_conditions'] == 8), 1, 0 )
accident_master['Unknown'] = np.where((accident_master['weather_conditions'] == 9), 1, 0 )


# In[33]:


accident_master['Dry_road'] = np.where((accident_master['road_surface_conditions'] == 1), 1, 0 ) 
accident_master['Wet_or_damp_road'] = np.where((accident_master['road_surface_conditions'] == 2), 1, 0 )
accident_master['Snow_road'] = np.where((accident_master['road_surface_conditions'] == 3), 1, 0 )
accident_master['Frost_or_ice_road'] = np.where((accident_master['road_surface_conditions'] == 4), 1, 0 )
accident_master['Flood_over_3cm_deep_road'] = np.where((accident_master['road_surface_conditions'] == 5), 1, 0 )
accident_master['Oil_or_diesel_road'] = np.where((accident_master['road_surface_conditions'] == 6), 1, 0 )
accident_master['Mud_road'] = np.where((accident_master['road_surface_conditions'] == 7), 1, 0 )
accident_master['Unknown'] = np.where((accident_master['road_surface_conditions'] == 9), 1, 0 )


# In[34]:


accident_master['Motorway_road_class'] = np.where((accident_master['first_road_class'] == 1), 1, 0 )
accident_master['A_M_road_class'] = np.where((accident_master['first_road_class'] == 2), 1, 0 )
accident_master['A_road_road_class'] = np.where((accident_master['first_road_class'] == 3), 1, 0 )
accident_master['B_road_class'] = np.where((accident_master['first_road_class'] == 4), 1, 0 )
accident_master['C_road_class'] = np.where((accident_master['first_road_class'] == 5), 1, 0 )
accident_master['Unclassified_road_class'] = np.where((accident_master['first_road_class'] == 6), 1, 0 )


# In[35]:


accident_master['20_mph'] = np.where((accident_master['speed_limit'] == 20), 1, 0 )
accident_master['30_mph'] = np.where((accident_master['speed_limit'] == 30), 1, 0 )
accident_master['40_mph'] = np.where((accident_master['speed_limit'] == 40), 1, 0 )
accident_master['50_mph'] = np.where((accident_master['speed_limit'] == 50), 1, 0 )
accident_master['60_mph'] = np.where((accident_master['speed_limit'] == 60), 1, 0 )
accident_master['70_mph'] = np.where((accident_master['speed_limit'] == 70), 1, 0 )


# In[36]:


accident_master['Urban_area'] = np.where((accident_master['urban_or_rural_area'] == 1), 1, 0 )
accident_master['Rural_area'] = np.where((accident_master['urban_or_rural_area'] == 2), 1, 0 )
accident_master['Unallocated_area'] = np.where((accident_master['urban_or_rural_area'] == 3), 1, 0 )


# In[37]:


#join district_name df with accident_clean df
accident_master = pd.merge(accident_master, district_name, 'left', on="local_authority_ons_district")


# The next step is to create a new column called `date_time` which combines the `date` and `time` columns by using the Pandas function `pd.to_datetime()`. By doing this, we are able to easily computate temporal information when required

# In[38]:


accident_master['date_time'] = accident_master['date'] +' '+ accident_master['time']
accident_master['date_time'] = pd.to_datetime(accident_master.date_time)


# In[39]:


# Check the data
accident_master.head()


# In[40]:


accident_master.info()


# ## 2.2 Statistical modelling and visualization

# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
import calendar
from ipyleaflet import Map, basemaps, basemap_to_tiles, Circle, Polyline
import folium as fl
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMapWithTime
import geopandas as gpd


# In[42]:


# Converting accident_severity to an ordered categorical type variable, to ensure proper functioning in 
# Ordinal Logistic Regression analysis
accident_master['accident_severity_OLR'] = pd.Categorical(accident_master['accident_severity'], ordered=True)
accident_master['accident_severity_OLR'].dtype


# ### A note on Ordinal Logistic Regression

# To assess the influence of variables upon accident severity, Ordinal Logistic Regression (OLR) was used. OLR takes categorical and continuous predictor variables and models the log odds outcomes of an ordinal categorical response variable. This technique was chosen because accident severity is an ordinal variable (slight, serious, fatal), and so must be analysed using an appropriate method. 

# OLR makes a number of **assumptions**:
# 
# **1.** The response variable is ordered.
# 
# Accident severity is an ordered variable, and was defined as such in _2.1 Data aquisition, cleaning and wrangling_
# 

# In[43]:


accident_master['accident_severity_OLR'].dtype


# **2.** One or more predictor variables are either continuous, categorical or ordinal- all variables of interest (driver age band, driver sex, day of week, light conditions, weather conditions, road conditions, road class, urban or rural area) are categorical in the original datasets, and were converted to dummy variables in the _2.1 Data aquisition, cleaning and wrangling_.
# 
# **3.** No multicollinearity is present- this will be assessed in turn for each analysis.
# 
# **4.** Proportional odds- the assumption that the effects of predictor variables are consistent or proportional across the thresholds of the ordinal response variable categories (in this case the threshold between a slight or serious accident, or between a serious and fatal accident). 
# 
# **OrderedModel** is the statsmodel package that will be used to conduct OLR. OrderedModel is a relatively new addition to the statsmodels package and, as of yet, there are no official methods to validate Assumption 4. As such, we cannot be entirely sure that the assumption is satisfied. However, if this assumption is violated, it is likely to have little bearing on the accuracy of model results.
# 

# ### Where are the UK accident hotspots?

# In[44]:


def accident_hotspots():
    print(accident_master['district_name'].value_counts().head())


# In[45]:


accident_hotspots()


# **Geospatial distribution of accidents**

# In[46]:


accident_location = accident_master[['latitude', 'longitude', 'accident_severity', 'date_time']].copy()
accident_location = accident_location.dropna(subset=['latitude','longitude'])


# In[47]:


def accidents_interactive_map():
    map1 = fl.Map(location=[accident_location.latitude.mean(),\
                           accident_location.longitude.mean()],\
                 zoom_start=4.5, control_scale=True)
    map1.add_child(FastMarkerCluster(accident_location[['latitude', 'longitude']].values.tolist()))
    return map1


# In[48]:


accidents_interactive_map() #Please check the interactive map in the HTML File


# In[49]:


def heat_interactive_map():
    map2 = fl.Map(location=[accident_location.latitude.mean(),\
                               accident_location.longitude.mean()],\
                     zoom_start=4.5, control_scale=True)
    hr_list = [[] for x in range(24)]
    for latitude,longitude,hour in zip(accident_location.latitude, accident_location.longitude, accident_location.date_time.dt.hour.values.tolist()):
        hr_list[hour].append([latitude, longitude])
    
    index = [str(i) + ' Hours' for i in range (24)]
    
    HeatMapWithTime(hr_list, index).add_to(map2)
    
    return map2


# In[50]:


heat_interactive_map() #Please check the interactive map in the HTML File


# ### The odds of being in a greater severity accident by gender

# #### Checking that the multicollinearity assumption is met

# In[51]:


df_gender_multicollinearity_test = accident_master.iloc[:,32:34]
df_gender_multicollinearity_test_print = df_gender_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_gender_multicollinearity_test_print)


# As the categories of male drivers and female drivers involved in accidents are effectively measuring the same thing (the proportion of drivers of each gender), complete multicollinearity is exhibited. As such, only 'sex_male' will be included in the OLR model. This means that 'sex_female' can be used as a benchmark to compare odds against.

# #### OLR Model

# In[52]:


def gender_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['sex_male']], distr='logit')

    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios (i.e. the proportional odds of being in an accident of greater 
    # severity (i.e. serious or fatal, rather than slight, as a result of the different light conditions))
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[53]:


gender_severity_logistic_ordinal_regression()


# ### How does the occurence of road accidents vary with gender?

# In[54]:


def gender_accident_occurences():
    df_accidents_by_gender = accident_master
    df_accidents_by_gender = df_accidents_by_gender.replace({'sex_female': 1}, 2)
    gender = []
    gender.append((df_accidents_by_gender['sex_male'] == 1).sum())
    gender.append((df_accidents_by_gender['sex_female'] == 2).sum())
    gender_df = pd.DataFrame (gender, columns = ['gender'])
    
    # Series with number of mild injuries and serious injuries
    gender_pie_chart = gender_df['gender']

    # Pie plot with the percentage of victims with slight, serious and fatal injuries
    explode = (0.00, 0.03)
    gender_pie_chart.plot(kind = 'pie',figsize = (10,10), explode = explode, colors = ['forestgreen','orange','orangered'], labels = None, autopct = '%1.1f%%', pctdistance = 1.2, fontsize = 18)

    # Title and legend
    plt.legend(labels = ['Male', 'Female'],bbox_to_anchor = (0.85, 0.95), loc='upper left', borderaxespad=0, fontsize = 15)
    plt.title('Occurence of road accidents by gender (2017-2021)', fontsize = 25)
    plt.ylabel('')


# In[55]:


gender_accident_occurences()


# ## Light conditions

# ### The odds of being in a greater severity accident by light conditions

# #### Checking that the multicollinearity assumption is met

# In[56]:


df_lightcond_multicollinearity_test = accident_master.iloc[:,43:47]
df_lightcond_multicollinearity_test_print = df_lightcond_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_lightcond_multicollinearity_test_print)


# The 'daylight' category exhibits the highest levels of multicollinearity with the other variables. Correlation of -0.80 exists between daylight and darkness (lights lit), and -0.37 between daylight and darkness (no lighting). Additionally, daylight light conditions comprise a majority of the occurences of accidents. As such, daylight is not included in the OLR model, meaning it can be used as the reference variable.

# Here, **Ordinal Logistic Regression** is used to understand whether light conditions affect the odds of being in an accident of greater severity (i.e. serious or fatal, rather than slight). This regression technique was chosen because accident severity is an ordinal variable (slight, serious, fatal), and so should not be analysed using linear regression techniques. 
# 
# Daylight conditions were chosen as the reference variable, as they provide an intuitive benchmark against which to compare, and because they comprise the bulk of accidents. The dummy variables (variables that have been converted from a scale in a single column (e.g. 1-5 in this example) to multiple binary columns): 1) Darkness_lights_lit, 2) Darkness_lights_unlit, 3) Darkness_no_lighting are used, while Darkness_lighting_unknown is removed from the analysis as it does not provide useful information.
# 
# A **logit** ordinal regression method was employed rather than a probit method, because this allows for more intuitive analysis, and because the choice is negligible for a large sample size.

# #### OLR Model

# In[57]:


def light_conditions_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Darkness_lights_lit', 'Darkness_lights_unlit', 
                                                       'Darkness_no_lighting']], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[58]:


light_conditions_severity_logistic_ordinal_regression()


# ### How does the occurence of road accidents vary with light conditions?

# In[59]:


def light_conditions_accident_occurences():
    a = len(accident_master[accident_master.Daylight == 1])
    b = len(accident_master[accident_master.Darkness_lights_lit == 1])
    c = len(accident_master[accident_master.Darkness_lights_unlit == 1])
    d = len(accident_master[accident_master.Darkness_no_lighting == 1])


    light_conditions_df = pd.DataFrame({'light_conditions': ['Daylight', 'Darkness_lights_lit', 'Darkness_lights_unlit',\
                                           'Darkness_no_lighting'],\
                       'accident_count': [a, b, c, d]})
    
    x = light_conditions_df.light_conditions
    y = light_conditions_df.accident_count
    fig,ax = plt.subplots(figsize=(15, 10))
    plt.bar(x, y)
    labels = ['Daylight', 'Darkness (lights lit)','Darkness (lights unlit)', 'Darkness (no lighting)']
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Occurence of road accidents by light conditions (2017-2021)', fontsize=25)
    plt.xlabel('Light conditions', fontsize=18)
    plt.ylabel('Accident Occurences', fontsize=18)

    plt.show()


# In[60]:


light_conditions_accident_occurences()


# ### The odds of being in a greater severity accident by weather conditions

# #### Checking that the multicollinearity assumption is met

# In[61]:


df_weathercond_multicollinearity_test = accident_master.iloc[:,48:54]
df_weathercond_multicollinearity_test_print = df_weathercond_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_weathercond_multicollinearity_test_print)


# Fine (no high winds) weather conditions exhibit the greatest levels of multicollinearity with the other categories. Correlation of -0.71 exists between fine (no high winds) and raining (no high winds), and -0.20 against fine (with high winds). As such, the fine (no high winds) weather condition is not used in the OLR analysis, and instead acts as the reference variable.

# **Ordinal Logistic Regression** is used to understand how weather conditions affect the odds of being in an accident of greater severity (i.e. serious or fatal, rather than slight). 
# 
# Fine with no high winds conditions were chosen as the reference variable, as they provide an intuitive benchmark against which to compare, and because they comprise the bulk of accidents. Dummy variables: 1) Raining_no_high_winds, 2) Snowing_no_high_winds, 3) Fine_and_high_winds, 4) Raining_and_high_winds, 5) Snowing_and_high_winds, 6) Fog_or_mist  are used.

# #### OLR Model

# In[62]:


def weather_conditions_severity_logistic_ordinal_regression():
# Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Raining_no_high_winds', 'Snowing_no_high_winds', 
                                                       'Fine_and_high_winds', 'Raining_and_high_winds', 
                                                       'Snowing_and_high_winds', 'Fog_or_mist']], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[63]:


weather_conditions_severity_logistic_ordinal_regression()


# ### How does the occurence of road accidents vary with weather conditions?

# In[64]:


def weather_conditions_accident_occurences():
    h = len(accident_master[accident_master.Fine_no_high_winds == 1])
    i = len(accident_master[accident_master.Raining_no_high_winds == 1])
    j = len(accident_master[accident_master.Snowing_no_high_winds == 1])
    k = len(accident_master[accident_master.Fine_and_high_winds == 1])
    l = len(accident_master[accident_master.Raining_and_high_winds == 1])
    m = len(accident_master[accident_master.Snowing_and_high_winds == 1])
    n = len(accident_master[accident_master.Fog_or_mist == 1])


    weather_conditions_df = pd.DataFrame({'weather_conditions': ['Fine_no_high_winds', 'Raining_no_high_winds', 'Snowing_no_high_winds',\
                                           'Fine_and_high_winds', 'Raining_and_high_winds', 'Snowing_and_high_winds',
                                         'Fog_or_mist'],\
                       'accident_count': [h, i, j, k, l, m, n]})

    x = weather_conditions_df.weather_conditions
    y = weather_conditions_df.accident_count
    fig,ax = plt.subplots(figsize=(20, 15))
    plt.bar(x, y)
    labels = ['Fine (no high winds)', 'Raining (no high winds)','Snowing (no high winds)', 'Fine (high winds)',
                 'Raining (high winds)', 'Snowing (high winds)', 'Fog or mist']
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=8)
    plt.title('Occurence of road accidents by weather conditions (2017-2021)', fontsize=25)
    plt.xlabel('Weather conditions', fontsize=18)
    plt.ylabel('Accident Occurences', fontsize=18)
    plt.show()


# In[65]:


weather_conditions_accident_occurences()


# ### The odds of being in a greater severity accident by road conditions

# #### Checking that the multicollinearity assumption is met

# In[66]:


df_roadcond_multicollinearity_test = accident_master.iloc[:,57:64]
df_roadcond_multicollinearity_test_print = df_roadcond_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_roadcond_multicollinearity_test_print)


# Dry road conditions exhibit the greatest levels of multicollinearity with the other categories. Correlation of -0.92 exists between dry road conditions and wet or damp roads, and -0.17 against frost or ice roads. As such, the dry road condition category is not used in the OLR analysis, and instead acts as the reference variable. Moreover, oil or diesel covered roads and mud covered roads are removed as the occurence of accidents in these conditions is zero.

# **Ordinal Logistic Regression** is used to understand how weather conditions affect the odds of being in an accident of greater severity (i.e. serious or fatal, rather than slight). 
# 
# Dry road conditions were chosen as the reference variable, as they provide an intuitive benchmark against which to compare, and because they comprise the bulk of accidents. Dummy variables: 1) Wet_or_damp_road, 2) Snow_road, 3) Frost_or_ice_road, 4) Flood_over_3cm_deep_road. Oil_or_diesel_road and Mud_road are removed as the occurences of accidents in these conditions is zero.
# 

# #### OLR Model

# In[67]:


def road_conditions_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Wet_or_damp_road', 'Snow_road', 
                                                       'Frost_or_ice_road', 'Flood_over_3cm_deep_road'
                                                       ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios (i.e. the odds of being in an accident of greater severity (i.e. serious
    # or fatal, rather than slight, as a result of the different road conditions))
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[68]:


road_conditions_severity_logistic_ordinal_regression()


# ### How does the occurence of road accidents vary with road conditions?
# 

# In[69]:


def road_conditions_accident_occurences():
    o = len(accident_master[accident_master.Dry_road == 1])
    p = len(accident_master[accident_master.Wet_or_damp_road == 1])
    q = len(accident_master[accident_master.Snow_road == 1])
    r = len(accident_master[accident_master.Frost_or_ice_road == 1])
    s = len(accident_master[accident_master.Flood_over_3cm_deep_road == 1])



    road_conditions_df = pd.DataFrame({'road_surface_conditions': ['Dry_road', 'Wet_or_damp_road', 'Snow_road',\
                                           'Frost_or_ice_road', 'Flood_over_3cm_deep_road'
                                         ],\
                       'accident_count': [o, p, q, r, s]})

    x = road_conditions_df.road_surface_conditions
    y = road_conditions_df.accident_count
    fig,ax = plt.subplots(figsize=(20, 15))
    plt.bar(x, y)
    labels = ['Dry', 'Wet or damp','Snow', 'Frost or ice',
                 'Flood over 3cm deep']
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=8)
    plt.title('Occurence of road accidents by road conditions (2017-2021)', fontsize=25)
    plt.xlabel('Road conditions', fontsize=18)
    plt.ylabel('Accident Occurences', fontsize=18)

    plt.show()


# In[70]:


road_conditions_accident_occurences()


# ### The odds of being in a greater severity accident by day of week
# 

# #### Checking that the multicollinearity assumption is met

# In[71]:


df_dayofweek_multicollinearity_test = accident_master.iloc[:,36:43]
df_dayofweek_multicollinearity_test_print = df_dayofweek_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_dayofweek_multicollinearity_test_print)


# Friday exhibits the greatest levels of multicollinearity with the other categories. Correlation of -0.19 exists between Friday and Thursday, -0.19 against Wednesday, and -0.18 against Tuesday. As such, the Friday category is not used in the OLR analysis, and instead acts as the reference variable.

# **Ordinal Logistic Regression** is used to understand how day of the week affects the odds of being in an accident of greater severity (i.e. serious or fatal, rather than slight). 
# 
# Monday was chosen as the reference variable for an intuitive comparison baseline.
# 
# Again, a **logit** ordinal regression method was employed rather than a probit method, because this allows for more intuitive analysis, and because the choice is negligible for a large sample size.

# #### OLR Model

# In[72]:


def day_of_week_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Tuesday', 'Wednesday', 
                                                       'Thursday', 'Friday' ,'Saturday', 'Sunday'
                                                       ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios (i.e. the odds of being in an accident of greater severity (i.e. serious
    # or fatal, rather than slight, as a result of the day of week))
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[73]:


day_of_week_severity_logistic_ordinal_regression()


# #### Distribution of Accidents Per Day of Week

# In[74]:


def accident_by_day():  
    accident_master['date_time'] = pd.to_datetime(accident_master.date_time)

    plt.figure(figsize = (15, 10))
    accident_master.date_time.dt.dayofweek.hist(bins = 7, rwidth = 0.75, alpha = 0.75, color = "royalblue")
    plt.title("Accidents Per Day Of Week", fontsize = 25)
    plt.ylabel('Accident Occurances', fontsize = 20)
    plt.xlabel('Monday     Tuesday     Wednesday     Thursday     Friday     Saturday     Sunday' , fontsize = 20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


# In[75]:


accident_by_day()


# #### Distribution of Accidents Per Day of hour

# In[76]:


def accidents_per_hr():
    accidents_hr = accident_master.groupby(accident_master['date_time'].dt.hour).count().date_time
    accidents_hr.plot(kind = 'bar', figsize = (15, 10), color = 'darkolivegreen', alpha = 0.75)

    plt.title("Accident Distribution Across Hour of Day", fontsize = 25)
    plt.ylabel('Number of Accidents', fontsize = 20)
    plt.xlabel('Hour of Day 00:00-23:00', fontsize = 20)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    


# In[77]:


accidents_per_hr()


# #### Accident variation between Fridays (max) and Sundays (min)

# In[78]:


def friday_sunday_accidents():
    accidents = accident_master.groupby(accident_master['date_time'].dt.date).count().date_time
    accidents.plot(figsize = (30, 10), color = 'mediumseagreen')

    friday = accident_master.groupby(accident_master[accident_master['date_time'].dt.dayofweek == 4].date_time.dt.date).count().date_time
    plt.scatter(friday.index, friday, color = 'red', label = 'Frdiay')

    sunday = accident_master.groupby(accident_master[accident_master['date_time'].dt.dayofweek == 6].date_time.dt.date).count().date_time
    plt.scatter(sunday.index, sunday, color = 'blue', label = 'Sunday')

    plt.title("Accident variation between Friday's & Sunday's in the UK 2016-2021", fontsize = 40)
    plt.ylabel('Number of Accidents', fontsize = 30)
    plt.xlabel('Date', fontsize = 30)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(loc = 1, prop = {'size':20})


# In[79]:


friday_sunday_accidents()


# #### Slight - Serious - Fatal Injuries

# In[81]:


def injury_by_type():
    # Series with number of mild injuries and serious injuries
    injury = accident_master[['accident_severity']].value_counts()

    # Pie plot with the percentage of victims with slight, serious and fatal injuries
    explode = (0.03, 0.03, 0)
    injury.plot(kind = 'pie',figsize = (10,10), colors = ['forestgreen','orange','orangered'], explode = explode, labels = None, autopct = '%1.1f%%', pctdistance = 1.2, fontsize = 18)

    # Title and legend
    plt.legend(title = "Injury Level", labels = ['Slight', 'Serious', 'Fatal'],bbox_to_anchor = (0.85, 0.95), loc='upper left', borderaxespad=0, fontsize = 15)
    plt.title('Injury Type from UK Road Accidents', fontsize = 25)
    plt.ylabel('')


# #### Rate of Injuries per Hour of Day

# In[82]:


def rate_of_injury_per_hour():
    accident_severity = accident_master['accident_severity']
    # Number of fatal injuries per day of the week
    accidents_fatal = accident_master[accident_master['accident_severity'] == 3].groupby(accident_master['date_time'].dt.hour).sum().accident_severity
    # Percentage of fatal injuries per day of the week
    rate_fatal = accidents_fatal/accident_severity.sum()

    # Number of serious injuries per day of the week
    accidents_serious = accident_master[accident_master['accident_severity'] == 2].groupby(accident_master['date_time'].dt.hour).sum().accident_severity
    # Percentage of seriois injuries per day of the week
    rate_serious = accidents_serious/accident_severity.sum()

    # Number of slight injuries per day of the week
    accidents_slight = accident_master[accident_master['accident_severity'] == 1].groupby(accident_master['date_time'].dt.hour).sum().accident_severity
    # Percentage of slight injuries per day of the week
    rate_slight = accidents_slight/accident_severity.sum()

    # Combine both series as a dataframe in order to plot them as a side by side bar
    rates = pd.DataFrame({'Fatal Injures':rate_fatal,'Serious Injuries':rate_serious,'Slight Injuries':rate_slight})
    rates.plot(kind = 'bar', figsize = (20,12), width = 0.5, color = ['orangered', 'orange', 'forestgreen'], alpha = 0.75)

    # Title and labels
    plt.title('Rate of Injury Type by Hour of Day',fontsize = 30)
    plt.xlabel('Hour 00:00-23:00',fontsize = 20)
    plt.ylabel('Percentage of Total Collisions (%)',fontsize = 20)
    plt.xticks(fontsize = 18, rotation = 0)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 20)


# In[83]:


rate_of_injury_per_hour()


# ### The odds of being in a greater severity accident by speed limit
# 

# #### Checking that the multicollinearity assumption is met

# In[85]:


df_speedlimit_multicollinearity_test = accident_master.iloc[:,70:76]
df_speedlimit_multicollinearity_test_print = df_speedlimit_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_speedlimit_multicollinearity_test_print)


# 30 mph exhibits the greatest levels of multicollinearity with the other categories. Correlation of -0.40 exists between 30 mph and 20 mph, -0.36 against 40 mph, -0.25 against 50 mph and -0.45 against 60 mph. As such, the 30mph category is not used in the OLR analysis, and instead acts as the reference variable.

# #### OLR Model

# In[86]:


def speedlimit_severity_logistic_ordinal_regression():
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['30_mph', 
                                                       '40_mph', '50_mph' ,'60_mph', '70_mph'
                                                       ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[87]:


speedlimit_severity_logistic_ordinal_regression()


# #### Distribution of accidents within Speed Limit Zones

# In[88]:


accident_speed_limit= accident_master[accident_master['speed_limit'] != -1] #discards accidents with no road speed limit data


# In[89]:


def accident_in_speed_limit():
    acc_speed_limit = accident_speed_limit.speed_limit.value_counts()
    labels = ['30mph', '60mph', '40mph', '70mph', '50mph', '20mph']

    explode = (0, 0.05, 0.06, 0.07, 0.08, 0.09)
    fig = plt.figure(figsize = (10,10))
    plt.pie(acc_speed_limit.values, labels=labels, autopct = '%.1f', pctdistance = 0.8, labeldistance = 1.08, explode = explode, shadow = False, startangle = 160, textprops = {'fontsize': 20})
    plt.axis('equal')
    plt.figtext(0.5, 0.93, 'Accidents in a given speed limit (%)', fontsize = 25, ha = 'center')
    plt.show()


# In[90]:


accident_in_speed_limit()


# ### The odds of being in a greater severity accident by road class
# 

# #### Checking that the multicollinearity assumption is met

# In[91]:


df_roadclass_multicollinearity_test = accident_master.iloc[:,64:69]
df_roadclass_multicollinearity_test_print = df_roadclass_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_roadclass_multicollinearity_test_print)


# The A-road class category exhibits the greatest levels of multicollinearity with the other categories. Correlation of -0.16 exists against the motorway category, -0.05 against A (M) roads, -0.33 against B roads, and -0.21 against C roads. As such, the A-road class category is not used in the OLR analysis, and instead acts as the reference variable.

# **Ordinal Logistic Regression** is used to understand how road class affects the odds of being in an accident of greater severity (i.e. serious or fatal, rather than slight). 
# 
# A-roads were chosen as the reference variable for an intuitive comparison baseline, and unclassified roads were removed as they do not infer any meaningful information for this analysis.
# 
# Again, a **logit** ordinal regression method was employed rather than a probit method, because this allows for more intuitive analysis, and because the choice is negligible for a large sample size.

# #### OLR Model

# In[92]:


def road_class_severity_logistic_ordinal_regression():
# Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['A_road_road_class', 'A_M_road_class', 
                                                   'B_road_class', 'C_road_class'
                                                   ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[93]:


road_class_severity_logistic_ordinal_regression()


# ### How does the occurence of road accidents vary with road class?
# 

# In[94]:


def road_class_accident_occurences():
    t = len(accident_master[accident_master.Motorway_road_class == 1])
    u = len(accident_master[accident_master.A_M_road_class == 1])
    v = len(accident_master[accident_master.A_road_road_class == 1])
    w = len(accident_master[accident_master.B_road_class == 1])
    y = len(accident_master[accident_master.C_road_class == 1])
    
    road_types_df = pd.DataFrame({'road_type': ['Motorway_road_class', 'A_M_road_class', 'A_road_road_class',\
                                           'B_road_class', 'C_road_class'
                                         ],\
                       'accident_count': [t, u, v, w, y]})

    x = road_types_df.road_type
    y = road_types_df.accident_count
    fig,ax = plt.subplots(figsize=(15, 10))
    plt.bar(x, y)
    labels = ['Motorway', 'A (M)','A', 'B',
             'C']
    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=8)
    plt.title('Occurence of road accidents by road class (2017-2021)', fontsize=25)
    plt.xlabel('Road class', fontsize=18)
    plt.ylabel('Accident Occurences', fontsize=18)

    plt.show()


# In[95]:


road_class_accident_occurences()


# ### The odds of being in a greater severity accident by age

# #### Checking that the multicollinearity assumption is met

# In[96]:


df_age_multicollinearity_test = accident_master.iloc[:,20:31]
df_age_multicollinearity_test_print = df_age_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_age_multicollinearity_test_print)


# The age (26-35) category exhibits the greatest levels of multicollinearity with the other categories. Correlation of -0.12 exists against the age (16-20) category, -0.13 against the age (21-25) cateogry, -0.15 against age (36-45), and -0.14 against age (46-55). As such, the age (26-35) category is not used in the OLR analysis, and instead acts as the reference variable.

# #### OLR Model

# In[97]:


def age_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['age_16_20', 'age_26_35', 
                                                       'age_36_45', 'age_46_55', 'age_56_65', 'age_66_75', 'age_75plus'
                                                       ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())
    
    # Creating the proportional odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[98]:


age_severity_logistic_ordinal_regression()


# #### Age of People Involved in accidents

# In[99]:


a = len(accident_master[accident_master.age_0_5 == 1])
b = len(accident_master[accident_master.age_6_10 == 1])
c = len(accident_master[accident_master.age_11_15 == 1])
d = len(accident_master[accident_master.age_16_20 == 1])
e = len(accident_master[accident_master.age_21_25 == 1]) 
f = len(accident_master[accident_master.age_26_35 == 1])
g = len(accident_master[accident_master.age_36_45 == 1])
h = len(accident_master[accident_master.age_46_55 == 1])
i = len(accident_master[accident_master.age_56_65 == 1])
j = len(accident_master[accident_master.age_66_75 == 1])
k = len(accident_master[accident_master.age_75plus == 1])

age_bands_df = pd.DataFrame({'driver_age_band': ['0-5', '6-10', '11-15',\
                                       '16-20', '21-25', '26-35', '36-45',\
                                       '46-55', '56-65', '66-75', '75+'],\
                   'accident_cont': [a, b, c, d, e, f, g, h, i, j, k]})


# In[100]:


def accident_age_range():
    x = age_bands_df.driver_age_band
    y = age_bands_df.accident_cont
    fig,ax = plt.subplots(figsize=(15, 10))
    plt.bar(x, y)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Age Range of People in Road Accidents', fontsize=25)
    plt.xlabel('Driver Age Groups', fontsize=18)
    plt.ylabel('Accident Occurences', fontsize=18)

    plt.show()


# In[101]:


accident_age_range()


# ## Urban or rural area

# ### The odds of being in a greater severity accident by area type

# #### Checking that the multicollinearity assumption is met

# In[102]:


df_urban_rural_multicollinearity_test = accident_master.iloc[:,76:78]
df_urban_rural_multicollinearity_test_print = df_urban_rural_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_urban_rural_multicollinearity_test_print)


# As the categories of urban and rural areas involved in accidents are effectively measuring the same thing (the proportion of accidents in an urban or rural area), complete multicollinearity is exhibited. As such, only 'Urban_area' will be included in the OLR model. This means that 'Rural_area' can be used as the reference variable to compare odds against.

# #### OLR Model

# In[103]:


def urban_or_rural_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Urban_area'
                                                           ]], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the proportional odds ratios (i.e. the odds of being in an accident of greater severity (i.e. serious
    # or fatal, rather than slight, as a result of the road type))
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# In[104]:


urban_or_rural_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by overall driving conditions

# #### Checking that the multicollinearity assumption is met

# In[105]:


df_driving_conditions_multicollinearity_test = accident_master.iloc[:,43:78]
df_driving_conditions_multicollinearity_test_print = df_driving_conditions_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_driving_conditions_multicollinearity_test_print)


# Multicollinearity amongst the driving conditions variables is low and so all factors remain in the analysis.

# #### OLR Model

# In[106]:


def driving_conditions_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['Darkness_lights_lit','Darkness_lights_unlit','Darkness_no_lighting','Fine_no_high_winds',
    'Raining_no_high_winds','Snowing_no_high_winds','Fine_and_high_winds','Raining_and_high_winds',
    'Snowing_and_high_winds','Fog_or_mist', 'Dry_road','Wet_or_damp_road','Snow_road','Frost_or_ice_road', 
    'Motorway_road_class','A_M_road_class','A_road_road_class','B_road_class','C_road_class','Urban_area',
    '20_mph','30_mph','40_mph','50_mph','60_mph', '70_mph']], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the odds ratios
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# ### The odds of being in a greater severity accident by overall demographics

# #### Checking that the multicollinearity assumption is met

# In[107]:


df_demographics_multicollinearity_test = accident_master.iloc[:,20:34]
df_demographics_multicollinearity_test_print = df_demographics_multicollinearity_test.corr()
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    display(df_demographics_multicollinearity_test_print)


# Multicollinearity amongst the demographic variables is low and so all factors remain in the analysis.

# #### OLR Model

# In[108]:


def demographics_severity_logistic_ordinal_regression():
    # Creating the model
    mod_log = OrderedModel(accident_master['accident_severity_OLR'],accident_master[['age_16_20','age_21_25','age_26_35','age_36_45',
    'age_46_55','age_56_65','age_66_75','age_75plus','sex_male']], distr='logit')
    # Fitting the model and printing the summary
    res_log = mod_log.fit(method='bfgs', disp=False)
    print(res_log.summary())

    # Creating the odds ratios (i.e. the odds of being in an accident of greater severity
    odds_ratios = pd.DataFrame(
        {"OR": res_log.params, "Lower CI": res_log.conf_int()[0], "Upper CI": res_log.conf_int()[1],})
    odds_ratios = np.exp(odds_ratios)
    print(odds_ratios)


# # 3. Project Outcome

# ## 3.1 Overview of Results
# 
# The high level spatial distribution of the data shows that the majority of accidents occurred in urban areas between 2016 and 2021. The three ONS districts with the highest number of accidents were Birmingham (11,900), Leeds (7,110) and Westminster (6,898).
# 
# People aged 26 to 35 were the most involved in accidents between 2016 and 2021. 66% of the drivers involved were males and 39% were females.
# 
# Ordinal Logistic Regression indicates that men have higher odds than women of being in an accident of greater severity (serious or fatal, rather than slight). Results also suggest that drivers aged 75 or older have the highest odds of an accident of greater severity amongst age groups.
# 
# Ordinal Logistic Regression indicates that dry road conditions produce the greatest odds of being in an accident of greater severity, whilst foggy conditions are the most impactful weather condition in increasing the odds of being in an accident of greater severity. Results further indicate that driving on roads with a 60 mph speed limit produces the greatest odds amongst all speed limit categories. Finally, the odds of being in an accident of greater severity are highest on B- class roads when compared to all other road classes.
# 

# ## 3.2 Objective 1: Exploring the spatial and temporal distribution of accidents within the UK using the Folium library
# 
# The below interactive map has been created to group accident locations into clusters using the Marker Cluster Object from the Folium library. It depicts vehicle accidents that experienced injuries (slight, serious or severe), where accidents are spatially grouped into clusters. The primary purpose of the map is to inform exploratory geospatial understanding of accident occurrence and guide the formulation of research questions. For example. Urban areas including London, Birmingham, and Manchester experience significantly higher rates of accidents than rural areas.
# 

# In[109]:


accidents_interactive_map() #Please check the interactive map in the HTML File


# The following animated heatmap shows all accident locations in each area for a specific hour. From exploring the visualization and timeline zoomed in on Leeds by specifying the zoom level we can observe how the number of accidents increases from 07:00-08:00 hours and again from 14:00-18:00 hours. Before, in between and after these observed intervals, the heat map highlights a lower frequency of accident occurrences within the Leeds region.
# 
# 
# The map also shows that during the night and early hours of the morning, accidents mostly occur on connecting roads between urban agglomerations. However, starting from 7:00  to 21:00, accidents occur more often within urban agglomerations. 
# 

# In[110]:


heat_interactive_map() #Please check the interactive map in the HTML File


# As shown in the below plot, the number of vehicle accidents in the UK decrease at the weekend. As weekdays progress, the number of observed collisions increase, with Friday experiencing the highest number of total accidents (~90,000). Conversly, accident rates at weekends are significantly reduced.
# 
# An explination for this trend is likely due to vehicle useage trends, where people use their vehicles less frequently on the weekends as workers are not required to commute to work. Conversely, Friday's may experience such high rates of accidents due to tiredness, rushing to get home, and a lack of attention to surroundings after a busy work week. 

# In[111]:


accident_by_day()


# As previously observed, Friday's typically experience the highest frequency of incidents, while Sunday's experience the fewest. The following graph demonstrates the varation of accidents occurence between Fridays and Sudnays from 2016 to 2021.  As a rule, we can observe total accidents on Friday's with the red markers, while Sunday's are represented by blue markers.
# 
# Using these two days to aid with this visualisation, we can see that there are between 200 and 500 accidents per day from 2017 to early 2020, whereas this range decreases to between 100 and 400 accidents per day from early 2020. 
# 
# This visualisation is particularly interesting as we can see clear fluctuations between 2020 and 2022, which is most likely due to the encorcement of COVID-19 lockdowns within the UK,: 
# - 1st lockdown = March 2020-June 2020
# - 2nd lockdown = November 2020
# - 3rd lockdown = January 2021-March 2021 
# 
# As a consequence of these lockdowns, road networks experienced significantly reduced useage as people opted to shop online, work from home and isolate from the virus. 

# In[112]:


friday_sunday_accidents()


# The following graph shows the percentage of injuries according the hour which highlights that accidents tend to be more severe in late-evenings and night. It clearly shows that accidents tend to be more severe during night and late-evening.

# In[113]:


rate_of_injury_per_hour()


# ## 3.3 Objective 2: Using the Matplotlib package, visualise how accident occurrences vary with driving and demographic conditions
# 
# The graph below demonstrates the level of involvment of each age group in road accidents. It is clear that people aged between 26 and 35 are the most invloved in road accidents between 2016 and 2021.

# In[114]:


accident_age_range()


# The following visualisation highlights some interesting information. As you can see, more than half of all collisions were on roads where the speed limit was 30 miles per hour (58.8%). We were expecting more collision to be on Motorways (70mph) or A-Roads (60mph).
# 
# However, this pattern does seem to make sense as zones with lower speed limits typically experience increased congestion, stop signs, changing lanes, traffic lights etc.

# In[115]:


accident_in_speed_limit()


# In terms of injury severity, only 1.4% of injuries recored between 2016 and 2021 were fatal. However, the plot does highlight that 19.2% of injuries recored were serious.
# 
# Although the majority (79.4%) of injuries were slight, it would be interesting to analyse under what circumstances (date, time, location) serious injuries are more frequent. The following pie chart demostrates the percentages of each injury type.

# In[116]:


injury_by_type()


# The following graph shows the percentage of males and females involved in road accidents between 2016 and 2021. We examine the odds of being in an accident of greater severity by gender later under Objective 4.

# In[117]:


gender_accident_occurences()


# ## 3.4 Objective 3: Statistically assess, using the statsmodels package, how driving conditions affect the odds of being in an accident of greater severity
# 
# Ordinal Logistic Regression was used to understand how various driving conditions (light, weather, road class, speed limit) are associated with the odds of being in an accident of greater severity (serious or fatal, rather than slight). Odds are variable among speed limits, with the odds of being in an accident of greater severity at 50 mph 1.34 times higher than if travelling at 30 mph, and 1.95 times higher at 60 mph. Interestingly, the odds at 70 mph are lower than those at 60 mph, with only 1.16 times higher odds compared to 30 mph. Moreover, results indicate that the odds of a greater severity accident on a B- class road are only 1.13 times higher than an A-road, whilst motorways have 0.83 times higher odds.
# 
# Results also indicate that complete darkness increases the odds of being in an accident of greater severity by 1.94 times when compared to daylight, whilst when lights are lit, these odds fall to 1.13 times higher. Moreover, fog conditions have the largest impact of all weather conditions, making the odds an accident of greater severity 1.35 times higher than in fine and non-windy conditions.
# 
# Having ascertained how each categorical value within a driving condition category affects the odds of an accident of greater severity, OLR was further implemented to analyse the proportional impacts of road type, speed limit, weather conditions and light conditions on the odds of an accident of greater severity, whilst holding all other variables constant. The results indicate that the two most important factors are dry roads and wet roads- increasing the odds, respectively, of an accident of greater severity by 1.82 and 1.81. One potential reason for these unexpected and conflicting results may be that in dry conditions, drivers perhaps become complacent. These findings are significant at the 95% confidence level, indicating that they are statistically associated with the odds of being in an accident of greater severity.
# 
# In contrast, the lowest odds of being in an accident of greater severity are seen when travelling on a motorway (0.63), a finding that echoes the results of the road class OLR. Overall, this analysis illustrates how the odds of being in an accident of greater severity are variable both within a driving condition category, and when compared to other influential driving condition factors. 
# 

# ### The odds of being in a greater severity accident by weather conditions
# 

# In[118]:


weather_conditions_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by speed limit
#  

# In[119]:


speedlimit_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by road class
# 

# In[120]:


road_class_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by light conditions
# 

# In[121]:


light_conditions_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by overall driving conditions
# 

# In[122]:


driving_conditions_severity_logistic_ordinal_regression()


# ## 3.5 Objective 4: Statistically assess, using the statsmodels package, how demographic  conditions affect the odds of being in an accident of greater severity
# 
# Ordinal Logistic Regression was also used to understand how demographic factors are associated with the odds of being in an accident of greater severity (serious or fatal, rather than slight). Results indicate that, for males, the odds of being in an accident of greater severity are 1.35 times higher than those for females. This suggests that men are more likely to be in an accident of greater severity. Comparisons of the odds of accidents of a greater severity amongst age groups yield less clear- cut patterns. When compared to drivers aged 26-35, results indicate that drivers aged 16-20 have 1.16 times higher odds of being in an accident of greater severity. However, drivers aged 21-25 have 0.97 times higher odds when compared to those aged 26-35, and drivers aged 36-45 have 0.90 times higher odds. Such decreasing odds as age bands progress may be a result of average driving experience and length accumulating, making drivers more road safe and thus decreasing the odds of being in an accident of greater severity.
# 
# Interestingly, in the next age band, 46-55, the odds of being in an accident of greater severity are 1.04 times higher than for drivers aged 26-35, while for drivers aged 56-65 these odds increase to 1.19 times higher. As age band increases, so to do the odds, with drivers aged 66-75 having odds of being in an accident of greater severity 1.29 times higher than those aged 26-35, and drivers aged 75 or over experiencing odds 1.44 times higher than drivers aged 26-35.
# 
# Thus, from the OLR results, it is clear that for new drivers (assuming aged 17- the age that one can achieve their driving licence) the odds of being in an accident of greater severity are initially high, but these reduce as age progresses until around 45 years old, before increasing again to reach a peak in the 75 or older age band. These findings contrast with the number of accident occurrences amongst age groups, as shown in …., where drivers aged 26-35 experience the greatest number of accident occurrences. As such, the results of this OLR analysis illustrate how the number of accidents does not accurately reflect the severity of accidents experienced.
# 
# OLR was further utilised to analyse the proportional impacts of driver age and sex upon the odds of an accident of greater severity, whilst holding all other variables constant. Results indicate that being a male increases the odds of being in an accident of a greater severity to the largest extent (1.42), whilst holding all other variables constant. Drivers aged 75 or over experience the second largest increase (1.30), whilst drivers aged 66-75 experience 1.16 times higher odds of being in an accident of greater severity. All results were significant at the 95% confidence interval. In summary, this analysis illustrates how the odds of being in an accident of greater severity are variable both within a demographic category, and when compared to other influential demographic factors. 

# ### The odds of being in a greater severity accident by gender
# 

# In[123]:


gender_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by age
# 

# In[124]:


age_severity_logistic_ordinal_regression()


# ### The odds of being in a greater severity accident by overall demographics
# 

# In[125]:


demographics_severity_logistic_ordinal_regression()


# # 4. Conclusion (5 marks)
# 
# ## 4.1 Achievements
# In conclusion, this project has visualised the spatial and temporal distribution of recorded accidents between 2017-2021. It has also identified trends in accident occurrence as driving conditions and demographic factors vary. Importantly, the project has also demonstrated how the odds of being in an accident of a greater severity are variable with different driving conditions and demographic considerations, and found that dry roads, complete darkness, and B- road type are the most influential factors in increasing the odds of a greater severity accident. Thus, the project successfully met the initial objectives.
# ## 4.2 Limitations
# 
# This project has various limitations, the most notable being an inability to predict the goodness-of-fit of the Ordinal Logistic Regression models produced. As OrderedModel is a relatively new addition to the statsmodel package, it does not yet have a method to conduct goodness-of-fits tests. One appropriate technique would be using a Brant test, but this could not be conducted in Python. As such, we cannot be certain as to how well the regression models perform, and so their results must be treated with a degree of caution. Moreover, OrderedModel does not have a method for assessing proportional odds (Assumption 4), meaning that confidence in the reliability of the model results is limited. 
# 
# Additionally, missing data in the original datasets meant the full scope of accidents recorded could not be analysed, potentially reducing the applicability and reproducibility of the results. Importantly, the data contained the non-typical period of the COVID-19 lockdowns, the impact of which upon mobility, traffic, and accident occurrences potentially skewed our analyses. 
# 
# ## 4.3 Future Work
# 
# Future work would likely explore additional potentially interesting variables in the original dataset, such as engine capacity or deprivation index of driver, over a more substantive time period, and would incorporate road usage data to understand whether accident occurrence and severity are statistically associated with the number of vehicles on the road network. Statistical tests such as the Mann-Whitney U test would provide an insight into whether differences in median accident occurrence between each value of categorical variable are statistically significant.
# 
# Modelling techniques such as random forests may improve the accuracy and reliability of this project, while taking a classification- based approach would allow this work to be used to forecast future trends in road traffic accident occurrence and severity, potentially allowing for better traffic management and resource distribution across road networks, minimising the risk to those using the road network. 
