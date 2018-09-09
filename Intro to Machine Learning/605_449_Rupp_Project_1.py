import pandas as panda

#### RETRIEVE ALL DATA FROM WEB ####

# Add the source archive url to the sub sets to make full paths
source_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
breast_cancer_url = source_url + "breast-cancer-wisconsin/wdbc.data"
soybean_url = source_url + "soybean/soybean-small.data"
glass_url = source_url + "glass/glass.data"
iris_url = source_url + "iris/iris.data"
vote_url = source_url + "voting-records/house-votes-84.data"

#Define the columns for the data
glass_cols = [
  "Id"
  , "Refractice_Index"
  , "Sodium"
  , "Magnesium"
  , "Aluminum"
  , "Silicon"
  , "Potassium"
  , "Calcium"
  , "Barium"
  , "Iron"
  , "Type"
]

#retrieve the data from the source
glass_set = panda.read_csv(
  glass_url
  , names = glass_cols
)

#Define the columns for the data
iris_cols = [
  "Sepal_Length"
  , "Sepal_Width"
  , "Petal_Length"
  , "Petal_Width"
  , "Class"
]

#retrieve the data from the source
iris_set = panda.read_csv(
  iris_url
  , names = iris_cols
)

#Define the columns for the data
vote_cols = [
  "Class"
  , "Handicapped_Infants"
  , "Water_Project_Cost_Sharing"
  , "Adoption_Of_Budget_Resolution"
  , "Physician_Fee_Freeze"
  , "El_Salvador_Aid"
  , "Religious_Groups_In_Schools"
  , "Anti_Satellite_Test_Ban"
  , "Aid_To_Nicaraguan_Contras"
  , "Mx_Missile"
  , "Immigration"
  , "Synfuels_Corporation_Cutback"
  , "Education_Spending"
  , "Superfund_Right_To_Sue"
  , "Crime"
  , "Duty_Free_Exports"
  , "Export_Administration_Act_South_Africa"
]

#retrieve the data from the source
vote_set = panda.read_csv(
  vote_url
  , names = vote_cols
  , na_values = '?'
)

































































































