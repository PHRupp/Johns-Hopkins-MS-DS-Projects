#######################################################################
# Project #1
# Author: Patrick H. Rupp
# Course: 605.449 - Introduction to Machine Learning
# Due: September 9, 2018

########################### RETRIEVE DATA #############################

# Set global environment options
options(stringAsFactors = FALSE)

# Add the source archive url to the sub sets to make full paths
source_url <- "https://archive.ics.uci.edu/ml/datasets/"
breast_cancer_url <- paste0(source_url, "Breast+Cancer+Wisconsin+%28Original%29")
soybean_url <- paste0(source_url, "Soybean+%28Small%29")
glass_url <- paste0(source_url, "Glass+Identification")
iris_url <- paste0(source_url, "Iris")
vote_url <- paste0(source_url, "Congressional+Voting+Records")

#https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data
vote_cols <- c(
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
)

#retrieve the data from the source
vote_set <- read.csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
  , as.is = TRUE
  , na.strings = "?"
  , col.names = vote_cols
  )

############################## END CODE ###############################