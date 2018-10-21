
setwd("B:/Users/pathr/Documents/Education/JHU/605.449.81.FA18 Introduction to Machine Learning/DATA/Project 4")

#### ABALONE ####
data <- read.csv("prep_abalone.DATA", as.is = TRUE)

new_data <- data.frame(CLASS = data[, 1])
new_data$LENGTH         <- ifelse( data$LENGTH >= median(data$LENGTH)					, 'H', 'L' )
new_data$DIAMETER       <- ifelse( data$DIAMETER >= median(data$DIAMETER)				, 'H', 'L' )
new_data$HEIGHT         <- ifelse( data$HEIGHT >= median(data$HEIGHT)					, 'H', 'L' )
new_data$WHOLE_WEIGHT   <- ifelse( data$WHOLE_WEIGHT >= median(data$WHOLE_WEIGHT)		, 'H', 'L' )
new_data$SHUCKED_WEIGHT <- ifelse( data$SHUCKED_WEIGHT >= median(data$SHUCKED_WEIGHT)	, 'H', 'L' )
new_data$VISCERA_WEIGHT <- ifelse( data$VISCERA_WEIGHT >= median(data$VISCERA_WEIGHT)	, 'H', 'L' )
new_data$SHELL_WEIGHT   <- ifelse( data$SHELL_WEIGHT >= median(data$SHELL_WEIGHT)		, 'H', 'L' )
new_data$RINGS          <- ifelse( data$RINGS >= median(data$RINGS)						, 'H', 'L' )

write.csv(
  new_data
  , "in_abalone.csv"
  , row.names = F
)

rm(list = ls())

#### IMAGE SEGMENTATION ####
data      <- read.csv("prep_segmentation.DATA", as.is = TRUE)
new_data  <- data.frame(CLASS = data[, 1])
new_data$REGION.CENTROID.COL  <- as.character( data$REGION.CENTROID.COL )
new_data$REGION.CENTROID.ROW  <- as.character( data$REGION.CENTROID.ROW )
new_data$REGION.PIXEL.COUNT   <- as.character( data$REGION.PIXEL.COUNT )
new_data$SHORT.LINE.DENSITY.5 <- ifelse( data$SHORT.LINE.DENSITY.5 >= median(data$SHORT.LINE.DENSITY.5)	, 'H', 'L' )
new_data$SHORT.LINE.DENSITY.2 <- ifelse( data$SHORT.LINE.DENSITY.2 >= median(data$SHORT.LINE.DENSITY.2)	, 'H', 'L' )
new_data$VEDGE.MEAN           <- ifelse( data$VEDGE.MEAN >= median(data$VEDGE.MEAN)						, 'H', 'L' )
new_data$VEDGE.SD             <- ifelse( data$VEDGE.SD >= median(data$VEDGE.SD)						  , 'H', 'L' )
new_data$RAWBLUE.MEAN         <- ifelse( data$RAWBLUE.MEAN >= median(data$RAWBLUE.MEAN)					, 'H', 'L' )
new_data$RAWGREEN.MEAN        <- ifelse( data$RAWGREEN.MEAN >= median(data$RAWGREEN.MEAN)					, 'H', 'L' )
new_data$EXRED.MEAN           <- ifelse( data$EXRED.MEAN >= median(data$EXRED.MEAN)						, 'H', 'L' )
new_data$EXBLUE.MEAN          <- ifelse( data$EXBLUE.MEAN >= median(data$EXBLUE.MEAN)						, 'H', 'L' )
new_data$EXGREEN.MEAN         <- ifelse( data$EXGREEN.MEAN >= median(data$EXGREEN.MEAN)					, 'H', 'L' )
new_data$VALUE.MEAN           <- ifelse( data$VALUE.MEAN >= median(data$VALUE.MEAN)						, 'H', 'L' )
new_data$SATURATION.MEAN      <- ifelse( data$SATURATION.MEAN >= median(data$SATURATION.MEAN)				, 'H', 'L' )
new_data$HUE.MEAN             <- ifelse( data$HUE.MEAN >= median(data$HUE.MEAN)						  , 'H', 'L' )

write.csv(
  new_data
  , "in_segmentation.csv"
  , row.names = F
)

rm(list = ls())

#### IMAGE SEGMENTATION ####
data      <- read.csv("prep_car.DATA", as.is = TRUE)
new_data  <- cbind(data.frame(CLASS = data$ACCEPTANCE_LVL), data[, -ncol(data)])


write.csv(
  new_data
  , "in_car.csv"
  , row.names = F
)
























