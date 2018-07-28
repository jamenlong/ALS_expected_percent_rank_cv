# This function was initially created for the Million Songs Echo Nest Taste Profile dataset which had 3 columns: userId, songId,
# and num_plays. The column num_plays was used as implicit ratings with the ALS algorithm.

def ROEM(predictions):
  #Creates predictions table that can be queried
  predictions.createOrReplaceTempView("predictions") 
  
  #Sum of total number of plays of all songs
  denominator = predictions.groupBy().sum("num_plays").collect()[0][0]
  
  #Calculating rankings of songs predictions by user
  spark.sql("SELECT userID, num_plays, PERCENT_RANK() OVER (PARTITION BY userId ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
  
  #Multiplies the rank of each song by the number of plays for each user
  #and adds the products together
  numerator = spark.sql('SELECT SUM(num_plays * rank) FROM rankings').collect()[0][0]
                         
  return numerator / denominator
