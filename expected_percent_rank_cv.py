def expected_percent_rank_cv(ratings_df, userCol = "userId", itemCol = "songId", ratingCol = "num_plays", ranks = [10, 50, 100, 150, 200], maxIters = [10, 25, 50, 100, 200, 400], regParams = [.05, .1, .15], alphas = [10, 40, 80, 100]):

  #Originally run on a subset of the Echo Next Taste Profile dataset found her:
  #https://labrosa.ee.columbia.edu/millionsong/tasteprofile

  from pyspark.sql.functions import rand
  from pyspark.ml.recommendation import ALS

  ratings_df = ratings_df.orderBy(rand()) #Shuffling to ensure randomness

  #Building train and validation test sets
  train, validate = ratings_df.randomSplit([0.8, 0.2], seed = 0)

  #Building 5 folds within the training set. The use of seeds will likely cause some random overlap between the folds.
  #To ensure that users with few ratings don't all get sent to the test set, you can filter out those with fewer than
  #a given threshold of ratings. 20 was used and found to be effective.
  train1, test1 = train.randomSplit([0.8, 0.2], seed = 1)
  train2, test2 = train.randomSplit([0.8, 0.2], seed = 15)
  train3, test3 = train.randomSplit([0.8, 0.2], seed = 388)
  train4, test4 = train.randomSplit([0.8, 0.2], seed = 217)
  train5, test5 = train.randomSplit([0.8, 0.2], seed = 686)

  #Creating variables that will be replaced by the best model's hyperparameters for subsequent printing
  best_validation_performance = 9999999999999
  best_rank = 0
  best_maxIter = 0
  best_regParam = 0
  best_alpha = 0
  best_model = 0
  best_predictions = 0

  #Looping through each combindation of hyperparameters to ensure all combinations are tested.
  for r in ranks:
    for mi in maxIters:
      for rp in regParams:
        for a in alphas:
          #Create ALS model
          als = ALS(rank = r, maxIter = mi, regParam = rp, alpha = a, userCol=userCol, itemCol=itemCol, ratingCol=ratingCol,
                    coldStartStrategy="drop", nonnegative = True, implicitPrefs = True)

          #Fit model to each fold in the training set
          model1 = als.fit(train1)
          model2 = als.fit(train2)
          model3 = als.fit(train3)
          model4 = als.fit(train4)
          model5 = als.fit(train5)

          #Generating model's predictions for each fold in the test set
          predictions1 = model1.transform(test1)
          predictions2 = model2.transform(test2)
          predictions3 = model3.transform(test3)
          predictions4 = model4.transform(test4)
          predictions5 = model5.transform(test5)

          #Expected percentile rank error metric function
          def exp_perc_rank(predictions):
              #Creates table that can be queried
              predictions.createOrReplaceTempView("predictions")

              #Sum of total number of plays of all songs
              denominator = predictions.groupBy().sum("num_plays").collect()[0][0]

              #Calculating rankings of songs predictions by user
              spark.sql("SELECT userID, num_plays, PERCENT_RANK() OVER (PARTITION BY userId ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")

              #Multiplies the rank of each song by the number of plays and adds the products together
              numerator = spark.sql('SELECT SUM(num_plays * rank) FROM rankings').collect()[0][0]

              performance = numerator/denominator

              return performance

          #Calculating expected percentile rank error metric for the model on each fold's prediction set
          performance1 = exp_perc_rank(predictions1)
          performance2 = exp_perc_rank(predictions2)
          performance3 = exp_perc_rank(predictions3)
          performance4 = exp_perc_rank(predictions4)
          performance5 = exp_perc_rank(predictions5)

          #Printing the model's performance on each fold
          print ("Model Parameters: ")("Rank:"), r, ("  MaxIter:"), mi, ("RegParam:"), rp, ("Alpha: "), a
#           print("Test Percent Rank Errors: "), performance1, performance2, performance3, performance4, performance5
          print("Test Percent Rank Errors: "), performance1, performance2, performance3, performance4, performance5

          #Validating the model's performance on the validation set
          validation_model = als.fit(train)
          validation_predictions = validation_model.transform(validate)
          validation_performance = exp_perc_rank(validation_predictions)

          #Printing model's final expected percentile ranking error metric
          print("Validation Percent Rank Error: "), validation_performance
          print(" ")

          #Filling in final hyperparameters with those of the best-performing model
          if validation_performance < best_validation_performance:
            best_validation_performance = validation_performance
            best_rank = r
            best_maxIter = mi
            best_regParam = rp
            best_alpha = a
            best_model = validation_model
            best_predictions = validation_predictions

  #Printing best model's expected percentile rank and hyperparameters
  print ("**Best Model** ")
  print ("  Percent Rank Error: "), best_validation_performance
  print ("  Rank: "), best_rank
  print ("  MaxIter: "), best_maxIter
  print ("  RegParam: "), best_regParam
  print ("  Alpha: "), best_alpha
  return best_model, best_predictions
