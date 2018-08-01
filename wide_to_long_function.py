# This function was originally provided for use in converting the MovieLens ratings dataframe from a conventional/"wide"
# dataframe to a row-based/"long" dataframe.

from pyspark.sql.functions import array, col, explode, lit, struct

def to_long(df, by = ["userId"]): # "by" is the column by which you want the final output dataframe to be grouped by

    cols = [c for c in df.columns if c not in by] 
    
    kvs = explode(array([struct(lit(c).alias("movieId"), col(c).alias("rating")) for c in cols])).alias("kvs") 
    
    long_df = df.select(by + [kvs]).select(by + ["kvs.movieId", "kvs.rating"]).filter("rating IS NOT NULL")
    # Excluding null ratings values since ALS in Pyspark doesn't want blank/null values
             
    return long_df
