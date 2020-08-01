using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MovieRecommender
{
    class Program
    {
        //Path donde se encuentra el archivo con todas las peliculas as well as the other files.
        public static string DatasetsRelativePath = @"../../../Data";
        public static string moviesDataPath = @"../../../Data/recommendation-movies.csv";

        static void Main(string[] args)
        {
            int userID;
            Console.WriteLine("Type a user Id, and i am going to give you the top 5 recommended movies for him or her\n using matrix factorization with ML.NET.");
            userID=Convert.ToInt32(Console.ReadLine());//leo el ID del usuario.

            //MlContext es el punto inicial, contiene todo lo necesario para utilizar algoritmos de ML
            MLContext mlContext = new MLContext();
            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);
            //Evalua
            EvaluateModel(mlContext, testDataView, model);
            //Guarda
            SaveModel(mlContext, trainingDataView.Schema, model);


            //Entrada para hacer la prediccion.
            var testInput = new MovieRating { userId = userID, movieId = 1 };
            //Creas el PredictinoEngine
            var predictionengine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

           //Predice y recomienda las 5 mejores peliculas basada en los scores segun  la prediccion hecha.
            Console.WriteLine("Calculating the top 5 movies for user "+userID.ToString());
            Console.WriteLine($"=============== Top 5 Recommended movies for user, {userID.ToString()}, ===============");
            var top5 = (from m in Movies.All
                        let p = predictionengine.Predict(testInput)
                        orderby p.Score descending
                        select (MovieId: m.ID, Score: p.Score)).Take(5);

            foreach (var t in top5)
                Console.WriteLine($" Movie Score:{t.Score}\tMovie Name: {Movies.Get(t.MovieId)?.Title}");
        }


        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }


        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }


        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

    }
}
