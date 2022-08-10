using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using MMIVR.BiosensorFramework.Extensions;
using MMIVR.BiosensorFramework.InputPipeline;
using MMIVR.BiosensorFramework.MachineLearningUtilities;
using static Microsoft.ML.DataOperationsCatalog;

namespace hai_stress_experiments
{ 
    class Program
    {
        static string ModelDir = "";
        static ITransformer BestRegModel;
        static ITransformer BestMultiModel;
        static ITransformer BestBinModel;
        static string[] ExpFiles = { 
            @"C:\GitHub\hai_stress\data\raw_data\hr_data_times1.csv", 
            @"C:\GitHub\hai_stress\data\raw_data\hr_data_times2.csv" };
        static string TopDir = @"C:\GitHub\hai_stress\data\raw_data";
        static int WindowSize = 5;
        static double TrainTestRatio = 0.3;
        static int[] SampleRates = { 96, 32, 32, 32, 64, 4, 4 };
        enum E4Data { ACC, BVP, EDA, HR, IBI, TAGS, TEMP }
        static Random rnd = new Random();

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            List<ExtractedMultiFeatures> MultiFeatureSet = ArceStevensDataset(TopDir);
            
            MultiFeatureSet = Train.TrimFeatureSet(MultiFeatureSet, new List<int>() { 0 });
            MultiFeatureSet = MultiFeatureSet.OrderBy(item => rnd.Next()).ToList();

            var CognitiveLoadSet = CombineLabels(MultiFeatureSet, new List<int>() { 2, 3 }, 0);
            CognitiveLoadSet = CombineLabels(CognitiveLoadSet, new List<int> { 1, 4 }, 1);
            
            //var HAIFeatureSet = Train.TrimFeatureSet(MultiFeatureSet, new List<int>() { 1, 4 });
            List<ExtractedBinFeatures> BinFeatureSet = MultiToBin(CognitiveLoadSet);
            List<ExtractedRegFeatures> RegFeatureSet = MultiToReg(CognitiveLoadSet);

            IDataView MultiClassView = mlContext.Data.LoadFromEnumerable(MultiFeatureSet);
            IDataView BinClassView = mlContext.Data.LoadFromEnumerable(BinFeatureSet);
            IDataView RegClassView = mlContext.Data.LoadFromEnumerable(RegFeatureSet);

            TrainTestData MultiClass = mlContext.Data.TrainTestSplit(MultiClassView, TrainTestRatio);
            TrainTestData BinClass = mlContext.Data.TrainTestSplit(BinClassView, TrainTestRatio);
            TrainTestData RegClass = mlContext.Data.TrainTestSplit(RegClassView, TrainTestRatio);

            List<ITransformer> RegMultiModels = BuildAndTrainRegressionModels(mlContext, RegClass.TrainSet);
            List<ITransformer> BinModels = BuildAndTrainBinClassModels(mlContext, BinClass.TrainSet);
            List<ITransformer> MultiModels = BuildAndTrainMultiClassModels(mlContext, MultiClass.TrainSet);

            double RegRSquared = 0;
            double BinAccuracy = 0;
            double MultiLogLoss = 1.0;

            BestRegModel = null;
            BestMultiModel = null;
            BestBinModel = null;

            DataViewSchema RegModelSchema = null;
            DataViewSchema MultiModelSchema = null;
            DataViewSchema BinModelSchema = null;

            Console.WriteLine("\nRegression Model Metrics");
            foreach (var model in RegMultiModels)
            {
                try
                {
                    IDataView testPred = model.Transform(RegClass.TestSet);
                    RegressionMetrics modelMetrics = mlContext.Regression.Evaluate(testPred);
                    Train.PrintRegMetrics(modelMetrics);
                    if (modelMetrics.RSquared > RegRSquared)
                    {
                        RegModelSchema = testPred.Schema;
                        RegRSquared = modelMetrics.RSquared;
                        BestRegModel = model;
                    }
                }
                catch
                {
                    continue;
                }
            }

            Console.WriteLine("\nBinary Classification Model Metrics");
            foreach (var model in BinModels)
            {
                try
                {
                    IDataView testpred = model.Transform(BinClass.TestSet);
                    BinaryClassificationMetrics BinMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(testpred);
                    Train.PrintBinMetrics(BinMetrics);
                    if (BinMetrics.Accuracy > BinAccuracy)
                    {
                        BinModelSchema = testpred.Schema;
                        BinAccuracy = BinMetrics.Accuracy;
                        BestBinModel = model;
                    }
                }
                catch
                {
                    continue;
                }
            }

            Console.WriteLine("\nMulti-Class Classification Model Metrics");
            foreach (var model in MultiModels)
            {
                try
                {
                    IDataView testpred = model.Transform(MultiClass.TestSet);
                    MulticlassClassificationMetrics MultiMetrics = mlContext.MulticlassClassification.Evaluate(testpred);
                    Train.PrintMultiMetrics(MultiMetrics);
                    if (MultiMetrics.LogLoss < MultiLogLoss)
                    {
                        MultiModelSchema = testpred.Schema;
                        MultiLogLoss = MultiMetrics.LogLoss;
                        BestMultiModel = model;
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    continue;
                }
            }
            if (ModelDir != null)
            {
                mlContext.Model.Save(BestMultiModel, MultiModelSchema, Path.Combine(ModelDir, "MultiModel.zip"));
                mlContext.Model.Save(BestBinModel, BinModelSchema, Path.Combine(ModelDir, "BinModel.zip"));
                mlContext.Model.Save(BestRegModel, RegModelSchema, Path.Combine(ModelDir, "RegModel.zip"));
            }
            Console.ReadLine();
        }
        /// <summary>
        /// Trains multi-class models built-in to Microsoft.ML on the TrainingSet provided.
        /// </summary>
        /// <param name="mlContext">The Microsoft.ML context to perform operations in.</param>
        /// <param name="TrainingSet">The time-series dataset to train the models on.</param>
        /// <returns>List of models that can be used in performance benchmarks.</returns>
        public static List<ITransformer> BuildAndTrainMultiClassModels(MLContext mlContext, IDataView TrainingSet)
        {
            List<ITransformer> Models = new List<ITransformer>();
            var Pipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(ExtractedMultiFeatures.Result))
                .Append(mlContext.Transforms.CopyColumns("Label", "Result"))
                .Append(mlContext.Transforms.Concatenate("Features", "StressFeatures"));

            var APOPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron()));
            var FFOPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest()));
            var FTOPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree()));
            var LBFGSPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()));
            var LBFGSMEPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy());
            var LGBMPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.LightGbm());
            var LSVMOPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm()));
            var SdcaPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy());
            var SGDCOPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdCalibrated()));
            //var SSGDLROPipeline = Pipeline.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression()));
            var KPipeline = Pipeline.Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: 9));

            Models.Add(APOPipeline.Fit(TrainingSet));
            Models.Add(FFOPipeline.Fit(TrainingSet));
            Models.Add(FTOPipeline.Fit(TrainingSet));
            Models.Add(LBFGSPipeline.Fit(TrainingSet));
            Models.Add(LBFGSMEPipeline.Fit(TrainingSet));
            //Models.Add(LGBMPipeline.Fit(TrainingSet));
            Models.Add(LSVMOPipeline.Fit(TrainingSet));
            //Models.Add(SdcaPipeline.Fit(TrainingSet));
            //Models.Add(SGDCOPipeline.Fit(TrainingSet));
            //Models.Add(SSGDLROPipeline.Fit(TrainingSet));
            Models.Add(KPipeline.Fit(TrainingSet));

            return Models;
        }
        /// <summary>
        /// Trains binary classification models built-in to Microsoft.ML on the provided TrainingSet data.
        /// </summary>
        /// <param name="mlContext">The Microsoft.ML context to perform operations in.</param>
        /// <param name="TrainingSet">The time-series dataset to train the models on.</param>
        /// <returns>List of models that can be used in performance benchmarks.</returns>
        public static List<ITransformer> BuildAndTrainBinClassModels(MLContext mlContext, IDataView TrainingSet)
        {
            List<ITransformer> Models = new List<ITransformer>();
            var Pipeline = mlContext.Transforms.Concatenate("Features", "Features");

            //var APPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron());
            var FFPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.FastForest());
            var FTPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.FastTree());
            /*var LBFGSLRPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
            var LGBMPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm());
            var LSVMPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.LinearSvm());
            var SdcaLRPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
            var SGDCPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.SgdCalibrated());
            var SSGDLRPipeline = Pipeline.Append(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression());*/
            var KPipeline = Pipeline.Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: 9));

            //Models.Add(APPipeline.Fit(TrainingSet));
            Models.Add(FFPipeline.Fit(TrainingSet));
            Models.Add(FTPipeline.Fit(TrainingSet));
            //Models.Add(LBFGSLRPipeline.Fit(TrainingSet));
            //Models.Add(LGBMPipeline.Fit(TrainingSet));
            //Models.Add(LSVMPipeline.Fit(TrainingSet));
            //Models.Add(SdcaLRPipeline.Fit(TrainingSet));
            //Models.Add(SGDCPipeline.Fit(TrainingSet));
            //Models.Add(SSGDLRPipeline.Fit(TrainingSet));
            Models.Add(KPipeline.Fit(TrainingSet));

            return Models;
        }
        /// <summary>
        /// Train regression classification models built-in to Microsoft.ML on the provided TrainingSet data.
        /// </summary>
        /// <param name="mlContext">The Microsoft.ML context to perform operations in.</param>
        /// <param name="TrainingSet">The time-series dataset to train the models on.</param>
        /// <returns>List of models that can be used in performance benchmarks.</returns>
        public static List<ITransformer> BuildAndTrainRegressionModels(MLContext mlContext, IDataView TrainingSet)
        {
            List<ITransformer> Models = new List<ITransformer>();
            var Pipeline = mlContext.Transforms.CopyColumns("Label", "Result")
                .Append(mlContext.Transforms.Concatenate("Features", "StressFeatures"));
            var FastForestPipeline = Pipeline.Append(mlContext.Regression.Trainers.FastForest());
            var FastTreePipeline = Pipeline.Append(mlContext.Regression.Trainers.FastTree());
            var FastTreeTweediePipeline = Pipeline.Append(mlContext.Regression.Trainers.FastTreeTweedie());
            var LBFGSPipeline = Pipeline.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());
            var LGBMPipeline = Pipeline.Append(mlContext.Regression.Trainers.LightGbm());
            /*var OLSOptions = new OlsTrainer.Options()
            {
                L2Regularization = 0.5f,
            };
            var OLSPipeline = Pipeline.Append(mlContext.Regression.Trainers.Ols(OLSOptions));
            */
            /*
            //TODO: These models have errors to be addressed
            var OGDOptions = new OnlineGradientDescentTrainer.Options()
            {
                NumberOfIterations = 100,
                ResetWeightsAfterXExamples = 10,
                L2Regularization = 0.25f,
                DecreaseLearningRate = true,
            };
            var OGDPipeline = Pipeline.Append(mlContext.Regression.Trainers.OnlineGradientDescent(OGDOptions));
            var SdcaPipeline = Pipeline.Append(mlContext.Regression.Trainers.Sdca());
            // TODO: Exploding/vanishing gradients causing this line to fail
            Models.Add(OGDPipeline.Fit(TrainingSet));
            // TODO: Update, Actual error preventing this from completing: https://github.com/dotnet/machinelearning-samples/issues/833
            Models.Add(SdcaPipeline.Fit(TrainingSet));
            */

            Models.Add(FastForestPipeline.Fit(TrainingSet));
            Models.Add(FastTreePipeline.Fit(TrainingSet));
            Models.Add(FastTreeTweediePipeline.Fit(TrainingSet));
            Models.Add(LBFGSPipeline.Fit(TrainingSet));
            //Models.Add(LGBMPipeline.Fit(TrainingSet));
            //Models.Add(OLSPipeline.Fit(TrainingSet));

            return Models;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="Directories"></param>
        public static List<ExtractedMultiFeatures> ArceStevensDataset(string Directory)
        {
            List<Tuple<string, string>> SubjectConditions = GetSubjectConditions(ExpFiles);
            List<ExtractedMultiFeatures> MultiFeatureSet = new List<ExtractedMultiFeatures>();
            List<Tuple<string, List<double[]>, List<string>, List<double[]>>> Datasets = LoadE4Dataset(Directory);
            foreach (var Dataset in Datasets)
            {
                if (SubjectConditions.Exists(c => int.Parse(c.Item1) == int.Parse(Dataset.Item1)))
                {
                    Tuple<string, string> SubjectCondition = SubjectConditions.Find(c => int.Parse(c.Item1) == int.Parse(Dataset.Item1));
                    int ShortestData = 1;
                    for (int i = 0; i < Dataset.Item2.Count; i++)
                    {
                        if (Dataset.Item2[i].Length / SampleRates[i] < Dataset.Item2[ShortestData].Length / SampleRates[i])
                        {
                            ShortestData = i;
                        }
                    }
                    int NumberOfSamples = Dataset.Item2[ShortestData].Length / SampleRates[ShortestData];
                    List<int> EventIndices = GetEventIndices(Dataset, SubjectCondition.Item2, NumberOfSamples);
                    for (int i = 0; i < NumberOfSamples - WindowSize; i += WindowSize)
                    {
                        List<double> SubjectFeatures = new List<double>();
                        for (int j = 0; j < WindowSize; j++)
                        {
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[0].GetSubArray(j * 96, (j + 1) * 96)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[1].GetSubArray(j * 32, (j + 1) * 32)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[2].GetSubArray(j * 32, (j + 1) * 32)));
                            SubjectFeatures.AddRange(SignalProcessing.ProcessAccSignal(Dataset.Item2[3].GetSubArray(j * 32, (j + 1) * 32)));
                        }
                        SubjectFeatures.AddRange(SignalProcessing.ProcessPpgSignal(Dataset.Item2[4].GetSubArray(i * 64, (i + WindowSize) * 64)));
                        SubjectFeatures.AddRange(SignalProcessing.ProcessEdaSignal(Dataset.Item2[5].GetSubArray(i * 4, (i + WindowSize) * 4)));
                        SubjectFeatures.AddRange(SignalProcessing.ProcessTmpSignal(Dataset.Item2[6].GetSubArray(i * 4, (i + WindowSize) * 4)));

                        uint label = (uint)EventIndices[i];

                        MultiFeatureSet.Add(new ExtractedMultiFeatures()
                        {
                            StressFeatures = SubjectFeatures.ToArray().ToFloat(),
                            Result = label,
                        });

                        if (i + WindowSize > NumberOfSamples)
                        {
                            WindowSize = NumberOfSamples - i;
                        }
                    }
                }
            }
            return MultiFeatureSet;
        }

        public static List<ExtractedMultiFeatures> CombineLabels(List<ExtractedMultiFeatures> Features, List<int> LabelsToMerge, int FinalLabel)
        {
            for (int i = 0; i < Features.Count; i++)
            {
                if (LabelsToMerge.Contains((int)Features[i].Result))
                {
                    Features[i].Result = (uint)FinalLabel;
                }
            }
            return Features;
        }

        public static List<Tuple<string, string>> GetSubjectConditions(string[] ExpFiles)
        {
            List<Tuple<string, string>> SubjectConditions = new List<Tuple<string, string>>();
            string line;
            List<string> cells;
            // Opening the file for reading
            foreach (string file in ExpFiles)
            {
                using (StreamReader stream = File.OpenText(file))
                {
                    stream.ReadLine();
                    while ((line = stream.ReadLine()) != null)
                    {
                        cells = line.Split(new char[] { ',' }).ToList();
                        SubjectConditions.Add(Tuple.Create(cells[0], cells[2]));
                    }
                }
            }
            return SubjectConditions;
        }

        public static List<int> GetEventIndices(Tuple<string, List<double[]>, List<string>, List<double[]>> Subject, string SubjectCondition, 
            int WindowCount)
        {
            List<string> DataStarts = Subject.Item3;
            List<double[]> Tags = Subject.Item4;
            List<int[]> TagIndices = new List<int[]>();
            List<int> Labels = new List<int>();

            for (int i = 0; i < DataStarts.Count; i++)
            {
                if (SampleRates[i] != 0)
                {
                    int[] indices = new int[5];
                    double StartTime = double.Parse(DataStarts[i]);
                    int PrevTimeDelta = 0;
                    for (int j = 0; j < Tags[0].Length; j++)
                    {
                        int TimeDelta = (int)(Tags[0][j] - StartTime);
                        indices[j] = (TimeDelta * SampleRates[i]) + PrevTimeDelta;
                        StartTime = Tags[0][j];
                        PrevTimeDelta = indices[j];
                    }
                    TagIndices.Add(indices);
                }
            }
            // Convert indices to seconds, assign each second a label
            int[] TestIndices = TagIndices[0].ToList().Select(x => x / 96).ToArray();

            for (int i = 0; i < WindowCount; i++)
            {
                // Before precondition
                if (i < TestIndices[0])
                {
                    Labels.Add(0);
                }
                // During precondition
                else if (TestIndices[0] <= i && i < TestIndices[1])
                {
                    Labels.Add(1);
                }
                // During condition
                else if (TestIndices[1] <= i && i < TestIndices[2])
                {
                    if (SubjectCondition == "HAI")
                        Labels.Add(2);
                    else
                        Labels.Add(3);
                }
                // During postcondition
                else if (TestIndices[2] <= i && i < TestIndices[3])
                {
                    Labels.Add(4);
                }
                // After postcondition
                else
                {
                    Labels.Add(0);
                }
            }
            return Labels;
        }

        /// <summary>
        /// Parses the 
        /// </summary>
        /// <param name="Directories"></param>
        /// <returns></returns>
        public static List<Tuple<string, List<double[]>, List<string>, List<double[]>>> LoadE4Dataset(string TopDir)
        {
            List<Tuple<string, List<double[]>, List<string>, List<double[]>>> Dataset = new List<Tuple<string, List<double[]>, List<string>, List<double[]>>>();
            string[] Directories = Directory.GetDirectories(TopDir);
            foreach (string Dir in Directories)
            {
                string[] directories = Directory.GetDirectories(Dir);
                foreach (string dir in directories)
                {
                    List<double[]> Data = new List<double[]>(7);
                    List<string> DataStart = new List<string>(7);
                    List<double[]> Labels = new List<double[]>();
                    List<double> LabelIndices = new List<double>();
                    string FolderName = Path.GetFileName(dir);
                    string[] files = Directory.GetFiles(dir, "*.csv").OrderBy(f => f).ToArray();
                    if (files.Length != 7)
                    {
                        continue;
                    }
                    foreach (string file in files)
                    {
                        if (Path.GetFileNameWithoutExtension(file) == "ACC")
                        {
                            Tuple<string, double[]> AccData = ProcessE4File(file);
                            double[] AccX = new double[AccData.Item2.Length / 3],
                                AccY = new double[AccData.Item2.Length / 3],
                                AccZ = new double[AccData.Item2.Length / 3];
                            for (int i = 0; i < AccData.Item2.Length; i += 3)
                            {
                                AccX[i / 3] = AccData.Item2[i];
                                AccY[i / 3] = AccData.Item2[i + 1];
                                AccZ[i / 3] = AccData.Item2[i + 2];
                            }
                            Data.Add(AccData.Item2);
                            Data.Add(AccX);
                            Data.Add(AccY);
                            Data.Add(AccZ);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                            DataStart.Add(AccData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "BVP")
                        {
                            Tuple<string, double[]> BvpData = ProcessE4File(file);
                            Data.Add(BvpData.Item2);
                            DataStart.Add(BvpData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "EDA")
                        {
                            Tuple<string, double[]> EdaData = ProcessE4File(file);
                            Data.Add(EdaData.Item2);
                            DataStart.Add(EdaData.Item1);
                        }
                        /*else if (Path.GetFileNameWithoutExtension(file) == "HR")
                        {
                            Tuple<string, double[]> HrData = ProcessE4File(file);
                            Data.Add(HrData.Item2);
                            DataStart.Add(HrData.Item1);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "IBI")
                        {
                            Tuple<string, double[]> IbiData = ProcessE4File(file);
                            Data.Add(IbiData.Item2);
                            DataStart.Add(IbiData.Item1);
                        }*/
                        else if (Path.GetFileNameWithoutExtension(file) == "tags")
                        {
                            Tuple<string, double[]> TagData = ProcessE4File(file, true);
                            Labels.Add(TagData.Item2);
                        }
                        else if (Path.GetFileNameWithoutExtension(file) == "TEMP")
                        {
                            Tuple<string, double[]> TmpData = ProcessE4File(file);
                            Data.Add(TmpData.Item2);
                            DataStart.Add(TmpData.Item1);
                        }
                    }
                    Dataset.Add(Tuple.Create(FolderName, Data, DataStart, Labels));
                }
            }
            return Dataset;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Filepath"></param>
        /// <param name="Tags"></param>
        /// <returns></returns>
        public static Tuple<string, double[]> ProcessE4File(string Filepath, bool Tags = false)
        {
            List<string> lines = new List<string>();
            List<double> ExtractedData = new List<double>();
            string Date = "";
            // Opening the file for reading 
            using (StreamReader stream = File.OpenText(Filepath))
            {
                if (!Tags)
                {
                    Date = stream.ReadLine();
                    Date = Date.Split(new char[] { ',' }).ToList()[0];
                }
                string temp = "";
                while ((temp = stream.ReadLine()) != null)
                {
                    lines.Add(temp);
                }
            }
            foreach (string line in lines)
            {
                List<string> tempList = line.Split(new char[] { ',' }).ToList();
                foreach (string temp in tempList)
                {
                    ExtractedData.Add(double.Parse(temp));
                }
            }
            return Tuple.Create(Date, ExtractedData.ToArray());
        }

        /// <summary>
        /// Converts multi-class feature set to binary class representation.
        /// </summary>
        /// <param name="FeatureSet">The feature set to convert.</param>
        /// <returns>Binary class representation of the input data.</returns>
        public static List<ExtractedBinFeatures> MultiToBin(List<ExtractedMultiFeatures> FeatureSet)
        {
            List<ExtractedBinFeatures> ConFeatures = new List<ExtractedBinFeatures>();
            for (int i = 0; i < FeatureSet.Count; i++)
            {
                if (FeatureSet[i].Result == 1 || FeatureSet[i].Result == 4)
                    ConFeatures.Add(new ExtractedBinFeatures()
                    {
                        Label = false,
                        Features = FeatureSet[i].StressFeatures
                    });
                else
                    ConFeatures.Add(new ExtractedBinFeatures()
                    {
                        Label = true,
                        Features = FeatureSet[i].StressFeatures,
                    });
            }
            return ConFeatures;
        }
        /// <summary>
        /// Converts multi-class feature dataset to regression class feature dataset.
        /// </summary>
        /// <param name="FeatureSet">the feature set to convert.</param>
        /// <returns>Regression class representation of the input data.</returns>
        public static List<ExtractedRegFeatures> MultiToReg(List<ExtractedMultiFeatures> FeatureSet)
        {
            List<ExtractedRegFeatures> ConFeatures = new List<ExtractedRegFeatures>();
            for (int i = 0; i < FeatureSet.Count; i++)
            {
                ConFeatures.Add(new ExtractedRegFeatures()
                {
                    Result = FeatureSet[i].Result,
                    StressFeatures = FeatureSet[i].StressFeatures,
                });
            }
            return ConFeatures;
        }
    }
}
